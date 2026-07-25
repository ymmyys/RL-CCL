"""RMcast planner v0 — DESIGN.md §4 Steps 1-4 (demand classes -> NCCL group plan).

Pure stdlib. Input: topology + source sharding + target demands (byte intervals
over the logical weight space W). Output: a transfer plan of broadcast-group ops
(cross-host) and local-copy ops (intra-host), plus per-NIC egress accounting.

v0 scope (documented limits):
  - owner selection: least-loaded-NIC greedy among holders (DP>1 ready, untested)
  - one leader per (class, remote host) implicit in the NCCL bcast group (ring
    relays intra-host); no explicit chain/tree switch yet (DESIGN §4 Step 4)
  - no fractional striping of one class across NICs (DESIGN §3 integer caveat)
"""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from dataclasses import dataclass, field

Interval = tuple[int, int]  # [lo, hi)


@dataclass
class Host:
    name: str
    gpus: list[int]
    nics: dict[str, float]  # nic name -> bandwidth (GB/s), egress
    gpu_nic: dict[int, str]  # gpu -> nic it is bound to
    intra_bw: float = 20.0  # GB/s, intra-host GPU<->GPU (PCIe/NVLink)


@dataclass
class Topology:
    hosts: dict[str, Host] = field(default_factory=dict)

    def add_host(self, host: Host) -> None:
        self.hosts[host.name] = host

    def host_of(self, gpu: int) -> str:
        for h in self.hosts.values():
            if gpu in h.gpus:
                return h.name
        raise KeyError(f"gpu {gpu} not in topology")

    def nic_of(self, gpu: int) -> tuple[str, str]:
        h = self.hosts[self.host_of(gpu)]
        return h.name, h.gpu_nic[gpu]


@dataclass(frozen=True)
class DemandClass:
    """Atomic byte set with identical (holders, consumers)."""

    intervals: tuple[Interval, ...]
    nbytes: int
    holders: frozenset[int]  # GPUs that already hold these bytes
    dsts: frozenset[int]  # GPUs that need and do not hold them


@dataclass
class BcastOp:
    root: int
    members: tuple[int, ...]  # {root} U remote consumers — never non-consumers
    nbytes: int
    nic: str  # egress NIC on the root's host
    classes: tuple[int, ...] = ()  # indices into plan.classes


@dataclass
class CopyOp:
    src: int
    dsts: tuple[int, ...]
    nbytes: int


@dataclass
class Plan:
    classes: list[DemandClass]
    bcasts: list[BcastOp]
    copies: list[CopyOp]
    nic_egress: dict[tuple[str, str], int]  # (host, nic) -> bytes

    def cross_bytes(self) -> int:
        return sum(op.nbytes for op in self.bcasts)

    def summary(self) -> dict:
        return {
            "n_classes": len(self.classes),
            "n_bcast_groups": len(self.bcasts),
            "n_local_copies": len(self.copies),
            "cross_bytes_total": self.cross_bytes(),
            "nic_egress": {f"{h}/{n}": b for (h, n), b in sorted(self.nic_egress.items())},
        }


def _covers(ivs: list[Interval], lo: int, hi: int) -> bool:
    return any(a <= lo and hi <= b for a, b in ivs)


def build_demand_classes(
    src: dict[int, list[Interval]], dst: dict[int, list[Interval]]
) -> list[DemandClass]:
    """Step 1: overlay all interval endpoints -> atoms -> merge by (holders, dsts)."""
    pts: set[int] = set()
    for ivs in list(src.values()) + list(dst.values()):
        for a, b in ivs:
            if b <= a:
                raise ValueError(f"bad interval [{a},{b})")
            pts.update((a, b))
    cuts = sorted(pts)

    groups: dict[tuple[frozenset[int], frozenset[int]], list[Interval]] = defaultdict(list)
    for lo, hi in zip(cuts, cuts[1:]):
        holders = frozenset(g for g, ivs in src.items() if _covers(ivs, lo, hi))
        needers = frozenset(g for g, ivs in dst.items() if _covers(ivs, lo, hi))
        consumers = needers - holders  # a GPU that already holds bytes needs no transfer
        if not consumers:
            continue
        if not holders:
            raise ValueError(f"bytes [{lo},{hi}) demanded but held by no source GPU")
        groups[(holders, consumers)].append((lo, hi))

    classes = []
    for (holders, consumers), ivs in groups.items():
        ivs.sort()
        merged: list[Interval] = []
        for a, b in ivs:  # coalesce adjacent atoms
            if merged and merged[-1][1] == a:
                merged[-1] = (merged[-1][0], b)
            else:
                merged.append((a, b))
        nbytes = sum(b - a for a, b in merged)
        classes.append(DemandClass(tuple(merged), nbytes, holders, consumers))
    classes.sort(key=lambda c: -c.nbytes)
    return classes


def plan(topo: Topology, src: dict[int, list[Interval]], dst: dict[int, list[Interval]]) -> Plan:
    """Steps 2-4: host contraction, NIC waterfilling, per-class group emission."""
    classes = build_demand_classes(src, dst)
    nic_load: dict[tuple[str, str], int] = defaultdict(int)
    bcasts: list[BcastOp] = []
    copies: list[CopyOp] = []

    for idx, c in enumerate(classes):
        holders = sorted(c.holders)
        holder_hosts = {topo.host_of(g) for g in holders}

        # Step 2: any consumer colocated with a holder is served intra-host —
        # this also lets a churn survivor serve its neighbours locally.
        local_by_host: dict[str, list[int]] = defaultdict(list)
        remote: list[int] = []
        for d in sorted(c.dsts):
            h = topo.host_of(d)
            (local_by_host[h] if h in holder_hosts else remote).append(d)
        for h, dlist in sorted(local_by_host.items()):
            src_gpu = next(g for g in holders if topo.host_of(g) == h)
            copies.append(CopyOp(src_gpu, tuple(dlist), c.nbytes))
        if not remote:
            continue

        # Step 3: group holders by egress NIC. A single NIC -> one owner (the
        # DP=1 case). Multiple NICs (DP replicas / striped shards) -> split the
        # class across NICs proportional to bandwidth (fluid waterfilling).
        by_nic: dict[tuple[str, str], list[int]] = defaultdict(list)
        for g in holders:
            by_nic[topo.nic_of(g)].append(g)
        nics = sorted(by_nic)
        bw = [topo.hosts[h].nics[n] for h, n in nics]
        total_bw = sum(bw)
        sent = 0
        for i, key in enumerate(nics):
            part = (c.nbytes - sent if i == len(nics) - 1
                    else int(c.nbytes * bw[i] / total_bw))
            if part <= 0:
                continue
            sent += part
            root = min(by_nic[key], key=lambda g: nic_load[topo.nic_of(g)])
            nic_load[key] += part
            # Demand-driven membership: {root} U consumers only. This is what
            # kills mode A's backflow — non-consumer source GPUs are not members.
            bcasts.append(
                BcastOp(root, tuple(sorted((root, *remote))), part, key[1], (idx,))
            )

    return Plan(classes, bcasts, copies, dict(nic_load))


# ---------------------------------------------------------------- demo inputs

GB = 1 << 30


def demo_2x2(w: int = 15_231_233_024) -> tuple[Topology, dict, dict]:
    """The 112->113 experiment: TP=2 source, two TP=1 instances on one host."""
    topo = Topology()
    topo.add_host(Host("h112", [0, 1], {"mlx5_1": 11.55, "mlx5_3": 11.55},
                       {0: "mlx5_1", 1: "mlx5_3"}, intra_bw=20.0))
    topo.add_host(Host("h113", [2, 3], {"mlx5_1": 11.55, "mlx5_3": 11.55},
                       {2: "mlx5_1", 3: "mlx5_3"}, intra_bw=18.0))
    half = w // 2
    src = {0: [(0, half)], 1: [(half, w)]}
    dst = {2: [(0, w)], 3: [(0, w)]}
    return topo, src, dst


def demo_single_nic(w: int = 15_231_233_024) -> tuple[Topology, dict, dict]:
    """Same demand, but the source host has one NIC shared by both GPUs."""
    topo = Topology()
    topo.add_host(Host("h112", [0, 1], {"mlx5_1": 11.55},
                       {0: "mlx5_1", 1: "mlx5_1"}, intra_bw=20.0))
    topo.add_host(Host("h113", [2, 3], {"mlx5_1": 11.55, "mlx5_3": 11.55},
                       {2: "mlx5_1", 3: "mlx5_3"}, intra_bw=18.0))
    half = w // 2
    src = {0: [(0, half)], 1: [(half, w)]}
    dst = {2: [(0, w)], 3: [(0, w)]}
    return topo, src, dst


def demo_tp2_tp4(w: int = 15_231_233_024) -> tuple[Topology, dict, dict]:
    """Misaligned reshard: TP=2 source -> one TP=4 instance across two hosts."""
    topo = Topology()
    topo.add_host(Host("h112", [0, 1], {"mlx5_1": 11.55, "mlx5_3": 11.55},
                       {0: "mlx5_1", 1: "mlx5_3"}, intra_bw=20.0))
    topo.add_host(Host("h113", [2, 3], {"mlx5_1": 11.55, "mlx5_3": 11.55},
                       {2: "mlx5_1", 3: "mlx5_3"}, intra_bw=18.0))
    topo.add_host(Host("h114", [4, 5], {"mlx5_1": 11.55, "mlx5_3": 11.55},
                       {4: "mlx5_1", 5: "mlx5_3"}, intra_bw=18.0))
    half, q = w // 2, w // 4
    src = {0: [(0, half)], 1: [(half, w)]}
    dst = {2: [(0, q)], 3: [(q, 2 * q)], 4: [(2 * q, 3 * q)], 5: [(3 * q, w)]}
    return topo, src, dst


DEMOS = {"2x2": demo_2x2, "single_nic": demo_single_nic, "tp2_tp4": demo_tp2_tp4}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--demo", choices=sorted(DEMOS), default="2x2")
    args = ap.parse_args()
    topo, src, dst = DEMOS[args.demo]()
    p = plan(topo, src, dst)
    print(json.dumps(p.summary(), indent=2))
    for op in p.bcasts:
        print(f"bcast root={op.root} members={op.members} "
              f"bytes={op.nbytes:,} nic={op.nic}")
    for op in p.copies:
        print(f"copy  {op.src} -> {op.dsts} bytes={op.nbytes:,}")


if __name__ == "__main__":
    main()
