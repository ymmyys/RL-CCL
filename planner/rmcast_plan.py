"""RMcast planner v1 — regime-aware two-hop pivot striping (CLASSICAL_FOUNDATIONS §3/§5).

Extends v0 (star broadcast per demand class) with the Regime-B structure:
when the source egress would saturate (many target hosts), each class is cut
into stripes; hop1 sends stripe_j once to pivot host j, hop2 relays it among
target hosts (pivot-rooted broadcast group, chunk-pipelined with hop1). Every
byte leaves the source exactly once (egress = W), and the relay work is spread
across target-host NICs that a direct star would leave idle.

Structure selection is not a hand-rule: the planner builds BOTH candidate
plans and picks by a fluid cost model (max over per-NIC egress/ingress loads)
— the evaluator IS the lower-bound formula family, so the choice inherits the
theory. Hosts expose `relay_frac` (rho): the fraction of egress budget the
inference host may spend relaying (KV-cache/decode traffic throttling). Hosts
with rho=0 are never pivots and are served by direct sends.

Classic identity (survey §4): hop1+hop2 == scatter(free, born-sharded) +
cross-host scatter + allgather among targets — the van de Geijn structure with
the scatter phase pre-paid by Megatron sharding.
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
    nics: dict[str, float]  # nic name -> bandwidth (GB/s), egress==ingress
    gpu_nic: dict[int, str]  # gpu -> nic it is bound to
    intra_bw: float = 20.0  # GB/s, intra-host GPU<->GPU (PCIe/NVLink)
    relay_frac: float = 1.0  # rho: share of egress available for relaying

    def egress_bw(self) -> float:
        return sum(self.nics.values())


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
class SendOp:
    """hop1: point-to-point stripe delivery (or direct serve of a rho=0 host)."""

    src: int
    dst: int
    nbytes: int
    nic: str  # egress NIC on the src host
    classes: tuple[int, ...] = ()
    stripe: int = -1  # stripe index within the class; -1 = unstriped direct


@dataclass
class BcastOp:
    root: int
    members: tuple[int, ...]  # {root} U consumers — never non-consumers
    nbytes: int
    nic: str  # egress NIC on the root's host
    classes: tuple[int, ...] = ()
    stripe: int = -1
    after_send: int = -1  # index into Plan.sends this bcast streams from
    #   (chunk-pipelined: starts as chunks of the hop1 send arrive — mode C's
    #    receive-all-then-forward serialization is explicitly NOT the contract)


@dataclass
class CopyOp:
    src: int
    dsts: tuple[int, ...]
    nbytes: int


@dataclass
class Plan:
    structure: str  # "star" | "stripe"
    classes: list[DemandClass]
    sends: list[SendOp]
    bcasts: list[BcastOp]
    copies: list[CopyOp]
    nic_egress: dict[tuple[str, str], int]  # (host, nic) -> bytes
    host_ingress: dict[str, int]  # host -> bytes received cross-host
    relay_egress: dict[str, int] = field(default_factory=dict)  # target-host forwarding bytes (rho-throttled)
    predicted_s: float = 0.0  # fluid cost-model estimate

    def cross_bytes(self) -> int:
        """Bytes injected into the fabric by op roots (v0 semantics: each op's
        payload leaves its root host once; intra-host ring relay is free)."""
        return sum(op.nbytes for op in self.sends) + sum(
            op.nbytes for op in self.bcasts
        )

    def source_egress(self, source_hosts: set[str]) -> int:
        return sum(
            b for (h, _), b in self.nic_egress.items() if h in source_hosts
        )

    def summary(self) -> dict:
        d = {
            "structure": self.structure,
            "n_classes": len(self.classes),
            "n_sends": len(self.sends),
            "n_bcast_groups": len(self.bcasts),
            "n_local_copies": len(self.copies),
            "predicted_s": round(self.predicted_s, 4),
            "nic_egress": {f"{h}/{n}": b for (h, n), b in sorted(self.nic_egress.items())},
            "host_ingress": dict(sorted(self.host_ingress.items())),
            "relay_egress": dict(sorted(self.relay_egress.items())),
        }
        return d


def _n_forward_hops(op: BcastOp) -> int:
    return max(0, len(op.members) - 1)


def _fluid_seconds(topo: Topology, nic_egress: dict, host_ingress: dict,
                   relay_egress: dict | None = None) -> float:
    """Fluid makespan estimate in seconds. Mirrors the LB formula family:
    per-NIC source egress, per-host ingress, and rho-throttled relay egress
    for forwarding done by target hosts (the work-conservation resource)."""
    t = 0.0
    for (h, n), b in nic_egress.items():
        t = max(t, b / GB / topo.hosts[h].nics[n])
    for h, b in host_ingress.items():
        t = max(t, b / GB / topo.hosts[h].egress_bw())
    for h, b in (relay_egress or {}).items():
        if b <= 0:
            continue
        budget = topo.hosts[h].relay_frac * topo.hosts[h].egress_bw()
        if budget <= 0:
            return float("inf")  # plan needs forwarding a rho=0 host cannot do
        t = max(t, b / GB / budget)
    return t


# ------------------------------------------------------------- demand classes

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
        consumers = needers - holders
        if not consumers:
            continue
        if not holders:
            raise ValueError(f"bytes [{lo},{hi}) demanded but held by no source GPU")
        groups[(holders, consumers)].append((lo, hi))

    classes = []
    for (holders, consumers), ivs in groups.items():
        ivs.sort()
        merged: list[Interval] = []
        for a, b in ivs:
            if merged and merged[-1][1] == a:
                merged[-1] = (merged[-1][0], b)
            else:
                merged.append((a, b))
        nbytes = sum(b - a for a, b in merged)
        classes.append(DemandClass(tuple(merged), nbytes, holders, consumers))
    classes.sort(key=lambda c: -c.nbytes)
    return classes


# --------------------------------------------------------------- fluid model

def _least_loaded_nic(topo: Topology, host: str, load: dict) -> str:
    h = topo.hosts[host]
    return min(h.nics, key=lambda n: load[(host, n)] / h.nics[n])


def _gpu_on_nic(topo: Topology, host: str, nic: str) -> int:
    h = topo.hosts[host]
    for g in h.gpus:
        if h.gpu_nic[g] == nic:
            return g
    return h.gpus[0]


# ---------------------------------------------------------------- structures

def _plan_star(
    topo: Topology, classes: list[DemandClass]
) -> Plan:
    """v0 structure: per class, owner-NIC-split broadcast to ALL consumers."""
    nic_egress: dict[tuple[str, str], int] = defaultdict(int)
    host_ingress: dict[str, int] = defaultdict(int)
    relay_egress: dict[str, int] = defaultdict(int)
    bcasts: list[BcastOp] = []
    copies: list[CopyOp] = []

    for idx, c in enumerate(classes):
        holders = sorted(c.holders)
        holder_hosts = {topo.host_of(g) for g in holders}

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

        by_nic: dict[tuple[str, str], list[int]] = defaultdict(list)
        for g in holders:
            by_nic[topo.nic_of(g)].append(g)
        nics = sorted(by_nic)
        bw = [topo.hosts[h].nics[n] for h, n in nics]
        total_bw = sum(bw)
        remote_hosts = sorted({topo.host_of(d) for d in remote})
        sent = 0
        for i, key in enumerate(nics):
            part = (c.nbytes - sent if i == len(nics) - 1
                    else int(c.nbytes * bw[i] / total_bw))
            if part <= 0:
                continue
            sent += part
            root = min(by_nic[key], key=lambda g: nic_egress[topo.nic_of(g)])
            nic_egress[key] += part
            for h in remote_hosts:
                host_ingress[h] += part
            # NCCL ring across multiple target hosts: every host but the ring
            # terminal forwards the payload cross-host — real relay work,
            # charged against the host's rho budget in the cost model.
            for h in remote_hosts[:-1]:
                relay_egress[h] += part
            bcasts.append(
                BcastOp(root, tuple(sorted((root, *remote))), part, key[1], (idx,))
            )

    p = Plan("star", classes, [], bcasts, copies, dict(nic_egress),
             dict(host_ingress), dict(relay_egress))
    p.predicted_s = _fluid_seconds(topo, nic_egress, host_ingress, relay_egress)
    return p


def _plan_stripe(topo: Topology, classes: list[DemandClass]) -> Plan | None:
    """Regime-B structure: KR-style two-hop mix (Kumar-Ross Case C/D rates).

    Per class with target hosts H (N=|H|): every byte is delivered either
      - RELAYED: hop1 owner->pivot j once, hop2 pivot j -> the other N-1 hosts
        (star sends; pivot pays (N-1) uploads from its rho-throttled budget), or
      - DIRECT: owner sends it to each of the N hosts (source pays N uploads).
    The relayed volume R and stripe sizes s_j come from the closed form that
    makes source-egress and relay budgets bind simultaneously (fluid optimum):
        T  = N*W / (u_S + sum_j rho_j*u_j)        [work-conservation bound]
        s_j = rho_j*u_j * T / (N-1),  R = min(W, sum_j s_j),  D = W - R
    R == W  -> regime B2: all bytes relayed, source egress = W (mode-D trait).
    R == 0  -> pure direct (rho all zero), source egress = N*W (regime A form).
    Returns None only when the structure adds nothing over star (no remote demand).
    """
    nic_egress: dict[tuple[str, str], int] = defaultdict(int)
    host_ingress: dict[str, int] = defaultdict(int)
    relay_egress: dict[str, int] = defaultdict(int)
    sends: list[SendOp] = []
    bcasts: list[BcastOp] = []
    copies: list[CopyOp] = []
    any_remote = False

    # Global work-conservation water level: relay budgets are shared across
    # ALL classes, so stripe sizing must use the fleet-wide T_wc — sizing each
    # class as if it owned the budgets oversubscribes relays by #classes.
    total_deliver = 0
    src_nics: set[tuple[str, str]] = set()
    for c in classes:
        holder_hosts = {topo.host_of(g) for g in c.holders}
        n_c = len({topo.host_of(d) for d in c.dsts} - holder_hosts)
        if n_c:
            total_deliver += c.nbytes * n_c
            src_nics.update(topo.nic_of(g) for g in c.holders)
    u_src_total = sum(topo.hosts[h].nics[n] for h, n in src_nics)

    for idx, c in enumerate(classes):
        holders = sorted(c.holders)
        holder_hosts = {topo.host_of(g) for g in holders}

        local_by_host: dict[str, list[int]] = defaultdict(list)
        remote_by_host: dict[str, list[int]] = defaultdict(list)
        for d in sorted(c.dsts):
            h = topo.host_of(d)
            (local_by_host[h] if h in holder_hosts else remote_by_host[h]).append(d)
        for h, dlist in sorted(local_by_host.items()):
            src_gpu = next(g for g in holders if topo.host_of(g) == h)
            copies.append(CopyOp(src_gpu, tuple(dlist), c.nbytes))
        if not remote_by_host:
            continue
        any_remote = True

        targets = sorted(remote_by_host)
        n = len(targets)

        owner_pool: dict[tuple[str, str], list[int]] = defaultdict(list)
        for g in holders:
            owner_pool[topo.nic_of(g)].append(g)
        u_src = sum(topo.hosts[h].nics[nname] for h, nname in owner_pool)

        def pick_owner() -> tuple[int, tuple[str, str]]:
            key = min(owner_pool,
                      key=lambda k: nic_egress[k] / topo.hosts[k[0]].nics[k[1]])
            return owner_pool[key][0], key

        def leader(h: str) -> int:
            nic = _least_loaded_nic(topo, h, nic_egress)
            return _gpu_on_nic(topo, h, nic)

        def direct_send(nbytes: int, h: str, cls: int, stripe: int = -1) -> None:
            g, key = pick_owner()
            nic_egress[key] += nbytes
            host_ingress[h] += nbytes
            dst_gpu = leader(h)
            sends.append(SendOp(g, dst_gpu, nbytes, key[1], (cls,), stripe))
            mates = [d for d in remote_by_host[h] if d != dst_gpu]
            if mates:
                copies.append(CopyOp(dst_gpu, tuple(mates), nbytes))

        # ---- closed-form split (KR Case C/D, fleet-wide water level)
        budgets = {h: topo.hosts[h].relay_frac * topo.hosts[h].egress_bw()
                   for h in targets}
        sum_budget_all = sum(
            topo.hosts[h].relay_frac * topo.hosts[h].egress_bw()
            for h in topo.hosts
            if h not in {topo.host_of(g) for g in c.holders})
        sum_budget = sum(budgets.values())
        if n < 2 or sum_budget <= 0:
            for h in targets:
                direct_send(c.nbytes, h, idx)
            continue

        # fleet T_wc, then this class's share of each pivot's relay capacity
        t_wc = total_deliver / GB / (u_src_total + sum_budget_all)
        share = (c.nbytes * n) / total_deliver
        raw = {h: budgets[h] * GB * t_wc * share / (n - 1) for h in targets}
        scale = min(1.0, c.nbytes / sum(raw.values()))
        stripes = {h: int(raw[h] * scale) for h in targets}
        # rounding remainder goes to the largest budget (keeps sum == R)
        r_total = sum(stripes.values())
        want = min(c.nbytes, int(sum(raw.values())))
        stripes[max(targets, key=lambda h: budgets[h])] += want - r_total
        direct_bytes = c.nbytes - sum(stripes.values())

        for s_i, ph in enumerate(targets):
            part = stripes[ph]
            if part <= 0:
                continue
            g, key = pick_owner()
            nic_egress[key] += part          # hop1: each relayed byte leaves src once
            host_ingress[ph] += part
            pv_gpu = leader(ph)
            send_idx = len(sends)
            sends.append(SendOp(g, pv_gpu, part, key[1], (idx,), s_i))

            others = [h for h in targets if h != ph]
            relay_egress[ph] += part * len(others)   # hop2: star from pivot
            pv_nic = topo.hosts[ph].gpu_nic[pv_gpu]
            for h in others:
                host_ingress[h] += part
                dst_gpu = leader(h)
                op = SendOp(pv_gpu, dst_gpu, part, pv_nic, (idx,), s_i)
                sends.append(op)
                mates = [d for d in remote_by_host[h] if d != dst_gpu]
                if mates:
                    copies.append(CopyOp(dst_gpu, tuple(mates), part))
            bcasts.append(BcastOp(pv_gpu, tuple([pv_gpu] +
                          [leader(h) for h in others]), part, pv_nic,
                          (idx,), s_i, send_idx))
            mates = [d for d in remote_by_host[ph] if d != pv_gpu]
            if mates:
                copies.append(CopyOp(pv_gpu, tuple(mates), part))

        if direct_bytes > 0:
            for h in targets:
                direct_send(direct_bytes, h, idx)

    if not any_remote:
        return None
    p = Plan("stripe", classes, sends, bcasts, copies,
             dict(nic_egress), dict(host_ingress), dict(relay_egress))
    p.predicted_s = _fluid_seconds(topo, nic_egress, host_ingress, relay_egress)
    return p


def plan(
    topo: Topology,
    src: dict[int, list[Interval]],
    dst: dict[int, list[Interval]],
    structure: str = "auto",
) -> Plan:
    """Build both canonical structures, pick by fluid cost model.

    structure: "auto" (default) | "star" | "stripe" (forced, for experiments).
    """
    classes = build_demand_classes(src, dst)
    star = _plan_star(topo, classes)
    if structure == "star":
        return star
    stripe = _plan_stripe(topo, classes)
    if structure == "stripe":
        if stripe is None:
            raise ValueError("striping impossible: <2 relay-capable target hosts")
        return stripe
    if stripe is not None and stripe.predicted_s < star.predicted_s:
        return stripe
    return star


# ---------------------------------------------------------------- demo inputs

GB = 1 << 30
W_QWEN = 15_231_233_024


def demo_2x2(w: int = W_QWEN) -> tuple[Topology, dict, dict]:
    """The 112->113 experiment: TP=2 source, two TP=1 instances on one host."""
    topo = Topology()
    topo.add_host(Host("h112", [0, 1], {"mlx5_1": 11.55, "mlx5_3": 11.55},
                       {0: "mlx5_1", 1: "mlx5_3"}, intra_bw=20.0))
    topo.add_host(Host("h113", [2, 3], {"mlx5_1": 11.55, "mlx5_3": 11.55},
                       {2: "mlx5_1", 3: "mlx5_3"}, intra_bw=18.0))
    half = w // 2
    return topo, {0: [(0, half)], 1: [(half, w)]}, {2: [(0, w)], 3: [(0, w)]}


def demo_single_nic(w: int = W_QWEN) -> tuple[Topology, dict, dict]:
    topo = Topology()
    topo.add_host(Host("h112", [0, 1], {"mlx5_1": 11.55},
                       {0: "mlx5_1", 1: "mlx5_1"}, intra_bw=20.0))
    topo.add_host(Host("h113", [2, 3], {"mlx5_1": 11.55, "mlx5_3": 11.55},
                       {2: "mlx5_1", 3: "mlx5_3"}, intra_bw=18.0))
    half = w // 2
    return topo, {0: [(0, half)], 1: [(half, w)]}, {2: [(0, w)], 3: [(0, w)]}


def demo_tp2_tp4(w: int = W_QWEN) -> tuple[Topology, dict, dict]:
    topo = Topology()
    for name in ("h112", "h113", "h114"):
        gpus = {"h112": [0, 1], "h113": [2, 3], "h114": [4, 5]}[name]
        topo.add_host(Host(name, gpus, {"mlx5_1": 11.55, "mlx5_3": 11.55},
                           {gpus[0]: "mlx5_1", gpus[1]: "mlx5_3"}, intra_bw=18.0))
    half, q = w // 2, w // 4
    src = {0: [(0, half)], 1: [(half, w)]}
    dst = {2: [(0, q)], 3: [(q, 2 * q)], 4: [(2 * q, 3 * q)], 5: [(3 * q, w)]}
    return topo, src, dst


def demo_fanout(n_targets: int = 4, w: int = W_QWEN,
                rho: float = 1.0) -> tuple[Topology, dict, dict]:
    """Regime-B showcase: 1 source host (2 GPU x 2 NIC) -> n single-GPU hosts,
    each needing full W. Symmetric NICs => source egress saturates at n>=2."""
    topo = Topology()
    topo.add_host(Host("hsrc", [0, 1], {"mlx5_1": 11.55, "mlx5_3": 11.55},
                       {0: "mlx5_1", 1: "mlx5_3"}, intra_bw=20.0))
    dst: dict[int, list[Interval]] = {}
    for i in range(n_targets):
        g = 2 + i
        topo.add_host(Host(f"ht{i}", [g], {"nic0": 11.55}, {g: "nic0"},
                           intra_bw=18.0, relay_frac=rho))
        dst[g] = [(0, w)]
    half = w // 2
    return topo, {0: [(0, half)], 1: [(half, w)]}, dst


DEMOS = {"2x2": demo_2x2, "single_nic": demo_single_nic,
         "tp2_tp4": demo_tp2_tp4, "fanout": demo_fanout}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--demo", choices=sorted(DEMOS), default="2x2")
    ap.add_argument("--structure", choices=["auto", "star", "stripe"], default="auto")
    args = ap.parse_args()
    topo, src, dst = DEMOS[args.demo]()
    p = plan(topo, src, dst, structure=args.structure)
    print(json.dumps(p.summary(), indent=2))
    for op in p.sends:
        tag = f" stripe={op.stripe}" if op.stripe >= 0 else ""
        print(f"send  {op.src} -> {op.dst} bytes={op.nbytes:,} nic={op.nic}{tag}")
    for op in p.bcasts:
        dep = f" after_send={op.after_send}" if op.after_send >= 0 else ""
        print(f"bcast root={op.root} members={op.members} "
              f"bytes={op.nbytes:,} nic={op.nic}{dep}")
    for op in p.copies:
        print(f"copy  {op.src} -> {op.dsts} bytes={op.nbytes:,}")


if __name__ == "__main__":
    main()
