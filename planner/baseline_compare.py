"""Baseline comparison bench — planner vs deployed-system dataflows (fluid model).

Puts every contender through the SAME fluid evaluator on the SAME scenarios:

  ce-serial   checkpoint-engine style: per-shard broadcast on one global group;
              forward traffic collapses onto one source NIC (measured mode A),
              ring relays through target hosts (needs relay budget).
  p2p-direct  fabric-lib / Awex style: every target host gets its own full copy
              straight from the owners; source egress = N*W (measured mode B).
  mode-d      parallel per-shard broadcast groups (measured mode D): source
              NICs balanced, but targets relay via the NCCL ring.
  planner     rmcast_plan v1 "auto": picks star / two-hop stripe mix by the
              cost model, rho-aware.
  LB          max{ LB_cut, LB_work } — the theory floor (CLASSICAL_FOUNDATIONS §2).

Fluid caveat: these are dataflow costs (volume/balance), not NCCL constants —
mode D's measured 1.68s vs its 0.66s fluid figure is software-stack tax, which
applies to every contender roughly equally (relative rankings were preserved
in the four-mode experiment).

Scenarios: fanout N sweep, relay-throttle rho sweep, heterogeneous NIC count,
multi-host TP=4 targets. Writes results to analysis/baseline_compare.{json,md}.

Usage: python baseline_compare.py [--out-dir ../analysis]
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import sys

sys.path.insert(0, str(Path(__file__).parent))

from rmcast_plan import (  # noqa: E402
    GB, Host, Topology, build_demand_classes, plan, _fluid_seconds,
)

W = 15_231_233_024  # Qwen2.5-7B BF16, matches all measured runs
NIC = 11.55  # GB/s goodput per 100G port (ib_write_bw measured)


# ------------------------------------------------------------------ scenarios

def scenario_fanout(n_targets: int, rho: float = 1.0, tgt_nics: int = 1,
                    src_nics: int = 2) -> tuple[Topology, dict, dict]:
    """1 source host (2 GPUs) -> N single-instance target hosts, full W each."""
    topo = Topology()
    nics = {f"s{i}": NIC for i in range(src_nics)}
    gpu_nic = {0: "s0", 1: f"s{min(1, src_nics - 1)}"}
    topo.add_host(Host("hsrc", [0, 1], nics, gpu_nic, intra_bw=20.0))
    dst: dict[int, list] = {}
    for i in range(n_targets):
        g = 2 + i
        tnics = {f"t{j}": NIC for j in range(tgt_nics)}
        topo.add_host(Host(f"ht{i}", [g], tnics, {g: "t0"},
                           intra_bw=18.0, relay_frac=rho))
        dst[g] = [(0, W)]
    half = W // 2
    return topo, {0: [(0, half)], 1: [(half, W)]}, dst


def scenario_tp4_hosts(n_instances: int, rho: float = 1.0
                       ) -> tuple[Topology, dict, dict]:
    """TP=2 source -> N instances, each TP=4 across 2 dual-GPU hosts.
    Misaligned reshard: quarters, each quarter wanted by one GPU per instance."""
    topo = Topology()
    topo.add_host(Host("hsrc", [0, 1], {"s0": NIC, "s1": NIC},
                       {0: "s0", 1: "s1"}, intra_bw=20.0))
    dst: dict[int, list] = {}
    g = 2
    q = W // 4
    for i in range(n_instances):
        for hpart in range(2):  # instance spans two hosts
            name = f"ht{i}_{hpart}"
            gpus = [g, g + 1]
            topo.add_host(Host(name, gpus, {"t0": NIC, "t1": NIC},
                               {g: "t0", g + 1: "t1"}, intra_bw=18.0,
                               relay_frac=rho))
            dst[g] = [(2 * hpart * q, (2 * hpart + 1) * q)]
            dst[g + 1] = [((2 * hpart + 1) * q, (2 * hpart + 2) * q or W)]
            g += 2
    half = W // 2
    return topo, {0: [(0, half)], 1: [(half, W)]}, dst


# ------------------------------------------------------------------ baselines
# Each baseline builds (nic_egress, host_ingress, relay_egress) load maps for
# the same demand classes, then is scored by the shared fluid evaluator.

def _class_split(topo, classes):
    """Common helper: per class, (owner nic keys sorted, remote target hosts)."""
    for c in classes:
        holders = sorted(c.holders)
        holder_hosts = {topo.host_of(g) for g in holders}
        remote_hosts = sorted({topo.host_of(d) for d in c.dsts
                               if topo.host_of(d) not in holder_hosts})
        owner_nics = sorted({topo.nic_of(g) for g in holders})
        yield c, owner_nics, remote_hosts


def eval_ce_serial(topo, classes) -> dict:
    """Global-group serial broadcast (measured mode A): every shard's forward
    hop leaves through ONE source NIC (the ring's single cross-host edge);
    target hosts chain-relay (ring), each non-terminal forwards once."""
    nic_e, host_i, relay_e = {}, {}, {}
    first_nic = None
    for c, owner_nics, remote in _class_split(topo, classes):
        if not remote:
            continue
        if first_nic is None:
            first_nic = owner_nics[0]
        nic_e[first_nic] = nic_e.get(first_nic, 0) + c.nbytes
        for h in remote:
            host_i[h] = host_i.get(h, 0) + c.nbytes
        for h in remote[:-1]:
            relay_e[h] = relay_e.get(h, 0) + c.nbytes
    return {"nic_egress": nic_e, "host_ingress": host_i, "relay_egress": relay_e,
            "note": "single forward rail (ring topology of the global group)"}


def eval_p2p_direct(topo, classes) -> dict:
    """fabric-lib/Awex dataflow: owners write a full copy to every target host;
    egress = N*W spread across owner NICs; zero target relay."""
    nic_e, host_i = {}, {}
    for c, owner_nics, remote in _class_split(topo, classes):
        if not remote:
            continue
        total = c.nbytes * len(remote)
        bws = [topo.hosts[h].nics[n] for h, n in owner_nics]
        sum_bw = sum(bws)
        for (key, bw) in zip(owner_nics, bws):
            nic_e[key] = nic_e.get(key, 0) + int(total * bw / sum_bw)
        for h in remote:
            host_i[h] = host_i.get(h, 0) + c.nbytes
    return {"nic_egress": nic_e, "host_ingress": host_i, "relay_egress": {},
            "note": "no relay; source pays N copies"}


def eval_mode_d(topo, classes) -> dict:
    """Per-shard concurrent broadcast groups (measured mode D): source NICs
    waterfilled, each group's ring relays through the target hosts."""
    nic_e, host_i, relay_e = {}, {}, {}
    for c, owner_nics, remote in _class_split(topo, classes):
        if not remote:
            continue
        bws = [topo.hosts[h].nics[n] for h, n in owner_nics]
        sum_bw = sum(bws)
        for (key, bw) in zip(owner_nics, bws):
            nic_e[key] = nic_e.get(key, 0) + int(c.nbytes * bw / sum_bw)
        for h in remote:
            host_i[h] = host_i.get(h, 0) + c.nbytes
        for h in remote[:-1]:
            relay_e[h] = relay_e.get(h, 0) + c.nbytes
    return {"nic_egress": nic_e, "host_ingress": host_i, "relay_egress": relay_e,
            "note": "balanced source NICs; ring relay through targets"}


def lower_bound(topo, classes, src_hosts: set[str]) -> float:
    """max{LB_cut(three canonical cuts), LB_work} in seconds (fluid)."""
    lb = 0.0
    # (a) per-owner-NIC egress cut: bytes that MUST leave via this NIC if all
    # its holders' bytes have no other holder elsewhere (DP=1 here).
    per_nic: dict[tuple, int] = {}
    total_deliver = 0
    for c in classes:
        holder_nics = sorted({topo.nic_of(g) for g in c.holders})
        remote = {topo.host_of(d) for d in c.dsts} - \
                 {topo.host_of(g) for g in c.holders}
        if not remote:
            continue
        total_deliver += c.nbytes * len(remote)
        if len(holder_nics) == 1:
            k = holder_nics[0]
            per_nic[k] = per_nic.get(k, 0) + c.nbytes
    for (h, n), b in per_nic.items():
        lb = max(lb, b / GB / topo.hosts[h].nics[n])
    # aggregated source egress cut: each demanded byte crosses >= once
    w_cross = sum(c.nbytes for c in classes
                  if {topo.host_of(d) for d in c.dsts} -
                     {topo.host_of(g) for g in c.holders})
    u_src = sum(topo.hosts[h].egress_bw() for h in src_hosts)
    lb = max(lb, w_cross / GB / u_src)
    # (c) per-target ingress cut
    need: dict[str, int] = {}
    for c in classes:
        for h in {topo.host_of(d) for d in c.dsts}:
            if h not in src_hosts:
                need[h] = need.get(h, 0) + c.nbytes
    for h, b in need.items():
        lb = max(lb, b / GB / topo.hosts[h].egress_bw())
    # (work) N copies delivered / total effective upload
    u_relay = sum(topo.hosts[h].relay_frac * topo.hosts[h].egress_bw()
                  for h in topo.hosts if h not in src_hosts)
    lb = max(lb, total_deliver / GB / (u_src + u_relay))
    return lb


CONTENDERS = ("ce-serial", "p2p-direct", "mode-d", "planner")


def run_scenario(name: str, topo, src, dst) -> dict:
    classes = build_demand_classes(src, dst)
    src_hosts = {topo.host_of(g) for g in src}
    row: dict = {"scenario": name, "lb_s": round(lower_bound(topo, classes, src_hosts), 3)}
    loads = {
        "ce-serial": eval_ce_serial(topo, classes),
        "p2p-direct": eval_p2p_direct(topo, classes),
        "mode-d": eval_mode_d(topo, classes),
    }
    for k, ld in loads.items():
        t = _fluid_seconds(topo, ld["nic_egress"], ld["host_ingress"],
                           ld["relay_egress"])
        eg = sum(b for (h, _), b in ld["nic_egress"].items() if h in src_hosts)
        row[k] = {"t_s": round(t, 3) if t != float("inf") else "inf",
                  "egress_xW": round(eg / W, 2)}
    p = plan(topo, src, dst)
    eg = p.source_egress(src_hosts)
    row["planner"] = {"t_s": round(p.predicted_s, 3),
                      "egress_xW": round(eg / W, 2),
                      "structure": p.structure}
    lb = row["lb_s"]
    row["planner"]["vs_lb"] = round(p.predicted_s / lb, 2) if lb > 0 else None
    return row


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-dir", default=str(Path(__file__).parent.parent / "analysis"))
    args = ap.parse_args()

    rows = []
    # 1) fanout N sweep, full relay budget
    for n in (2, 4, 8, 16):
        topo, src, dst = scenario_fanout(n)
        rows.append(run_scenario(f"fanout N={n} rho=1.0", topo, src, dst))
    # 2) rho sweep at N=8 (the agentic-RL KV-cache throttle axis)
    for rho in (1.0, 0.5, 0.2, 0.0):
        topo, src, dst = scenario_fanout(8, rho=rho)
        rows.append(run_scenario(f"fanout N=8 rho={rho}", topo, src, dst))
    # 3) fat source (4 NICs) vs skinny targets: regime-A territory
    topo, src, dst = scenario_fanout(4, rho=1.0, src_nics=4)
    rows.append(run_scenario("fanout N=4 fat-src(4NIC)", topo, src, dst))
    # 4) dual-NIC targets (the measured 2x2 shape, scaled out): exposes the
    # ce-serial single-rail collapse that target-ingress-bound scenarios hide
    for n in (2, 8):
        topo, src, dst = scenario_fanout(n, rho=1.0, tgt_nics=2)
        rows.append(run_scenario(f"fanout N={n} 2NIC-tgt", topo, src, dst))
    # 5) misaligned TP=4 instances across hosts
    for n in (2, 4):
        topo, src, dst = scenario_tp4_hosts(n)
        rows.append(run_scenario(f"tp2->tp4 x{n} inst", topo, src, dst))

    out = Path(args.out_dir)
    out.mkdir(exist_ok=True)
    (out / "baseline_compare.json").write_text(json.dumps(rows, indent=2))

    lines = [
        "# Planner vs baselines — fluid dataflow comparison",
        "",
        f"W = {W / GB:.2f} GiB (Qwen2.5-7B BF16), NIC goodput {NIC} GB/s.",
        "Times are fluid-model seconds (dataflow volume/balance only — NCCL",
        "stack tax excluded; measured mode D carries ~2.5x constant on 2x2).",
        "",
        "| Scenario | LB | ce-serial | p2p-direct | mode-d | planner | struct | vs LB |",
        "|---|---:|---:|---:|---:|---:|---|---:|",
    ]
    for r in rows:
        lines.append(
            f"| {r['scenario']} | {r['lb_s']} "
            f"| {r['ce-serial']['t_s']} ({r['ce-serial']['egress_xW']}xW) "
            f"| {r['p2p-direct']['t_s']} ({r['p2p-direct']['egress_xW']}xW) "
            f"| {r['mode-d']['t_s']} ({r['mode-d']['egress_xW']}xW) "
            f"| **{r['planner']['t_s']}** ({r['planner']['egress_xW']}xW) "
            f"| {r['planner']['structure']} | {r['planner']['vs_lb']} |"
        )
    (out / "baseline_compare.md").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
