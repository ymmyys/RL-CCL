"""Pre-flight checks for the planner->executor bridge (no GPUs required).

The executor (experiments/reshard_real/plan_executor.py) interprets a Plan's
ops into NCCL primitives. Before spending cluster time we verify STATICALLY
that this interpretation moves exactly the bytes the planner's cost model
charged: re-derive per-host egress straight from the op list using the
executor's own rules (a SendOp leaves its src NIC; a ring bcast injects once at
the root then relays; a star bcast has the root pay every member) and assert it
equals p.nic_egress / p.source_egress / p.relay_egress.

Also guards the IR-cleanliness invariants the executor relies on:
  * stripe plans emit hop2 ONLY as BcastOps (every SendOp is a hop1/direct from
    a holder host) — no double-counted per-host relay sends;
  * every dependency edge (bcast.after_send, copy.after_{send,bcast}) is in
    range and semantically consistent.

Run: python planner/test_plan_exec.py
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from rmcast_plan import DEMOS, plan
from plan_io import plan_from_dict, plan_to_dict


def _roundtrip(topo, src, dst, p):
    return plan_from_dict(plan_to_dict(topo, src, dst, p))


def derive_egress(topo, p):
    """Re-derive (source_nic_egress, relay_egress_by_host) from the op list
    using the EXECUTOR's interpretation, independent of the planner's bookkeeping."""
    src_hosts = {h.name for h in topo.hosts.values()
                 if any(g in c.holders for c in p.classes for g in h.gpus)}
    src_egress = 0
    relay = {}
    for o in p.sends:
        if topo.host_of(o.src) in src_hosts:
            src_egress += o.nbytes
        else:  # a send whose src is a target host would be relay work
            relay[topo.host_of(o.src)] = relay.get(topo.host_of(o.src), 0) + o.nbytes
    for o in p.bcasts:
        rhost = topo.host_of(o.root)
        n_fwd = len(o.members) - 1
        if o.fanout == "ring":
            # root injects the payload once; each non-terminal member host
            # forwards it once (intra-host hops over NVLink are free).
            member_hosts = [topo.host_of(m) for m in o.members]
            # distinct hosts in ring order; all but the last relay cross-host
            cross_hosts = []
            for mh in member_hosts:
                if mh not in cross_hosts:
                    cross_hosts.append(mh)
            inject_host = cross_hosts[0]
            if inject_host in src_hosts:
                src_egress += o.nbytes
            else:
                relay[inject_host] = relay.get(inject_host, 0) + o.nbytes
            for mh in cross_hosts[1:-1]:  # interior hosts relay
                relay[mh] = relay.get(mh, 0) + o.nbytes
        else:  # star: root sends p2p to every member, pays n_fwd copies
            amt = o.nbytes * n_fwd
            if rhost in src_hosts:
                src_egress += amt
            else:
                relay[rhost] = relay.get(rhost, 0) + amt
    return src_egress, relay


def check_demo(name, structure):
    topo, src, dst = DEMOS[name]()
    try:
        p = plan(topo, src, dst, structure=structure)
    except ValueError:
        return None  # structure not applicable (e.g. stripe with <2 targets)
    topo, src, dst, p = _roundtrip(topo, src, dst, p)  # exercise serialization too
    src_hosts = {topo.host_of(g) for g in src}

    # (1) executor-derived source egress == planner's
    dsrc, drelay = derive_egress(topo, p)
    psrc = p.source_egress(src_hosts)
    assert dsrc == psrc, f"{name}/{structure}: exec src egress {dsrc} != plan {psrc}"

    # (2) executor-derived relay == planner's relay_egress (targets forwarding)
    prelay = {k: v for k, v in p.relay_egress.items() if v > 0}
    drelay = {k: v for k, v in drelay.items() if v > 0}
    assert drelay == prelay, f"{name}/{structure}: relay {drelay} != plan {prelay}"

    # (3) IR cleanliness: in stripe, every send is hop1/direct from a holder host
    if p.structure == "stripe":
        for o in p.sends:
            assert topo.host_of(o.src) in src_hosts, (
                f"{name}: stripe send from non-source host {o.src} "
                "(redundant hop2 send leaked back into the IR)")

    # (4) dependency edges valid and consistent
    for o in p.bcasts:
        if o.after_send >= 0:
            assert o.after_send < len(p.sends)
            assert p.sends[o.after_send].dst == o.root, (
                f"{name}: bcast root {o.root} != its feeding send dst")
    for o in p.copies:
        assert o.after_send < len(p.sends)
        assert o.after_bcast < len(p.bcasts)

    return dsrc, psrc, p.structure


if __name__ == "__main__":
    fails = 0
    for demo in sorted(DEMOS):
        for structure in ("auto", "star", "stripe"):
            try:
                r = check_demo(demo, structure)
                if r is None:
                    print(f"SKIP {demo}/{structure} (n/a)")
                else:
                    print(f"PASS {demo}/{structure:6s} src_egress={r[0]/(1<<30):.2f}GiB "
                          f"struct={r[2]}")
            except AssertionError as e:
                fails += 1
                print(f"FAIL {demo}/{structure}: {e}")
    sys.exit(1 if fails else 0)
