#!/usr/bin/env python3
"""NCCL interpretive executor for rmcast Plan JSON — the planner->hardware bridge.

Replaces reshard_bench's hand-written run_mode_{A,B,C,D} with a generic
interpreter: load a Plan produced offline by `planner/rmcast_plan.plan()`
(serialized via `planner/plan_io.py`), execute its sends / bcasts / copies with
torch.distributed, and report wall time + per-host IB egress so the planner's
fluid prediction can be checked against real iron.

The whole point of routing every contender through ONE executor: the
baseline dataflows (ce-serial / p2p-direct / mode-d) are themselves expressible
as Plans, so "planner vs baseline" becomes a fair comparison with no
per-implementation constant differences.

Op -> primitive mapping
  SendOp                 point-to-point (batch_isend_irecv). hop1 owner->pivot,
                         or a direct owner->target serve.
  BcastOp fanout="ring"  dist.broadcast over a pre-created {root}Uconsumers group
                         (NCCL ring: non-terminal members relay — the star plan's
                         mode-A/D dataflow).
  BcastOp fanout="star"  root sends p2p to each member (only the root/pivot pays
                         egress — the stripe plan's rho-respecting relay).
  CopyOp                 intra-host p2p (NCCL routes over NVLink/PCIe).

Scheduling (v1): three dependency levels separated by waits.
  1 hop1     all SendOps + resident CopyOps (after_*=-1)
  2 relay    all BcastOps (ring async, then star p2p)
  3 fanout   all CopyOps that depend on a hop1/relay op
Ops within a level on disjoint NICs overlap (async / batched). Chunk-pipelining
across levels (mode C -> mode D's overlap) is the documented next step, NOT v1.

Constraints (asserted at startup):
  * plan GPU ids == global ranks (identity mapping; the demos satisfy this).
  * each rank's NCCL_IB_HCA == topology's gpu_nic for its GPU (so the planner's
    per-NIC egress plan is what actually happens on the wire).

Launch exactly like run_reshard.sh (4 ranks, one NIC per local GPU), e.g.
  ... python plan_executor.py --plan plan_2x2.json --outdir results/... \
      --iters 5 --warmup 2
"""
from __future__ import annotations

import argparse
import json
import os
import socket
import statistics
import sys
import time
from pathlib import Path

import torch
import torch.distributed as dist

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "go_nogo"))
from ib_counters import delta, snapshot  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent / "planner"))
from plan_io import read_plan  # noqa: E402

GB = 1 << 30


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--plan", required=True, help="plan JSON (from plan_io.py)")
    p.add_argument("--outdir", required=True)
    p.add_argument("--iters", type=int, default=5)
    p.add_argument("--warmup", type=int, default=2)
    return p.parse_args()


# ------------------------------------------------------------- setup helpers

def validate_pinning(topo, rank, gpu):
    """The planner assigned each shard to a NIC; that only holds if this rank's
    GPU is actually bound to that NIC. Fail loud rather than silently move bytes
    on the wrong rail (which would make the egress comparison meaningless)."""
    host = topo.host_of(gpu)
    want = topo.hosts[host].gpu_nic[gpu]
    got = os.environ.get("NCCL_IB_HCA", "")
    if want != got:
        raise RuntimeError(
            f"rank {rank} gpu {gpu}: plan expects NIC {want!r} but "
            f"NCCL_IB_HCA={got!r} — launch env disagrees with topology")


def make_ring_groups(bcasts):
    """Pre-create one process group per distinct ring-bcast member set. NCCL
    requires new_group to be called collectively on ALL ranks in the SAME order;
    non-members receive None. Returns {sorted_members_tuple: group}."""
    sets = sorted({tuple(sorted(o["members"])) for o in bcasts
                   if o.get("fanout", "ring") == "ring"})
    groups = {}
    for members in sets:  # identical iteration order on every rank
        groups[members] = dist.new_group(ranks=list(members), backend="nccl")
    return groups


def alloc(nbytes, device):
    return torch.empty(max(1, nbytes // 2), device=device, dtype=torch.bfloat16)


# ------------------------------------------------------------- plan execution

def build_buffers(p, rank, device):
    """One buffer per op this rank participates in (v1: no reuse). Keyed by
    ('send'|'bcast'|'copy', index)."""
    bufs = {}
    for i, o in enumerate(p.sends):
        if rank in (o.src, o.dst):
            bufs[("send", i)] = alloc(o.nbytes, device)
    for i, o in enumerate(p.bcasts):
        if rank in o.members:
            bufs[("bcast", i)] = alloc(o.nbytes, device)
    for i, o in enumerate(p.copies):
        if rank == o.src or rank in o.dsts:
            bufs[("copy", i)] = alloc(o.nbytes, device)
    return bufs


def _p2p(op_send, my_rank, buf, ops):
    """Append the P2POps this rank owns for a src->{dst...} movement."""
    src, dsts = op_send
    if my_rank == src:
        for d in dsts:
            ops.append(dist.P2POp(dist.isend, buf, d))
    elif my_rank in dsts:
        ops.append(dist.P2POp(dist.irecv, buf, src))


def run_plan(p, rank, bufs, ring_groups, device):
    """Execute the full plan once. Assumes buffers already allocated."""
    # -- level 1: hop1 sends + resident copies (data already held by src) ----
    ops = []
    for i, o in enumerate(p.sends):
        _p2p((o.src, [o.dst]), rank, bufs.get(("send", i)), ops)
    for i, o in enumerate(p.copies):
        if o.after_send < 0 and o.after_bcast < 0:
            _p2p((o.src, list(o.dsts)), rank, bufs.get(("copy", i)), ops)
    if ops:
        for w in dist.batch_isend_irecv(ops):
            w.wait()

    # -- level 2: relay bcasts (ring collective, then star p2p) --------------
    works = []
    for i, o in enumerate(p.bcasts):
        if o.fanout == "ring" and rank in o.members:
            g = ring_groups[tuple(sorted(o.members))]
            works.append(dist.broadcast(bufs[("bcast", i)], src=o.root,
                                        group=g, async_op=True))
    for w in works:
        w.wait()
    ops = []
    for i, o in enumerate(p.bcasts):
        if o.fanout == "star":
            members = [m for m in o.members if m != o.root]
            _p2p((o.root, members), rank, bufs.get(("bcast", i)), ops)
    if ops:
        for w in dist.batch_isend_irecv(ops):
            w.wait()

    # -- level 3: dependent intra-host fanout copies -------------------------
    ops = []
    for i, o in enumerate(p.copies):
        if o.after_send >= 0 or o.after_bcast >= 0:
            _p2p((o.src, list(o.dsts)), rank, bufs.get(("copy", i)), ops)
    if ops:
        for w in dist.batch_isend_irecv(ops):
            w.wait()


# ------------------------------------------------------------- measurement

def measure(hcas, fn, gloo, device):
    dist.barrier(group=gloo)
    torch.cuda.synchronize(device)
    dist.barrier(group=gloo)
    before = snapshot(hcas)
    t0 = time.perf_counter()
    fn()
    torch.cuda.synchronize(device)
    dist.barrier(group=gloo)
    t1 = time.perf_counter()
    return t1 - t0, delta(before, after=snapshot(hcas))


def sum_xmit(ib):
    return sum(v.get("xmit_bytes", 0) for v in ib.values())


def main():
    args = parse_args()
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    rank = int(os.environ["RANK"])
    local_rank = int(os.environ["LOCAL_RANK"])
    world = int(os.environ["WORLD_SIZE"])
    device = torch.device(f"cuda:{local_rank}")
    torch.cuda.set_device(device)

    topo, src, dst, p = read_plan(args.plan)
    total_gpus = sum(len(h.gpus) for h in topo.hosts.values())
    assert world == total_gpus, f"world={world} != plan gpus={total_gpus}"

    my_gpu = rank  # identity mapping (asserted convention)
    validate_pinning(topo, rank, my_gpu)
    my_host = topo.host_of(my_gpu)
    hcas = sorted(topo.hosts[my_host].nics)

    dist.init_process_group(backend="nccl", init_method="env://", device_id=device)
    gloo = dist.new_group(backend="gloo")

    plan_dict = json.loads(Path(args.plan).read_text())
    ring_groups = make_ring_groups(plan_dict["plan"]["bcasts"])
    bufs = build_buffers(p, rank, device)

    if rank == 0:
        print(f"[rank0] plan={args.plan} structure={p.structure} "
              f"predicted={p.predicted_s:.3f}s sends={len(p.sends)} "
              f"bcasts={len(p.bcasts)} copies={len(p.copies)}", flush=True)

    host = socket.gethostname()
    recs = []
    for it in range(args.warmup + args.iters):
        dist.barrier(group=gloo)
        if it < args.warmup:
            run_plan(p, rank, bufs, ring_groups, device)
            torch.cuda.synchronize(device)
            dist.barrier(group=gloo)
            continue
        wall, ib = measure(hcas, lambda: run_plan(p, rank, bufs, ring_groups,
                                                  device), gloo, device)
        recs.append({"iter": it - args.warmup, "rank": rank, "host": host,
                     "wall_s": wall, "xmit_bytes": sum_xmit(ib)})
        if rank == 0:
            print(f"[iter {it - args.warmup}] wall={wall:.3f}s "
                  f"xmit={sum_xmit(ib)/1e9:.3f}GB", flush=True)

    gathered = [None] * world
    dist.all_gather_object(gathered, recs)

    if rank == 0:
        src_hosts = {topo.host_of(g) for g in src}
        # one representative rank per host reads that host's system-wide HCA
        # counters; sum the source hosts for total source egress.
        rep = {}
        for h in topo.hosts.values():
            rep[h.name] = min(h.gpus)  # gpu==rank
        n = args.iters
        W = sum(c.nbytes for c in p.classes)  # total distinct bytes to move
        per_iter = []
        for i in range(n):
            walls = [gathered[r][i]["wall_s"] for r in range(world)]
            egress_by_host = {hn: gathered[rr][i]["xmit_bytes"]
                              for hn, rr in rep.items()}
            src_egress = sum(egress_by_host[h] for h in src_hosts)
            per_iter.append({"wall_s": max(walls),
                             "egress_by_host": egress_by_host,
                             "src_egress_bytes": src_egress,
                             "src_egress_over_W": src_egress / W if W else None})
        walls = [e["wall_s"] for e in per_iter]
        segr = [e["src_egress_bytes"] for e in per_iter]
        planned_src = p.source_egress(src_hosts)
        out = {
            "plan": args.plan,
            "structure": p.structure,
            "predicted_s": p.predicted_s,
            "planned_src_egress": planned_src,
            "planned_src_egress_over_W": planned_src / W if W else None,
            "wall_mean_s": statistics.mean(walls),
            "wall_std_s": statistics.stdev(walls) if len(walls) > 1 else 0.0,
            "measured_src_egress_mean": statistics.mean(segr),
            "measured_src_egress_over_W": statistics.mean(segr) / W if W else None,
            "wall_vs_predicted": statistics.mean(walls) / p.predicted_s
            if p.predicted_s else None,
            "W_bytes": W,
            "iters": per_iter,
            "raw_by_rank": gathered,
        }
        (outdir / "exec_summary.json").write_text(json.dumps(out, indent=2))
        print(f"WALL {out['wall_mean_s']:.3f}+-{out['wall_std_s']:.3f}s "
              f"(predicted {p.predicted_s:.3f}s, "
              f"ratio {out['wall_vs_predicted']:.2f}x)", flush=True)
        print(f"SRC EGRESS measured {out['measured_src_egress_over_W']:.3f}xW "
              f"vs planned {out['planned_src_egress_over_W']:.3f}xW", flush=True)
        print(f"wrote {outdir / 'exec_summary.json'}", flush=True)

    dist.barrier(group=gloo)
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
