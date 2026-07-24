#!/usr/bin/env python3
"""
Four-mode Megatron TP=2 → 2×vLLM TP=1 weight reshard bench (112→113).

Ranks: 0,1 = train shards on 112; 2,3 = infer replicas on 113 (each needs full W).
Modes A/B/C/D per INSTRUCTIONS.md.
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

# local import
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "go_nogo"))
from ib_counters import delta, snapshot  # noqa: E402

from gen_traffic_matrix import (  # noqa: E402
    SHARD_BYTES,
    W_BYTES,
    build_tensor_specs,
    pack_buckets,
    verify_specs,
)


HCAS = ["mlx5_1", "mlx5_3"]


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--iters", type=int, default=5)
    p.add_argument("--warmup", type=int, default=2)
    p.add_argument("--outdir", type=str, required=True)
    p.add_argument(
        "--modes",
        type=str,
        default="A,B,C,D",
        help="Comma list / interleaved order",
    )
    return p.parse_args()


def gloo_barrier(group):
    dist.barrier(group=group)


def sync_all(device):
    torch.cuda.synchronize(device)


def alloc_buckets(buckets, device, rank, fill: bool):
    """Each rank: owned buckets as real tensors; others allocate recv-sized buffers.

    Returns dict mode buffers:
      owned: list[(owner, tensor)] in global bucket order for this rank's owned
      all_bufs: list[tensor] aligned to global buckets (for broadcast/recv)
    """
    all_bufs = []
    for b in buckets:
        t = torch.empty(b.nbytes // 2, device=device, dtype=torch.bfloat16)
        if fill and b.owner == rank:
            t.uniform_()
        else:
            t.zero_()
        all_bufs.append(t)
    return all_bufs


def measure_window(hcas, fn, gloo_group, device):
    """Run fn(); return (wall_s, ib_delta_local). Barriers use gloo only."""
    gloo_barrier(gloo_group)
    sync_all(device)
    gloo_barrier(gloo_group)
    before = snapshot(hcas)
    t0 = time.perf_counter()
    fn()
    sync_all(device)
    gloo_barrier(gloo_group)
    t1 = time.perf_counter()
    after = snapshot(hcas)
    return t1 - t0, delta(before, after)


def run_mode_A(bufs, buckets, device):
    """Serial per-shard broadcast on default process group."""
    for buf, b in zip(bufs, buckets):
        dist.broadcast(buf, src=b.owner)


def run_mode_B(bufs, buckets, rank):
    """Direct P2P: each owner sends every owned bucket to ranks 2 and 3."""
    for buf, b in zip(bufs, buckets):
        if rank == b.owner:
            works = dist.batch_isend_irecv(
                [dist.P2POp(dist.isend, buf, 2), dist.P2POp(dist.isend, buf, 3)]
            )
            for w in works:
                w.wait()
        elif rank in (2, 3):
            dist.recv(buf, src=b.owner)


def run_mode_C_phased(bufs, buckets, rank, device):
    """Phase1: 0→2, 1→3; Phase2: 2↔3 exchange. Returns (fn, times_dict)."""
    times = {"p1": 0.0, "p2": 0.0}

    def _run():
        sync_all(device)
        t0 = time.perf_counter()
        for buf, b in zip(bufs, buckets):
            dst = 2 if b.owner == 0 else 3
            if rank == b.owner:
                dist.send(buf, dst=dst)
            elif rank == dst:
                dist.recv(buf, src=b.owner)
        sync_all(device)
        times["p1"] = time.perf_counter() - t0
        t1 = time.perf_counter()
        for buf, b in zip(bufs, buckets):
            if b.owner == 0:
                if rank == 2:
                    dist.send(buf, dst=3)
                elif rank == 3:
                    dist.recv(buf, src=2)
            else:
                if rank == 3:
                    dist.send(buf, dst=2)
                elif rank == 2:
                    dist.recv(buf, src=3)
        sync_all(device)
        times["p2"] = time.perf_counter() - t1

    return _run, times


def run_mode_D(bufs, buckets, rank, pg_a, pg_b):
    """Concurrent shard broadcasts: pg_a={0,2,3} root0, pg_b={1,2,3} root1."""
    # Pair buckets by index within each owner for rough concurrency
    b0 = [(buf, b) for buf, b in zip(bufs, buckets) if b.owner == 0]
    b1 = [(buf, b) for buf, b in zip(bufs, buckets) if b.owner == 1]
    n = max(len(b0), len(b1))
    for i in range(n):
        works = []
        if i < len(b0) and rank in (0, 2, 3):
            works.append(
                dist.broadcast(b0[i][0], src=0, group=pg_a, async_op=True)
            )
        if i < len(b1) and rank in (1, 2, 3):
            works.append(
                dist.broadcast(b1[i][0], src=1, group=pg_b, async_op=True)
            )
        for w in works:
            w.wait()


def sum_xmit(ib_delta: dict) -> int:
    return sum(v.get("xmit_bytes", 0) for v in ib_delta.values())


def main():
    args = parse_args()
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    local_rank = int(os.environ["LOCAL_RANK"])
    rank = int(os.environ["RANK"])
    world = int(os.environ["WORLD_SIZE"])
    assert world == 4, f"need 4 ranks, got {world}"
    device = torch.device(f"cuda:{local_rank}")
    torch.cuda.set_device(device)

    # NCCL default PG for data; separate Gloo PG for clean barriers
    dist.init_process_group(backend="nccl", init_method="env://", device_id=device)
    gloo_group = dist.new_group(backend="gloo")

    specs = build_tensor_specs()
    summary = verify_specs(specs)
    buckets = pack_buckets(specs)
    if rank == 0:
        (outdir / "matrix_summary.json").write_text(json.dumps(summary, indent=2))
        print(f"[rank0] W={summary['W_bytes']} buckets={len(buckets)}", flush=True)

    # Subgroups for mode D (must create on all ranks; non-members get None)
    pg_a = dist.new_group(ranks=[0, 2, 3], backend="nccl")
    pg_b = dist.new_group(ranks=[1, 2, 3], backend="nccl")

    bufs = alloc_buckets(buckets, device, rank, fill=True)
    modes = [m.strip() for m in args.modes.split(",") if m.strip()]

    # Interleave: for each iter, run all modes
    results = {m: [] for m in modes}
    host = socket.gethostname()

    for it in range(args.warmup + args.iters):
        counting = it >= args.warmup
        for mode in modes:
            # re-fill sources so dests don't short-circuit
            for buf, b in zip(bufs, buckets):
                if b.owner == rank:
                    buf.uniform_()
                else:
                    buf.zero_()
            sync_all(device)
            gloo_barrier(gloo_group)

            phase_times = None
            if mode == "A":
                fn = lambda: run_mode_A(bufs, buckets, device)
            elif mode == "B":
                fn = lambda: run_mode_B(bufs, buckets, rank)
            elif mode == "C":
                fn, phase_times = run_mode_C_phased(bufs, buckets, rank, device)
            elif mode == "D":
                fn = lambda: run_mode_D(bufs, buckets, rank, pg_a, pg_b)
            else:
                raise ValueError(mode)

            if not counting:
                fn()
                sync_all(device)
                gloo_barrier(gloo_group)
                continue

            wall, ib = measure_window(HCAS, fn, gloo_group, device)
            # gather phase times for C
            pt = None
            if phase_times is not None:
                pt = {"p1": phase_times["p1"], "p2": phase_times["p2"]}

            rec = {
                "mode": mode,
                "iter": it - args.warmup,
                "rank": rank,
                "host": host,
                "wall_s": wall,
                "ib_delta": ib,
                "xmit_bytes": sum_xmit(ib),
                "phase_times": pt,
            }
            results[mode].append(rec)
            if rank == 0:
                print(
                    f"[iter {rec['iter']}] mode={mode} wall={wall:.3f}s "
                    f"xmit={rec['xmit_bytes']/1e9:.3f}GB",
                    flush=True,
                )

    # Gather all rank records to rank0
    gathered = [None] * world
    dist.all_gather_object(gathered, results)

    if rank == 0:
        # Flatten and summarize
        rows = []
        for mode in modes:
            # train node = ranks 0,1 on 112 — sum xmit across both ranks' NICs
            # Each rank reads the SAME NIC counters if they share HCAs!
            # IMPORTANT: mlx5_1 is on NUMA0 near GPU0; mlx5_3 near GPU1.
            # Both ranks on same node reading both HCAs would double-count.
            # Protocol: only use rank0's snapshot for 112 (has both HCA counters
            # system-wide), and rank2's for 113.
            per_iter = []
            mode_recs = gathered[0][mode]  # rank0 local list length = iters
            n_iters = len(mode_recs)
            for i in range(n_iters):
                r0 = gathered[0][mode][i]
                r2 = gathered[2][mode][i]
                # wall = max across ranks
                walls = [gathered[r][mode][i]["wall_s"] for r in range(4)]
                wall = max(walls)
                xmit_112 = r0["xmit_bytes"]  # both NICs on 112
                xmit_113 = r2["xmit_bytes"]
                entry = {
                    "iter": i,
                    "wall_s": wall,
                    "walls_per_rank": walls,
                    "xmit_112_bytes": xmit_112,
                    "xmit_113_bytes": xmit_113,
                    "xmit_112_over_W": xmit_112 / W_BYTES,
                    "phase": None,
                }
                if mode == "C":
                    # max phase times across ranks that participate
                    p1s = [
                        gathered[r][mode][i]["phase_times"]["p1"]
                        for r in range(4)
                        if gathered[r][mode][i]["phase_times"]
                    ]
                    p2s = [
                        gathered[r][mode][i]["phase_times"]["p2"]
                        for r in range(4)
                        if gathered[r][mode][i]["phase_times"]
                    ]
                    entry["phase"] = {
                        "p1_max": max(p1s),
                        "p2_max": max(p2s),
                        "sum": max(p1s) + max(p2s),
                        "max_overlap_lb": max(max(p1s), max(p2s)),
                    }
                per_iter.append(entry)

            walls = [e["wall_s"] for e in per_iter]
            x112 = [e["xmit_112_bytes"] for e in per_iter]
            row = {
                "mode": mode,
                "wall_mean_s": statistics.mean(walls),
                "wall_std_s": statistics.stdev(walls) if len(walls) > 1 else 0.0,
                "xmit_112_mean": statistics.mean(x112),
                "xmit_112_std": statistics.stdev(x112) if len(x112) > 1 else 0.0,
                "xmit_112_over_W_mean": statistics.mean(x112) / W_BYTES,
                "iters": per_iter,
            }
            rows.append(row)
            print(
                f"MODE {mode}: wall={row['wall_mean_s']:.3f}±{row['wall_std_s']:.3f}s  "
                f"112_egress={row['xmit_112_mean']/1e9:.3f}GB "
                f"({row['xmit_112_over_W_mean']:.3f}×W)",
                flush=True,
            )

        out = {
            "W_bytes": W_BYTES,
            "shard_bytes": SHARD_BYTES,
            "summary_matrix": summary,
            "host_train": gathered[0][modes[0]][0]["host"],
            "host_infer": gathered[2][modes[0]][0]["host"],
            "rows": rows,
            "raw_by_rank": gathered,
        }
        (outdir / "summary.json").write_text(json.dumps(out, indent=2))
        print(f"wrote {outdir / 'summary.json'}", flush=True)

    gloo_barrier(gloo_group)
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
