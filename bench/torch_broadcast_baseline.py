#!/usr/bin/env python3
"""Torch/NCCL broadcast baseline on same 4-rank stack as reshard_real mode A/D.

Reports algbw = W/wall (GB/s) which is the apples-to-apples ceiling for mode D.
"""
from __future__ import annotations

import argparse
import json
import os
import statistics
import time
from pathlib import Path

import torch
import torch.distributed as dist

W_BYTES = 15_231_233_024


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--outdir", required=True)
    ap.add_argument("--bytes", type=int, default=W_BYTES)
    ap.add_argument("--iters", type=int, default=5)
    ap.add_argument("--warmup", type=int, default=2)
    ap.add_argument("--root", type=int, default=0)
    args = ap.parse_args()

    dist.init_process_group(backend="nccl")
    rank = dist.get_rank()
    local = int(os.environ.get("LOCAL_RANK", 0))
    torch.cuda.set_device(local)
    device = torch.device(f"cuda:{local}")
    n = args.bytes // 2
    buf = torch.empty(n, device=device, dtype=torch.bfloat16)
    if rank == args.root:
        buf.uniform_()

    def once():
        dist.broadcast(buf, src=args.root)
        torch.cuda.synchronize(device)

    gloo = dist.new_group(backend="gloo")
    for _ in range(args.warmup):
        dist.barrier(group=gloo)
        once()
        dist.barrier(group=gloo)

    walls = []
    for _ in range(args.iters):
        dist.barrier(group=gloo)
        torch.cuda.synchronize(device)
        dist.barrier(group=gloo)
        t0 = time.perf_counter()
        once()
        dist.barrier(group=gloo)
        t1 = time.perf_counter()
        t = torch.tensor([t1 - t0], device=device)
        dist.all_reduce(t, op=dist.ReduceOp.MAX)
        walls.append(float(t.item()))

    if rank == 0:
        mean = statistics.mean(walls)
        out = {
            "bytes": args.bytes,
            "wall_mean_s": mean,
            "wall_std_s": statistics.stdev(walls) if len(walls) > 1 else 0.0,
            "algbw_GBs": (args.bytes / mean) / 1e9,
            "busbw_GBs": (args.bytes / mean) / 1e9,  # same for single-root broadcast goodput
            "walls": walls,
            "stack": "torch+NCCL (same as reshard_real)",
            "nccl_env": {
                k: os.environ.get(k)
                for k in (
                    "NCCL_PROTO",
                    "NCCL_ALGO",
                    "NCCL_IB_GID_INDEX",
                    "NCCL_IB_HCA",
                    "NCCL_SOCKET_IFNAME",
                )
            },
        }
        Path(args.outdir).mkdir(parents=True, exist_ok=True)
        (Path(args.outdir) / "torch_broadcast.json").write_text(json.dumps(out, indent=2))
        print(json.dumps(out, indent=2), flush=True)
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
