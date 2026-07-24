#!/usr/bin/env python3
"""
Go/No-Go microbench: reshard-multicast injection vs lower bound.

Compares two SOTA-style weight update patterns on a 1-train + N-infer topology:
  broadcast — NCCL Broadcast (checkpoint-engine style; implies replica relay)
  p2p       — NCCL Send to each infer rank (fabric-lib / Awex style; N×W egress)

Measures IB port TX/RX bytes on every node and reports gap vs cut lower bounds.

Topology (default, 2 nodes):
  rank 0 on 112 = train / weight source (holds full fake weight tensor W)
  ranks 1..N on 113 = inference replicas (each needs a full copy of W)

Usage (via run_go_nogo.sh):
  mpirun ... python weight_xfer_bench.py --mode broadcast --size-gb 2
"""
from __future__ import annotations

import argparse
import json
import os
import socket
import time
from pathlib import Path

import torch
import torch.distributed as dist

from ib_counters import delta, snapshot


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--mode", choices=["broadcast", "p2p", "both"], default="both")
    p.add_argument("--size-gb", type=float, default=2.0, help="Fake weight size in GiB (fp16 bytes)")
    p.add_argument("--iters", type=int, default=5)
    p.add_argument("--warmup", type=int, default=2)
    p.add_argument("--hca", type=str, default=os.environ.get("NCCL_IB_HCA", "mlx5_1"))
    p.add_argument("--outdir", type=str, default="")
    p.add_argument("--dtype", choices=["fp16", "bf16", "fp32"], default="fp16")
    return p.parse_args()


def dtype_of(name: str) -> torch.dtype:
    return {"fp16": torch.float16, "bf16": torch.bfloat16, "fp32": torch.float32}[name]


def elem_nbytes(dt: torch.dtype) -> int:
    return torch.tensor([], dtype=dt).element_size()


def barrier():
    dist.barrier()


def run_broadcast(buf: torch.Tensor, src: int = 0):
    dist.broadcast(buf, src=src)


def run_p2p(buf: torch.Tensor, src: int = 0):
    rank = dist.get_rank()
    world = dist.get_world_size()
    if rank == src:
        reqs = [dist.isend(buf, dst=d) for d in range(world) if d != src]
        for r in reqs:
            r.wait()
    else:
        dist.recv(buf, src=src)


def measure_mode(
    mode: str,
    buf: torch.Tensor,
    iters: int,
    warmup: int,
    hca: str,
) -> dict:
    rank = dist.get_rank()
    world = dist.get_world_size()
    fn = run_broadcast if mode == "broadcast" else run_p2p

    # Warmup (not counted in IB delta)
    for _ in range(warmup):
        fn(buf)
    torch.cuda.synchronize()
    barrier()

    before = snapshot([hca] if hca else None)
    t0 = time.perf_counter()
    for _ in range(iters):
        fn(buf)
    torch.cuda.synchronize()
    barrier()
    t1 = time.perf_counter()
    after = snapshot([hca] if hca else None)

    d = delta(before, after)
    host = socket.gethostname()
    nbytes = buf.numel() * buf.element_size()
    return {
        "mode": mode,
        "rank": rank,
        "world_size": world,
        "host": host,
        "hca": hca,
        "payload_bytes": nbytes,
        "iters": iters,
        "wall_s": t1 - t0,
        "wall_s_per_iter": (t1 - t0) / max(iters, 1),
        "ib_delta": d,
        "xmit_bytes": d.get(hca, {}).get("xmit_bytes", 0),
        "rcv_bytes": d.get(hca, {}).get("rcv_bytes", 0),
    }


def main():
    args = parse_args()
    local_rank = int(os.environ.get("LOCAL_RANK", os.environ.get("OMPI_COMM_WORLD_LOCAL_RANK", "0")))
    device = torch.device(f"cuda:{local_rank}")
    torch.cuda.set_device(device)

    # Prefer env:// (RANK/WORLD_SIZE/MASTER_ADDR/MASTER_PORT from launcher)
    # device_id is required when global rank != local GPU index (multi-node).
    if "RANK" in os.environ and "WORLD_SIZE" in os.environ:
        dist.init_process_group(backend="nccl", init_method="env://", device_id=device)
    else:
        dist.init_process_group(backend="nccl", device_id=device)
    rank = dist.get_rank()
    world = dist.get_world_size()
    assert world >= 2, "need >=2 ranks (1 train + >=1 infer)"

    dt = dtype_of(args.dtype)
    nbytes = int(args.size_gb * (1024**3))
    nelem = nbytes // elem_nbytes(dt)
    # Align to 1024 elems for clean sizing
    nelem = (nelem // 1024) * 1024
    nbytes = nelem * elem_nbytes(dt)

    buf = torch.empty(nelem, device=device, dtype=dt)
    if rank == 0:
        buf.uniform_()
    else:
        buf.zero_()
    torch.cuda.synchronize()
    barrier()

    modes = ["broadcast", "p2p"] if args.mode == "both" else [args.mode]
    results = []
    for mode in modes:
        # Re-init content so P2P/broadcast both move real data
        if rank == 0:
            buf.uniform_()
        else:
            buf.zero_()
        torch.cuda.synchronize()
        barrier()
        r = measure_mode(mode, buf, args.iters, args.warmup, args.hca)
        results.append(r)
        if rank == 0:
            print(
                f"[rank0] {mode}: wall={r['wall_s']:.3f}s "
                f"({r['wall_s_per_iter']*1e3:.1f} ms/iter) "
                f"xmit={r['xmit_bytes']/1e9:.3f} GB rcv={r['rcv_bytes']/1e9:.3f} GB",
                flush=True,
            )

    # Gather all rank reports to rank0
    gathered = [None] * world
    dist.all_gather_object(gathered, results)

    if rank == 0:
        out = {
            "payload_bytes": nbytes,
            "payload_gib": nbytes / (1024**3),
            "iters": args.iters,
            "warmup": args.warmup,
            "world_size": world,
            "n_infer": world - 1,
            "dtype": args.dtype,
            "hca": args.hca,
            "ranks": gathered,
            # Cut lower bounds (train-side egress, per iter, bytes)
            "lower_bound": {
                "with_relay_bytes_per_iter": nbytes,  # each byte leaves train once
                "p2p_naive_bytes_per_iter": nbytes * (world - 1),  # N×W
            },
        }
        outdir = args.outdir or os.environ.get("OUTDIR", "")
        if outdir:
            Path(outdir).mkdir(parents=True, exist_ok=True)
            path = Path(outdir) / "raw_results.json"
            path.write_text(json.dumps(out, indent=2))
            print(f"wrote {path}", flush=True)
        else:
            print(json.dumps(out, indent=2), flush=True)

    barrier()
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
