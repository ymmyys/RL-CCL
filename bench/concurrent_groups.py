#!/usr/bin/env python3
"""
Day 2: concurrent NCCL process-group microbench.

Split W into G equal shards; each shard uses PG={owner, infer0, infer1} and
launches async broadcast concurrently. Measure wall + aggregate goodput vs G.

Ranks (same as reshard_real): 0,1 train on 112; 2,3 infer on 113.
Owners: shard i → rank (i % 2).

Env (document in run script header):
  NCCL_PROTO=Simple NCCL_ALGO=Ring NCCL_IB_GID_INDEX=3
  NCCL_SOCKET_IFNAME=bond0 per-GPU NCCL_IB_HCA=mlx5_1|mlx5_3
"""
from __future__ import annotations

import argparse
import json
import os
import statistics
import sys
import time
from pathlib import Path

import torch
import torch.distributed as dist

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "experiments" / "go_nogo"))
from ib_counters import delta, snapshot  # noqa: E402

W_BYTES_DEFAULT = 15_231_233_024  # Qwen2.5-7B BF16
HCAS = ["mlx5_1", "mlx5_3"]


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--outdir", required=True)
    p.add_argument("--w-bytes", type=int, default=W_BYTES_DEFAULT)
    p.add_argument("--groups", type=str, default="1,2,4,8,16")
    p.add_argument("--iters", type=int, default=5)
    p.add_argument("--warmup", type=int, default=2)
    p.add_argument("--channels", type=str, default="",
                    help="Optional comma list of NCCL_MAX_NCHANNELS to sweep")
    return p.parse_args()


def sync_all(device):
    torch.cuda.synchronize(device)


def barrier_gloo(gloo):
    dist.barrier(group=gloo)


def run_concurrent(bufs, owners, groups, members_list, rank, device):
    """Async broadcast each shard on its PG; wait all. Skip if rank ∉ PG."""
    works = []
    for buf, owner, pg, members in zip(bufs, owners, groups, members_list):
        if pg is None or rank not in members:
            continue
        works.append(dist.broadcast(buf, src=owner, group=pg, async_op=True))
    for w in works:
        w.wait()
    sync_all(device)


def main():
    args = parse_args()
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    dist.init_process_group(backend="nccl")
    rank = dist.get_rank()
    world = dist.get_world_size()
    assert world == 4, f"expect 4 ranks, got {world}"
    local_rank = int(os.environ.get("LOCAL_RANK", rank % 2))
    torch.cuda.set_device(local_rank)
    device = torch.device(f"cuda:{local_rank}")

    gloo = dist.new_group(backend="gloo")
    group_counts = [int(x) for x in args.groups.split(",") if x.strip()]
    channel_sweep = [None]
    if args.channels.strip():
        channel_sweep = [int(x) for x in args.channels.split(",") if x.strip()]

    results = []
    for nch in channel_sweep:
        if nch is not None:
            os.environ["NCCL_MAX_NCHANNELS"] = str(nch)
            if rank == 0:
                print(f"NCCL_MAX_NCHANNELS={nch}", flush=True)

        for G in group_counts:
            assert args.w_bytes % G == 0 or True  # allow pad
            shard = (args.w_bytes // G) // 2 * 2  # bf16 align
            # pad last if needed — keep equal shards for fairness
            shard = (args.w_bytes // G)
            shard = shard - (shard % 2)
            nbytes = shard * G
            owners = [i % 2 for i in range(G)]

            # Build PGs: {owner, 2, 3}
            pgs = []
            members_list = []
            for i in range(G):
                members = sorted({owners[i], 2, 3})
                members_list.append(members)
                pgs.append(dist.new_group(ranks=members, backend="nccl"))

            bufs = [
                torch.empty(shard // 2, device=device, dtype=torch.bfloat16)
                for _ in range(G)
            ]
            for i, buf in enumerate(bufs):
                if rank == owners[i]:
                    buf.uniform_()
                else:
                    buf.zero_()

            def once():
                run_concurrent(bufs, owners, pgs, members_list, rank, device)

            # warmup
            for _ in range(args.warmup):
                barrier_gloo(gloo)
                once()
                barrier_gloo(gloo)

            walls = []
            xmits = []
            for it in range(args.iters):
                barrier_gloo(gloo)
                sync_all(device)
                barrier_gloo(gloo)
                before = snapshot(HCAS)
                t0 = time.perf_counter()
                once()
                sync_all(device)
                barrier_gloo(gloo)
                t1 = time.perf_counter()
                after = snapshot(HCAS)
                dlt = delta(before, after)
                wall = t1 - t0
                # allreduce max wall
                t_t = torch.tensor([wall], device=device)
                dist.all_reduce(t_t, op=dist.ReduceOp.MAX)
                wall = float(t_t.item())
                xmit = sum(dlt[h]["xmit_bytes"] for h in HCAS)
                x_t = torch.tensor([float(xmit)], device=device)
                # sum train egress: only ranks 0,1 matter; gather via allreduce sum then /2?
                # Use rank0/1 host sum: allreduce SUM of local xmit, train hosts contribute both
                dist.all_reduce(x_t, op=dist.ReduceOp.SUM)
                # 112 and 113 both counted — take train-only by restricting to rank<2 contribution
                # Re-do: collect per-rank
                walls.append(wall)
                xmits.append(xmit)
                if rank == 0:
                    print(
                        f"G={G} nch={nch} iter={it} wall={wall:.4f}s "
                        f"local_xmit={xmit/1e9:.3f}GB",
                        flush=True,
                    )

            # Gather train egress properly
            local_mean_xmit = statistics.mean(xmits) if xmits else 0.0
            xt = torch.tensor(
                [local_mean_xmit if rank < 2 else 0.0], device=device
            )
            dist.all_reduce(xt, op=dist.ReduceOp.SUM)
            train_egress = float(xt.item()) / 2.0  # ranks 0 and 1 see same NIC counters

            # Actually on Linux IB counters are per-host: both local ranks see same
            # mlx5_* counters. So rank0+rank1 would double-count. Use max among train.
            xt2 = torch.tensor(
                [local_mean_xmit if rank < 2 else 0.0], device=device
            )
            dist.all_reduce(xt2, op=dist.ReduceOp.MAX)
            train_egress = float(xt2.item())

            wall_mean = statistics.mean(walls)
            wall_std = statistics.stdev(walls) if len(walls) > 1 else 0.0
            goodput = nbytes / wall_mean  # B/s payload
            row = {
                "G": G,
                "NCCL_MAX_NCHANNELS": nch,
                "payload_bytes": nbytes,
                "shard_bytes": shard,
                "wall_mean_s": wall_mean,
                "wall_std_s": wall_std,
                "goodput_GBs": goodput / 1e9,
                "train_egress_bytes_mean": train_egress,
                "train_egress_over_payload": train_egress / nbytes if nbytes else None,
                "walls": walls,
            }
            results.append(row)
            if rank == 0:
                print(json.dumps(row, indent=2), flush=True)

            dist.barrier()
            # destroy groups
            for pg in pgs:
                dist.destroy_process_group(pg)

    if rank == 0:
        # Relative to G=1 baseline (same channel setting)
        by_ch = {}
        for r in results:
            by_ch.setdefault(r["NCCL_MAX_NCHANNELS"], []).append(r)
        for ch, rows in by_ch.items():
            base = next(x for x in rows if x["G"] == 1)
            for x in rows:
                x["goodput_rel_G1"] = x["goodput_GBs"] / base["goodput_GBs"]
                x["attenuation_vs_G1"] = 1.0 - x["goodput_rel_G1"]
        out = {
            "W_bytes": args.w_bytes,
            "host": os.uname().nodename,
            "nccl_env": {
                "NCCL_PROTO": os.environ.get("NCCL_PROTO"),
                "NCCL_ALGO": os.environ.get("NCCL_ALGO"),
                "NCCL_IB_GID_INDEX": os.environ.get("NCCL_IB_GID_INDEX"),
                "NCCL_SOCKET_IFNAME": os.environ.get("NCCL_SOCKET_IFNAME"),
            },
            "rows": results,
            "go_nogo": {},
        }
        # Day2 criterion: G=8 attenuation < 20%
        for ch, rows in by_ch.items():
            r8 = next((x for x in rows if x["G"] == 8), None)
            if r8:
                out["go_nogo"][str(ch)] = {
                    "G8_attenuation": r8["attenuation_vs_G1"],
                    "pass": r8["attenuation_vs_G1"] < 0.20,
                }
        (outdir / "summary.json").write_text(json.dumps(out, indent=2))
        print("WROTE", outdir / "summary.json", flush=True)

    dist.destroy_process_group()


if __name__ == "__main__":
    main()
