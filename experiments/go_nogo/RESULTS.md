# Go/No-Go run — 2026-07-24

## Setup
- Train: 192.168.5.112 × 1 GPU (rank 0)
- Infer: 192.168.5.113 × 2 GPU (ranks 1–2) → **N=2** replicas
- Payload W = 2.0 GiB fp16; iters=5 warmup=2; HCA=`mlx5_1`
- Patterns: NCCL **broadcast** (checkpoint-engine-like) vs NCCL **P2P** multi-send (fabric-lib/Awex-like)
- Artifacts: `results/run_20260724_063055/`

## Traffic matrix (train-side egress, per iter)

| Mode | train TX / iter | vs W | vs opt bound | wall / iter |
|---|---:|---:|---:|---:|
| broadcast | 2.272 GB | 1.06× | **+5.8%** | 187 ms |
| p2p | 4.546 GB | 2.12× | **+111.7%** | 373 ms |

Lower bounds used:
- with replica relay: **W** (each byte leaves train once)
- naive P2P: **N×W**

Infer-node `mlx5_1` RX (shared NIC, both ranks see same counter):
- broadcast ≈ 1.06×W → only one wire copy; 2nd GPU filled locally (NVLink/PCIe relay)
- p2p ≈ 2.12×W → two full copies over the wire

## Verdict: **GO**

P2P train egress is **111.7% above** the relay lower bound (threshold >50% → go).
Broadcast is already within **6%** of the relay bound — so the two SOTA regimes are complementary:
- broadcast ≈ optimal **bytes**, but gather/static-group/reshard costs remain
- p2p ≈ **N×** bytes, but high parallelism / no gather bottleneck

This is exactly the structural claim for RMcast (hybrid shard-exchange × replica-relay).

## How to re-run
```bash
cd /home/xiajinyi25/ymy/RL-CCL
SIZE_GB=2 ITERS=5 bash experiments/go_nogo/run_go_nogo.sh
```

## Note on fabric-lib
This go/no-go measures the **injection-byte** gap with NCCL stand-ins for the two communication structures.
Native `fabric-lib` (`pplx-garden`) still needs libfabric + GDRCopy (+ preferably their Docker image) before timing/parallelism claims; CX-7 + CUDA 13.3 already satisfy the hard NIC/CUDA gates.
