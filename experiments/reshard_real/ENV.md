# Environment — reshard_real (112 / 113)

Captured before the four-mode bench. `nvidia-smi topo -m` on **both** nodes:

## GPU interconnect
| Node | GPU0↔GPU1 | Implication for mode C |
|---|---|---|
| 112 (xfusion-2) | **SYS** (PCIe + NUMA/UPI) | — |
| 113 (xfusion-3) | **SYS** (PCIe + NUMA/UPI) | **No NVLink** — use PCIe-only timing band (1.1–1.5s), not 0.7–0.9s |

GPU↔NIC affinity (both nodes identical):
- GPU0 ↔ mlx5_0/mlx5_1 via **PIX**; GPU1 ↔ mlx5_2/mlx5_3 via **PIX**
- Active RoCE ports: **mlx5_1**, **mlx5_3** (mlx5_0/2 DOWN)

## GPUDirect
`nvidia_peermem` loaded on 112 and 113.

## Intra-node copy (113, torch BF16 512MiB)
- GPU0→GPU1: **14.6 GB/s**
- GPU1→GPU0: **22.2 GB/s**
(asymmetry expected on SYS/UPI path)

## Cross-rail RDMA (`ib_write_bw` 8MiB, GID=3)
| Path | BW |
|---|---:|
| 112 mlx5_1 → 113 mlx5_1 (same rail, 10.99.3.2→3.3) | **92.56 Gbps** |
| 112 mlx5_1 → 113 mlx5_3 (cross rail, 10.99.3.2→3.6) | **92.56 Gbps** |

Cross-rail is full-rate — mode B dual-NIC / cross-rail paths are OK.

## NCCL pins (run_reshard.sh)
`NCCL_PROTO=Simple NCCL_ALGO=Ring NCCL_IB_GID_INDEX=3 NCCL_SOCKET_IFNAME=bond0`

**HCA binding:** per-process `NCCL_IB_HCA=mlx5_1` (local GPU0) / `mlx5_3` (local GPU1).  
Do **not** use `NCCL_IB_HCA="=mlx5_1,=mlx5_3"` — this NCCL build treats it as a single exact-match token and **drops mlx5_3** (verified). Comma list without `=` works for discovery, but per-GPU pin is what actually splits rails.
