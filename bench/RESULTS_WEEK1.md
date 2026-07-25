# Week-1 Day 1–3 results (112↔113)

Environment knobs (all runs): `NCCL_PROTO=Simple NCCL_ALGO=Ring NCCL_IB_GID_INDEX=3 NCCL_SOCKET_IFNAME=bond0`; per-GPU `NCCL_IB_HCA=mlx5_1|mlx5_3`. Torch stack NCCL **2.28.9**.

## Day 1 — baselines + efficiency table

Artifacts: `bench/results/nic_baseline_day1/`, `bench/results/torch_bcast_day1/`, `analysis/lb_table.md`

| Metric | Value |
|---|---|
| ib_write_bw (min) | **92.44 Gbps** (same/cross rail all ~92.5) |
| Dual-NIC fluid cut | **0.659 s** for W=14.19 GiB |
| nccl-tests broadcast @W | algbw **9.35 GB/s** (~1.63 s) |
| torch+NCCL broadcast @W | algbw **8.50 GB/s** (~1.79 s) — ceiling for mode D stack |

Four-mode vs ceilings (from `analysis/lb_table.md`):

| Mode | wall | % dual-NIC cut | % torch NCCL |
|---|---:|---:|---:|
| A | 2.37s | 27.8% | 75.6% |
| B | 3.47s | 19.0% | 51.6% |
| C | 1.94s | 34.0% | 92.6% |
| D | 1.68s | 39.3% | **107%** |

**go/no-go:** Mode D ≥ 80% of NCCL ceiling → **PASS**.  
(D > 100% of *single-root* broadcast is expected: D uses two concurrent shard PGs / dual NIC.)

## Day 2 — concurrent NCCL groups

Artifact: `bench/results/concurrent_day2/summary.json`  
Payload **4 GiB** (40GB GPU headroom); G ∈ {1,2,4,8,16}.

| G | goodput GB/s | vs G=1 attenuation |
|---:|---:|---:|
| 1 | 11.45 | 0% |
| 2 | 10.46 | 8.6% |
| 4 | 10.45 | 8.7% |
| 8 | 10.92 | **4.6%** |
| 16 | 11.15 | 2.6% |

**go/no-go:** G=8 attenuation < 20% → **PASS**.  
Demand-class merge stays an optimization, not a redesign blocker (at least at G≤16 on this topo).

## Day 3 — TE-CCL / fluid oracle

Artifact: `experiments/teccl_oracle/results/summary.json`

| Solver | makespan / status | solve time |
|---|---|---|
| Closed-form fluid (dual-NIC) | **0.658 s** | ~µs |
| TE-CCL MILP | **blocked** (`no_gurobi`; cluster has no license / no PyPI) | — |

D / fluid ≈ **2.55×** (matches DESIGN caveat; do not claim D = LB).  
TE-CCL wall-clock comparison deferred until Gurobi is available; topology + RMcast demand patch ready under `experiments/teccl_oracle/`.

## Verdict summary

| Day | Criterion | Result |
|---|---|---|
| 1 | D > 80% NCCL ceiling | **PASS** |
| 2 | G=8 atten < 20% | **PASS** |
| 3 | solver ≫ 1.68s; D near small-scale opt | **partial** (fluid OK; MILP pending Gurobi) |
