# Lower-bound / efficiency table (Day 1)

- W = 15231233024 B (14.19 GiB)
- ib_write_bw (min observed) = **92.44 Gbps** → 11.55 GB/s per NIC
- Dual-NIC cut (fluid) T ≥ W/(2·nic) = **0.659 s**
- Single-NIC cut T ≥ W/nic = **1.318 s**
- ib map: `{'cross_rail_1to3': 92.53, 'same_rail_1': 92.5, 'same_rail_3': 92.44}`
- NCCL broadcast ceiling (torch+NCCL) @W: algbw/busbw=**8.496 GB/s**, time≈1.793s

Efficiency uses payload goodput `W/wall` over the NCCL ceiling (torch+NCCL; preferred = torch stack matching mode D).

| Mode | wall_s | goodput_GBs (W/t) | % of dual-NIC cut | % of NCCL busbw |
|---|---:|---:|---:|---:|
| A | 2.372 | 6.42 | 27.8% | 75.6% |
| B | 3.474 | 4.38 | 19.0% | 51.6% |
| C | 1.937 | 7.86 | 34.0% | 92.6% |
| D | 1.676 | 9.09 | 39.3% | 107.0% |

## Day1 go/no-go
- Mode D goodput / NCCL busbw = **107.0%** (threshold 80%) → **PASS**
- D can exceed *single-root* broadcast: mode D runs two concurrent shard PGs (dual NIC); the torch baseline is one root×full W.

> Note: dual-NIC fluid cut is link-capacity upper bound (~0.66s); NCCL ceiling is the reachable collective stack. Do **not** claim D reaches LB_cut — D is only 39% of dual-NIC cut.
