# Day 3 — TE-CCL / fluid oracle for 2+2 RMcast

## What we run

1. **Closed-form fluid** (`run_oracle.py`): dual-NIC cut `T* = W/(2·nic)`.
2. **TE-CCL MILP** (optional): clone upstream into `TE-CCL/` (gitignored),
   custom topology `rmcast_2x2_topology.py`, demand override in `run_oracle.py`.
   Requires `gurobipy` + Gurobi license. This cluster currently reports
   `no_gurobi`.

```bash
git clone --depth 1 https://github.com/microsoft/TE-CCL.git experiments/teccl_oracle/TE-CCL
```

## Run

```bash
python experiments/teccl_oracle/run_oracle.py --outdir experiments/teccl_oracle/results
# skip TE-CCL attempt:
python experiments/teccl_oracle/run_oracle.py --skip-teccl
```

## Outputs

- `results/summary.json`, `results/results.csv`
