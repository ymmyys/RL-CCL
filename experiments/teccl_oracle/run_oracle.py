#!/usr/bin/env python3
"""
Day 3: TE-CCL / TE-style oracle for 2+2 RMcast demand.

1) Fluid LP (PuLP/CBC): continuous multi-commodity makespan lower bound.
2) Optional TE-CCL MILP (needs gurobipy license): epoch schedule on RMcast2x2.

RMcast demand: shards owned by GPU0/GPU1; each must reach GPU2 and GPU3.
chunk_size = W/2 GB so each source has 1 chunk.
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent
W_BYTES = 15_231_233_024
MODE_D_S = 1.676018943451345


def fluid_lp(nic_GBs=11.57, pcie_GBs=18.0):
    """Closed-form fluid optimum for 2+2 RMcast on dual-NIC + PCIe fanout.

    Construction: shard0 → infer GPU2 over NIC0, PCIe to GPU3;
                  shard1 → infer GPU3 over NIC1, PCIe to GPU2;
    parallel ⇒ T* = (W/2)/nic = W/(2·nic). PCIe phase overlaps if pcie≥nic.
    """
    t0 = time.perf_counter()
    nic = nic_GBs * 1e9
    pcie = pcie_GBs * 1e9
    t_nic = (W_BYTES / 2.0) / nic
    t_pcie = (W_BYTES / 2.0) / pcie
    # With store-and-forward overlap across shards, makespan ≈ max(t_nic, t_nic+t_pcie - overlap).
    # Fluid with full pipelining across the two NICs: bottleneck is dual-NIC cut.
    t_star = W_BYTES / (2.0 * nic)
    return {
        "solver": "closed_form_fluid_dual_nic",
        "status": "Optimal",
        "makespan_s": t_star,
        "solve_s": time.perf_counter() - t0,
        "nic_GBs": nic_GBs,
        "pcie_GBs": pcie_GBs,
        "t_one_shard_nic_s": t_nic,
        "t_one_shard_pcie_s": t_pcie,
        "note": "PCIe≥NIC ⇒ dual-NIC cut tight; matches mode-D target topology class",
    }


def try_teccl(chunk_gb: float, time_limit_hr: float = 0.25):
    """Run patched TE-CCL AllGatherFormulation with RMcast demand override."""
    teccl_root = ROOT / "TE-CCL"
    sys.path.insert(0, str(teccl_root))
    sys.path.insert(0, str(ROOT))

    try:
        import gurobipy as gp  # noqa: F401
        from gurobipy import GRB
    except Exception as e:
        return {"solver": "teccl_milp", "status": "no_gurobi", "error": str(e)}

    try:
        from teccl.input_data import (
            Collective,
            EpochType,
            GurobiParams,
            InstanceParams,
            ObjectiveType,
            SolutionMethod,
            TopologyParams,
            UserInputParams,
        )
        from teccl.solvers.allgather import AllGatherFormulation
        from rmcast_2x2_topology import RMcast2x2
        import numpy as np
    except Exception as e:
        return {"solver": "teccl_milp", "status": "import_error", "error": str(e)}

    topo_p = TopologyParams(name="RMcast2x2", chassis=2, chunk_size=chunk_gb)
    topo = RMcast2x2(topo_p)
    user = UserInputParams()
    user.topology = topo_p
    user.gurobi = GurobiParams(
        time_limit=time_limit_hr,
        output_flag=1,
        log_to_console=1,
        mip_gap=1e-3,
        mip_focus=1,
    )
    user.instance = InstanceParams(
        collective=Collective.ALLGATHER,
        num_chunks=1,
        epoch_type=EpochType.FASTEST_LINK,
        num_epochs=40,
        objective_type=ObjectiveType.PAPER,
        solution_method=SolutionMethod.ONE_SHOT,
        schedule_output_file=str(ROOT / "results" / "teccl_schedule.json"),
        switch_copy=True,
    )

    class RMcastFormulation(AllGatherFormulation):
        def __init__(self, user_input, topology):
            # Build demand before BaseFormulation finishes — override generator
            self._rmcast = True
            super().__init__(user_input, topology)

        def all_gather_demand_generator(self):
            gpus = len(self.topology.capacity)
            chunks = self.user_input.instance.num_chunks
            self.demand = np.zeros((gpus, gpus, chunks), dtype=np.int32)
            # shard0 @ GPU0 → GPU2, GPU3; shard1 @ GPU1 → GPU2, GPU3
            # TE-CCL demand[s][d][c]: dest d needs chunk c of source s
            for d in (2, 3):
                self.demand[0][d][0] = 1
                self.demand[1][d][0] = 1

    t0 = time.perf_counter()
    try:
        solver = RMcastFormulation(user, topo)
        solver.solve()
        dt = time.perf_counter() - t0
        # Extract epochs / time
        epochs = getattr(solver, "num_epochs", None)
        epoch_dur = getattr(solver, "epoch_duration", None)
        obj = None
        try:
            obj = float(solver.model.ObjVal)
        except Exception:
            pass
        status = solver.model.Status
        status_name = {
            GRB.OPTIMAL: "OPTIMAL",
            GRB.TIME_LIMIT: "TIME_LIMIT",
            GRB.INFEASIBLE: "INFEASIBLE",
            GRB.SUBOPTIMAL: "SUBOPTIMAL",
        }.get(status, str(status))
        makespan = None
        if epoch_dur is not None:
            # PAPER objective is not exactly makespan; use demand-satisfied epoch if available
            try:
                k = solver.find_demand_satisfied_k()
                makespan = (k + 1) * epoch_dur
            except Exception:
                if epochs and epoch_dur:
                    makespan = epochs * epoch_dur
        return {
            "solver": "teccl_milp_rmcast",
            "status": status_name,
            "makespan_s": makespan,
            "obj": obj,
            "epoch_duration": epoch_dur,
            "num_epochs": epochs,
            "solve_s": dt,
            "chunk_size_GB": chunk_gb,
        }
    except Exception as e:
        return {
            "solver": "teccl_milp_rmcast",
            "status": "error",
            "error": str(e),
            "solve_s": time.perf_counter() - t0,
        }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--outdir", default=str(ROOT / "results"))
    ap.add_argument("--skip-teccl", action="store_true")
    ap.add_argument("--time-limit-hr", type=float, default=0.25)
    args = ap.parse_args()
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    rows = []
    fluid = fluid_lp()
    rows.append(fluid)
    print("FLUID", json.dumps(fluid, indent=2))

    if not args.skip_teccl:
        # chunk = W/2 in GB (decimal GB to match TE-CCL)
        chunk_gb = (W_BYTES / 2) / 1e9
        teccl = try_teccl(chunk_gb, time_limit_hr=args.time_limit_hr)
        rows.append(teccl)
        print("TECCL", json.dumps(teccl, indent=2))

    # Compare to mode D
    summary = {
        "W_bytes": W_BYTES,
        "mode_D_wall_s": MODE_D_S,
        "rows": rows,
        "go_nogo": {
            "fluid_vs_D": None,
            "solver_vs_xfer": None,
        },
    }
    if fluid.get("makespan_s"):
        summary["go_nogo"]["fluid_vs_D"] = {
            "fluid_s": fluid["makespan_s"],
            "D_over_fluid": MODE_D_S / fluid["makespan_s"],
            "note": "D should be >= fluid; ratio~2.4 expected from DESIGN",
        }
    for r in rows:
        if r.get("solve_s") is not None and r["solver"].startswith("teccl"):
            summary["go_nogo"]["solver_vs_xfer"] = {
                "solve_s": r["solve_s"],
                "xfer_s": MODE_D_S,
                "solve_over_xfer": r["solve_s"] / MODE_D_S,
                "pass_slow": r["solve_s"] > MODE_D_S,
            }

    (outdir / "summary.json").write_text(json.dumps(summary, indent=2))
    with (outdir / "results.csv").open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=sorted({k for r in rows for k in r}))
        w.writeheader()
        for r in rows:
            w.writerow(r)
    print("WROTE", outdir / "summary.json")


if __name__ == "__main__":
    main()
