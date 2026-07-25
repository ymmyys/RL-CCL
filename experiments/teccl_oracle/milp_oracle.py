"""Day-3 completion: exact time-expanded MILP oracle for the 2x2 RMcast instance.

TE-CCL-style store-and-forward formulation solved with scipy's bundled HiGHS
(no Gurobi needed). Time is discretized into epochs of length tau; per epoch a
link (i,j) moves at most cap_ij * tau chunks; a node may forward a chunk in
epoch k only if it held it before k (multicast copy allowed: one held chunk can
go out on several links in the same epoch). Objective: min #epochs until every
demand (d,c) is satisfied, found by solving feasibility MILPs for decreasing K.

Instance = the 112->113 bench: GPUs 0,1 (train, shards) -> 2,3 (infer, full W).
Cross-host edges 11.57 GB/s per (GPU,NIC) pair; intra-host PCIe 18 GB/s.
Chunks: W split into C equal pieces, chunk c owned by GPU (c < C/2 ? 0 : 1).

Usage: python milp_oracle.py [--chunks 8] [--tau 0.15] [--max-epochs 14]
Writes results/milp_oracle.json next to this file.
"""

from __future__ import annotations

import argparse
import itertools
import json
import time
from pathlib import Path

import numpy as np
from scipy.optimize import Bounds, LinearConstraint, milp

W_GB = 14.185  # GiB->GB is irrelevant here; use the bench's 14.19 GiB in GB units
NIC = 11.57    # GB/s per cross-host (gpu,nic) path
PCIE = 18.0    # GB/s intra-host

EDGES = [
    (0, 1, PCIE), (1, 0, PCIE),          # train intra-host
    (2, 3, PCIE), (3, 2, PCIE),          # infer intra-host
    (0, 2, NIC), (0, 3, NIC),            # rank0 egress (mlx5_1)
    (1, 2, NIC), (1, 3, NIC),            # rank1 egress (mlx5_3)
    (2, 0, NIC), (3, 0, NIC),            # reverse dir (unused by demand, kept honest)
    (2, 1, NIC), (3, 1, NIC),
]
N = 4


def solve_feasible(C: int, K: int, tau: float, nic_budget: bool = True) -> dict | None:
    """Feasibility MILP: can all demands complete within K epochs of length tau?

    Vars: x[e,k,c] in {0,1}  — chunk c crosses edge e during epoch k
          h[v,k,c] in {0,1}  — v holds c at the END of epoch k (k=-1: initial)
    """
    chunk_gb = W_GB / C
    owners = [0 if c < C // 2 else 1 for c in range(C)]
    E = len(EDGES)
    nx = E * K * C
    nh = N * K * C

    def xi(e: int, k: int, c: int) -> int:
        return (e * K + k) * C + c

    def hi(v: int, k: int, c: int) -> int:
        return nx + (v * K + k) * C + c

    nvar = nx + nh
    A_rows, A_lb, A_ub = [], [], []

    def add(coeffs: dict[int, float], lb: float, ub: float) -> None:
        A_rows.append(coeffs)
        A_lb.append(lb)
        A_ub.append(ub)

    held0 = lambda v, c: 1.0 if owners[c] == v else 0.0  # noqa: E731

    for k, c in itertools.product(range(K), range(C)):
        for e, (u, v, _) in enumerate(EDGES):
            # causality: u may send c in epoch k only if it held c before k
            prev = held0(u, c) if k == 0 else None
            co = {xi(e, k, c): 1.0}
            if prev is None:
                co[hi(u, k - 1, c)] = -1.0
                add(co, -np.inf, 0.0)
            else:
                add(co, -np.inf, prev)
        for v in range(N):
            # holding propagation: h[v,k] <= h[v,k-1] + sum(inbound x this epoch)
            co = {hi(v, k, c): 1.0}
            rhs = 0.0
            if k == 0:
                rhs = held0(v, c)
            else:
                co[hi(v, k - 1, c)] = -1.0
            for e, (u, w, _) in enumerate(EDGES):
                if w == v:
                    co[xi(e, k, c)] = -1.0
            add(co, -np.inf, rhs)

    # link capacity per epoch: sum_c x[e,k,c] * chunk_gb <= cap_e * tau
    for e, (_, _, cap) in enumerate(EDGES):
        for k in range(K):
            add({xi(e, k, c): chunk_gb for c in range(C)}, -np.inf, cap * tau)

    # per-GPU NIC caps: edges (0,2)+(0,3) share rank0's mlx5_1; (0,2)+(1,2)
    # share GPU2's ingress NIC. Without these the MILP could double-count NICs.
    for u in (0, 1):  # train-side egress
        for k in range(K):
            co = {xi(e, k, c): chunk_gb
                  for e, (a, b, _) in enumerate(EDGES) if a == u and b in (2, 3)
                  for c in range(C)}
            add(co, -np.inf, NIC * tau)
    for v in (2, 3):  # infer-side ingress
        for k in range(K):
            co = {xi(e, k, c): chunk_gb
                  for e, (a, b, _) in enumerate(EDGES) if b == v and a in (0, 1)
                  for c in range(C)}
            add(co, -np.inf, NIC * tau)

    # optional aggregate egress budget of the train host (cut honesty check)
    if nic_budget:
        for k in range(K):
            co = {xi(e, k, c): chunk_gb
                  for e, (u, v, _) in enumerate(EDGES) if u in (0, 1) and v in (2, 3)
                  for c in range(C)}
            add(co, -np.inf, 2 * NIC * tau)

    # demand: infer GPUs 2,3 hold every chunk at the end
    for v, c in itertools.product((2, 3), range(C)):
        add({hi(v, K - 1, c): 1.0}, 1.0, 1.0)

    # objective: minimize total link-seconds (tie-break; feasibility is the ask)
    cost = np.zeros(nvar)
    for e in range(E):
        for k in range(K):
            for c in range(C):
                cost[xi(e, k, c)] = 1.0

    A = np.zeros((len(A_rows), nvar))
    for r, co in enumerate(A_rows):
        for j, val in co.items():
            A[r, j] = val

    res = milp(
        c=cost,
        constraints=LinearConstraint(A, A_lb, A_ub),
        integrality=np.ones(nvar),
        bounds=Bounds(0, 1),
        options={"time_limit": 300},
    )
    if not res.success:
        return None
    xs = res.x[:nx].reshape(E, K, C)
    cross = float(sum(xs[e].sum() * (W_GB / C)
                      for e, (u, v, _) in enumerate(EDGES)
                      if u in (0, 1) and v in (2, 3)))
    return {"epochs": K, "makespan_s": K * tau, "cross_gb": round(cross, 3)}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--chunks", type=int, default=8)
    ap.add_argument("--tau", type=float, default=0.165)
    ap.add_argument("--max-epochs", type=int, default=16)
    args = ap.parse_args()

    chunk_gb = W_GB / args.chunks
    if chunk_gb > NIC * args.tau:
        ap.error(f"chunk ({chunk_gb:.2f} GB) exceeds per-epoch link budget "
                 f"({NIC * args.tau:.2f} GB) — raise --tau or --chunks")

    fluid_lb = W_GB / (2 * NIC)
    out = {"W_GB": W_GB, "chunks": args.chunks, "tau_s": args.tau,
           "fluid_cut_lb_s": round(fluid_lb, 4), "trials": []}

    # feasibility is monotone in K: search upward from the fluid bound,
    # first feasible K is the discrete optimum.
    best = None
    t0 = time.perf_counter()
    k_lo = max(1, int(np.ceil(fluid_lb / args.tau)))
    for K in range(k_lo, args.max_epochs + 1):
        t1 = time.perf_counter()
        r = solve_feasible(args.chunks, K, args.tau)
        dt = time.perf_counter() - t1
        status = "feasible" if r else "infeasible"
        out["trials"].append({"K": K, "status": status, "solve_s": round(dt, 2)})
        print(f"K={K:2d} ({K * args.tau:.2f}s): {status}  [{dt:.2f}s solve]")
        if r:
            best = r
            break

    out["optimal"] = best
    out["total_solve_s"] = round(time.perf_counter() - t0, 2)
    if best:
        print(f"\nMILP optimal makespan: {best['makespan_s']:.2f}s "
              f"(fluid LB {fluid_lb:.3f}s, ratio {best['makespan_s'] / fluid_lb:.2f}x)")
        print(f"cross-host bytes moved: {best['cross_gb']} GB (W={W_GB})")

    dest = Path(__file__).parent / "results" / "milp_oracle.json"
    dest.parent.mkdir(exist_ok=True)
    dest.write_text(json.dumps(out, indent=2))
    print(f"wrote {dest}")


if __name__ == "__main__":
    main()
