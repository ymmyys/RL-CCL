"""Pure-Python fluid path LP for 2+2 RMcast (no PuLP dependency).

Uses a tiny revised simplex on the equality-form path LP.
"""
from __future__ import annotations

import time
from typing import Dict, List, Tuple

W_BYTES = 15_231_233_024


def _simplex(A, b, c, max_iter=10000):
    """Maximize c·x s.t. A x = b, x≥0. A is m×n list-of-lists. Returns x or None."""
    import copy

    m, n = len(A), len(A[0])
    # Add slack? already equality. Use two-phase / big-M with artificial vars.
    # Tableau: [A | I | b] with artificials; then optimize.
    art = m
    N = n + art
    T = [row[:] + [1.0 if i == j else 0.0 for j in range(art)] + [b[i]] for i, row in enumerate(A)]
    # Phase 1: minimize sum artificials ≡ maximize -sum art
    # basis = artificial columns
    basis = list(range(n, N))
    # obj row for phase1: -sum of artificial rows
    obj = [0.0] * (N + 1)
    for i in range(m):
        for j in range(N + 1):
            obj[j] -= T[i][j]
    T.append(obj)

    def pivot(col, row):
        piv = T[row][col]
        T[row] = [v / piv for v in T[row]]
        for i in range(len(T)):
            if i == row:
                continue
            factor = T[i][col]
            T[i] = [T[i][j] - factor * T[row][j] for j in range(N + 1)]
        basis[row] = col

    def run(phase_obj_is_last=True):
        for _ in range(max_iter):
            # enter: most positive in obj (we stored -obj for max)
            objrow = T[-1]
            enters = [(j, objrow[j]) for j in range(N) if objrow[j] > 1e-12]
            if not enters:
                return True
            col = max(enters, key=lambda x: x[1])[0]
            # leave: min ratio
            ratios = []
            for i in range(m):
                if T[i][col] > 1e-12:
                    ratios.append((i, T[i][-1] / T[i][col]))
            if not ratios:
                return False  # unbounded
            row = min(ratios, key=lambda x: x[1])[0]
            pivot(col, row)
        return False

    if not run():
        return None
    # Check artificials zero
    if abs(T[-1][-1]) > 1e-6:
        return None
    # Remove artificial columns from consideration; set phase2 obj = c
    # Rebuild obj row from c for original vars
    T.pop()
    obj2 = [0.0] * (N + 1)
    for j in range(n):
        obj2[j] = -c[j]  # maximize c → tableau stores -c
    # make basic vars zero in obj
    for i, bj in enumerate(basis):
        if bj < n and abs(obj2[bj]) > 1e-15:
            coef = obj2[bj]
            obj2 = [obj2[j] - coef * T[i][j] for j in range(N + 1)]
        elif bj >= n:
            # still basic artificial with 0 value hopefully
            pass
    T.append(obj2)
    # zero out artificials by never entering them: set their obj to -inf
    for j in range(n, N):
        T[-1][j] = -1e100
    if not run():
        return None
    x = [0.0] * n
    for i, bj in enumerate(basis):
        if bj < n:
            x[bj] = T[i][-1]
    return x


def solve_fluid(nic_GBs: float = 11.57, pcie_GBs: float = 18.0) -> dict:
    """Min T s.t. 4 commodities of W/2 delivered via path flows; link load ≤ cap*T."""
    t0 = time.perf_counter()
    shard = W_BYTES / 2.0
    nic = nic_GBs * 1e9
    pcie = pcie_GBs * 1e9
    links = ["01", "23", "02", "03", "12", "13"]
    caps = {
        "01": pcie,
        "23": pcie,
        "02": nic,
        "03": nic,
        "12": nic,
        "13": nic,
    }
    commodities = [(0, 2), (0, 3), (1, 2), (1, 3)]

    def path_links(s, d, name):
        peer_s = 1 - s
        peer_d = 5 - d
        if name == "direct":
            return [f"{min(s,d)}{max(s,d)}"]
        if name == "via_train":
            return ["01", f"{min(peer_s,d)}{max(peer_s,d)}"]
        if name == "via_infer":
            return [f"{min(s,peer_d)}{max(s,peer_d)}", "23"]
        raise ValueError(name)

    path_names = ["direct", "via_train", "via_infer"]
    # variables: f[c,p] and T. Reformulate: maximize λ=1/T equivalent is awkward.
    # Instead binary-search T and check feasibility of max-flow style LP.
    # Feasibility: exists f≥0, sum_p f_{c,p}=shard, sum_{(c,p): e in p} f ≤ cap_e * T

    def feasible(T: float) -> bool:
        # variables indexed
        var_keys: List[Tuple] = []
        for c in commodities:
            for pn in path_names:
                var_keys.append((c, pn))
        n = len(var_keys)
        # Equalities: 4 demand constraints
        A = []
        b = []
        for ci, c in enumerate(commodities):
            row = [0.0] * n
            for j, (cc, pn) in enumerate(var_keys):
                if cc == c:
                    row[j] = 1.0
            A.append(row)
            b.append(shard)
        # Inequalities → equalities with slacks: load_e + s_e = cap*T
        for e in links:
            row = [0.0] * n
            for j, (c, pn) in enumerate(var_keys):
                if e in path_links(c[0], c[1], pn):
                    row[j] = 1.0
            # add slack as extra var — expand
            A2 = [r + [0.0] for r in A]
            for r in A2:
                pass
            # rebuild properly below
        # Rebuild with slacks
        n_slack = len(links)
        A = []
        b = []
        for c in commodities:
            row = [0.0] * (n + n_slack)
            for j, (cc, pn) in enumerate(var_keys):
                if cc == c:
                    row[j] = 1.0
            A.append(row)
            b.append(shard)
        for ei, e in enumerate(links):
            row = [0.0] * (n + n_slack)
            for j, (c, pn) in enumerate(var_keys):
                if e in path_links(c[0], c[1], pn):
                    row[j] = 1.0
            row[n + ei] = 1.0  # slack
            A.append(row)
            b.append(caps[e] * T)
        # dummy maximize 0
        cobj = [0.0] * (n + n_slack)
        x = _simplex(A, b, cobj)
        return x is not None

    # analytical dual-NIC cut first
    t_lo = shard / nic  # one shard one NIC
    t_cut = W_BYTES / (2 * nic)  # full W over dual NIC once each
    lo, hi = t_cut * 0.5, t_cut * 4.0
    if not feasible(hi):
        hi = t_cut * 20.0
    for _ in range(40):
        mid = 0.5 * (lo + hi)
        if feasible(mid):
            hi = mid
        else:
            lo = mid
    return {
        "solver": "pure_python_fluid_binsearch",
        "status": "Optimal",
        "makespan_s": hi,
        "solve_s": time.perf_counter() - t0,
        "nic_GBs": nic_GBs,
        "pcie_GBs": pcie_GBs,
        "dual_nic_cut_s": t_cut,
        "note": "binary-search path LP; matches dual-NIC cut when PCIe not bottleneck",
    }


if __name__ == "__main__":
    import json

    print(json.dumps(solve_fluid(), indent=2))
