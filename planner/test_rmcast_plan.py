"""Sanity tests for rmcast_plan v0 (DESIGN.md §4: the two mandated checks + extras).

Run: python -m pytest planner/test_rmcast_plan.py -q   (or plain python)
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from rmcast_plan import DEMOS, Topology, Host, build_demand_classes, plan

W = 15_231_233_024
HALF = W // 2


def _plan(demo: str):
    topo, src, dst = DEMOS[demo]()
    return plan(topo, src, dst)


def test_2x2_reproduces_mode_d():
    """Mandated check 1: the 2+2 topology must yield mode D's group split."""
    p = _plan("2x2")
    assert len(p.bcasts) == 2, p.summary()
    by_root = {op.root: op for op in p.bcasts}
    # shard0: root 0 -> both target GPUs; shard1: root 1 -> both. No rank-1/0
    # in each other's group (mode A's backflow member is absent by construction).
    assert by_root[0].members == (0, 2, 3)
    assert by_root[1].members == (1, 2, 3)
    assert by_root[0].nbytes == HALF and by_root[1].nbytes == W - HALF
    # dual-NIC balance: each source NIC carries exactly one shard (~W/2)
    egress = p.nic_egress
    assert egress[("h112", "mlx5_1")] == HALF
    assert egress[("h112", "mlx5_3")] == W - HALF
    # cross-machine bytes = W exactly (volume-tight; counter tax excluded here)
    assert p.cross_bytes() == W


def test_single_nic_reproduces_mode_a_without_backflow():
    """Mandated check 2: single-NIC source -> mode A's chain, minus its defect."""
    p = _plan("single_nic")
    assert len(p.bcasts) == 2
    for op in p.bcasts:
        assert op.nic == "mlx5_1"  # all forward traffic on the only NIC
        # crucially: the OTHER source GPU is not a member (no W/2 backflow)
        other_src = 1 - op.root
        assert other_src not in op.members
    assert p.nic_egress[("h112", "mlx5_1")] == W
    assert p.cross_bytes() == W


def test_tp2_tp4_misaligned_reshard():
    """TP=2 -> TP=4 across two hosts: overlay splits shards, no byte inflation."""
    p = _plan("tp2_tp4")
    # each source half splits into two quarters with distinct consumers
    assert len(p.classes) == 4
    # every quarter has exactly one consumer GPU -> 4 point-to-point bcasts
    assert all(len(op.members) == 2 for op in p.bcasts)
    # cross bytes = W exactly: reshard must not inflate wire traffic (Lemma 0)
    assert p.cross_bytes() == W
    # source NICs balanced: two quarters each
    assert p.nic_egress[("h112", "mlx5_1")] == W // 4 * 2
    assert p.nic_egress[("h112", "mlx5_3")] == W - W // 4 * 2


def test_intra_host_demand_stays_off_network():
    """A consumer colocated with the owner must be served by local copy."""
    topo = Topology()
    topo.add_host(Host("hA", [0, 1], {"nic0": 11.55}, {0: "nic0", 1: "nic0"}))
    topo.add_host(Host("hB", [2], {"nic0": 11.55}, {2: "nic0"}))
    src = {0: [(0, 100)]}
    dst = {1: [(0, 100)], 2: [(0, 100)]}
    p = plan(topo, src, dst)
    assert len(p.copies) == 1 and p.copies[0].dsts == (1,)
    assert len(p.bcasts) == 1 and p.bcasts[0].members == (0, 2)
    assert p.cross_bytes() == 100  # GPU1 served intra-host, not over the wire


def test_holder_is_not_a_consumer():
    """Bytes a target already holds (warm cache / re-plan) create no demand."""
    topo, src, dst = DEMOS["2x2"]()
    src2 = dict(src)
    src2[2] = [(0, HALF)]  # GPU2 already holds shard0 (e.g. survived churn)
    p = plan(topo, src2, dst)
    # shard0 now only needs to reach GPU3; also GPU2 becomes a candidate holder
    shard0_ops = [op for op in p.bcasts if any(
        p.classes[i].intervals[0][0] == 0 for i in op.classes)]
    assert all(2 not in op.members or op.root == 2 for op in shard0_ops)


def test_multi_holder_owner_selection_balances_nics():
    """DP=2: same shard held by two GPUs on different NICs -> both get used."""
    topo = Topology()
    topo.add_host(Host("hA", [0, 1], {"nic0": 10.0, "nic1": 10.0},
                       {0: "nic0", 1: "nic1"}))
    topo.add_host(Host("hB", [2], {"nic0": 10.0}, {2: "nic0"}))
    # two equal classes, both held by BOTH source GPUs (DP replicas)
    src = {0: [(0, 100), (100, 200)], 1: [(0, 100), (100, 200)]}
    dst = {2: [(0, 200)]}
    p = plan(topo, src, dst)
    roots = sorted(op.root for op in p.bcasts)
    assert roots == [0, 1], f"expected both replicas used as roots, got {roots}"


def test_errors():
    topo, src, dst = DEMOS["2x2"]()
    try:
        build_demand_classes({0: [(10, 10)]}, dst)
        raise AssertionError("empty interval accepted")
    except ValueError:
        pass
    try:
        build_demand_classes({0: [(0, 5)]}, {2: [(0, 10)]})
        raise AssertionError("unheld demand accepted")
    except ValueError:
        pass


if __name__ == "__main__":
    fails = 0
    for name, fn in sorted(globals().items()):
        if name.startswith("test_") and callable(fn):
            try:
                fn()
                print(f"PASS {name}")
            except AssertionError as e:
                fails += 1
                print(f"FAIL {name}: {e}")
    sys.exit(1 if fails else 0)
