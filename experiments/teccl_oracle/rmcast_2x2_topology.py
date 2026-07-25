"""2-host × 2-GPU topology matching 112/113 Active RoCE + PCIe/SYS.

Nodes: 0,1 = train GPUs; 2,3 = infer GPUs.
Capacities in chunks/sec (chunk_size GB given in TopologyParams).
"""
from __future__ import annotations

from teccl.input_data import TopologyParams
from teccl.topologies.topology import Topology


class RMcast2x2(Topology):
    """Fully connected rail-aware abstraction of our 4-GPU bench."""

    def __init__(self, topo_input: TopologyParams):
        # NIC goodput ~92.56 Gbps = 11.57 GB/s; PCIe ~18 GB/s (avg of 14.6/22.2)
        self.nic_GBs = 11.57
        self.pcie_GBs = 18.0
        super().__init__(topo_input)
        self.node_per_chassis = 2

    def construct_topology(self, topo_input: TopologyParams):
        cs = float(topo_input.chunk_size)  # GB
        nic = self.nic_GBs / cs
        pcie = self.pcie_GBs / cs
        # 4 GPUs, no explicit switch nodes (direct edges; switch_copy irrelevant)
        # Cross-host: all pairs full NIC rate (ENV: cross-rail also 92.56 Gbps)
        # Intra-host: PCIe
        n = 4
        cap = [[0.0] * n for _ in range(n)]
        alpha = [[-1.0] * n for _ in range(n)]
        train, infer = (0, 1), (2, 3)
        for a, b in ((0, 1), (2, 3)):
            cap[a][b] = cap[b][a] = pcie
            alpha[a][b] = alpha[b][a] = 0.5e-6
        for s in train:
            for d in infer:
                cap[s][d] = cap[d][s] = nic
                alpha[s][d] = alpha[d][s] = 1.3e-6
        self.capacity = cap
        self.alpha = alpha
        self.topology = [[int(x > 0) for x in row] for row in cap]
        self.switch_indices = []

    def set_switch_indicies(self) -> None:
        self.switch_indices = []
