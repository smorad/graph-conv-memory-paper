import torch


class TemporalBackedge(torch.nn.Module):
    """Add temporal bidirectional back edge, but only if we have >1 nodes
    E.g., node_{t} <-> node_{t-1}"""

    def __init__(self, parent):
        super().__init__()
        self.parent = parent

    def forward(self, nodes, adj_mats, num_nodes, state, B, **kwargs):
        for b in range(B):
            if adj_mats[b].shape[0] < 2:
                continue

            adj_mats[b, b - 1] = 1
            adj_mats[b - 1, b] = 1
