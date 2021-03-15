import torch


class TemporalBackedge(torch.nn.Module):
    """Add temporal bidirectional back edge, but only if we have >1 nodes
    E.g., node_{t} <-> node_{t-1}"""

    def __init__(self, parent):
        self.parent = parent

    def forward(self, nodes, adj_mats, num_nodes, B):
        import pdb

        pdb.set_trace()
