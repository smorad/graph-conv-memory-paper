import torch


class TemporalBackedge(torch.nn.Module):
    """Add temporal bidirectional back edge, but only if we have >1 nodes
    E.g., node_{t} <-> node_{t-1}"""

    def __init__(self, parent):
        super().__init__()
        self.parent = parent

    def forward(self, nodes, adj_mats, edge_weights, num_nodes, B):
        [valid_batches] = torch.where(num_nodes >= 1)
        adj_mats[
            valid_batches, num_nodes[valid_batches], num_nodes[valid_batches] - 1
        ] = 1
        adj_mats[
            valid_batches, num_nodes[valid_batches] - 1, num_nodes[valid_batches]
        ] = 1

        return adj_mats, edge_weights
