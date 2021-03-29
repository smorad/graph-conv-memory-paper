import torch


class DenseEdge(torch.nn.Module):
    """Add temporal bidirectional back edge, but only if we have >1 nodes
    E.g., node_{t} <-> node_{t-1}"""

    def __init__(self):
        super().__init__()

    def forward(self, nodes, adj_mats, edge_weights, num_nodes, B):
        """Since this is called for each obs, it is sufficient to make row/col
        for obs 1"""

        # TODO: Batch this like DistanceEdge
        for b in range(B):
            i = num_nodes[b]
            adj_mats[b][i, :i] = 1
            adj_mats[b][:i, i] = 1
            # Self edge
            adj_mats[b][i, i] = 1

        return adj_mats, edge_weights
