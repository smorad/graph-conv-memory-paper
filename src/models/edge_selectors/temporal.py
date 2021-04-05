import torch


class TemporalBackedge(torch.nn.Module):
    """Add temporal bidirectional back edge, but only if we have >1 nodes
    E.g., node_{t} <-> node_{t-1}"""

    def __init__(self, num_hops=1):
        super().__init__()
        self.num_hops = num_hops

    def forward(self, nodes, adj_mats, edge_weights, num_nodes, B):
        [valid_batches] = torch.where(num_nodes >= self.num_hops)
        # TODO: Fix this to work with multiple hops
        assert self.num_hops == 1, "num_hops >1 not working yet"
        for hop in range(self.num_hops):
            adj_mats[
                valid_batches,
                num_nodes[valid_batches],
                num_nodes[valid_batches] - self.num_hops,
            ] = 1
            adj_mats[
                valid_batches,
                num_nodes[valid_batches] - self.num_hops,
                num_nodes[valid_batches],
            ] = 1

        return adj_mats, edge_weights
