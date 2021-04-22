import torch

# VISDOM TOP HALF PROVIDES BEST PERF

# adj[0,3] = 1
# neigh = matmul(adj, nodes) = nodes[0]
# [i,j] => base[j] neighbor[i]
# propagates from i to j

# neighbor: torch.matmul(Adj[i, j], x) = x[i] = adj[i]
# self: adj[j]
# Vis: should be top half of visdom


class TemporalBackedge(torch.nn.Module):
    """Add temporal bidirectional back edge, but only if we have >1 nodes
    E.g., node_{t} <-> node_{t-1}"""

    def __init__(self, num_hops=1, bidirectional=False):
        super().__init__()
        self.num_hops = num_hops
        self.bidirectional = bidirectional

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
            if self.bidirectional:
                adj_mats[
                    valid_batches,
                    num_nodes[valid_batches] - self.num_hops,
                    num_nodes[valid_batches],
                ] = 1

        return adj_mats, edge_weights
