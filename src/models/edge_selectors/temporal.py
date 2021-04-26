import torch
from typing import List

# VISDOM TOP HALF PROVIDES BEST PERF

# adj[0,3] = 1
# neigh = matmul(adj, nodes) = nodes[0]
# [i,j] => base[j] neighbor[i]
# propagates from i to j

# neighbor: torch.matmul(Adj[i, j], x) = x[i] = adj[i]
# self: adj[j]
# Vis: should be top half of visdom


class TemporalBackedge(torch.nn.Module):
    """Add temporal directional back edge, e.g., node_{t} -> node_{t-1}"""

    def __init__(self, hops: List[int] = [1], direction="forward"):
        """
        Hops: number of hops in the past to connect to
        E.g. [1] is t <- t-1, [2] is t <- t-2,
        [5,8] is t <- t-5 AND t <- t-8

        Direction: Directionality of graph edges. You likely want
        'forward', which indicates information flowing from past
        to future. Backward is information from future to past,
        and both is both.
        """
        super().__init__()
        self.hops = hops
        assert direction in ["forward", "backward", "both"]
        self.direction = direction

    def forward(self, nodes, adj_mats, edge_weights, num_nodes, B):
        # TODO: Fix this to work with multiple hops
        # assert self.hops == [1], "num_hops >1 not working yet"
        for hop in self.hops:
            [valid_batches] = torch.where(num_nodes >= hop)
            if self.direction in ["forward", "both"]:
                adj_mats[
                    valid_batches,
                    num_nodes[valid_batches],
                    num_nodes[valid_batches] - hop,
                ] = 1
            if self.direction in ["backward", "both"]:
                adj_mats[
                    valid_batches,
                    num_nodes[valid_batches] - hop,
                    num_nodes[valid_batches],
                ] = 1

        return adj_mats, edge_weights
