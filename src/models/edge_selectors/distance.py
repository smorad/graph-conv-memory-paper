import torch


class Distance(torch.nn.Module):
    def __init__(self, parent):
        super().__init__()
        self.parent = parent

    def forward(self, nodes, adj_mats, edge_weights, num_nodes, B):
        """Connect current obs to past obs based on distance of the node features"""
        B_idx = torch.arange(B)
        curr_nodes = nodes[B_idx, num_nodes[B_idx].squeeze()]
        # TODO: Only do distance up to nodes[num_nodes] so we don't compare
        # to zero entries
        # comp_nodes = torch.arange(num_nodes[B_idx])
        dists = self.dist_fn(curr_nodes, nodes)
        batch_idxs, node_idxs = torch.where(dists < self.MAX_DIST)
        import pdb

        pdb.set_trace()

        adj_mats[batch_idxs, num_nodes[batch_idxs].squeeze(), node_idxs] = 1
        adj_mats[batch_idxs, node_idxs, num_nodes[batch_idxs].squeeze()] = 1

        return adj_mats, edge_weights


class EuclideanEdge(Distance):
    """Mean per-dimension euclidean distance between obs vectors"""

    MAX_DIST = 3

    def __init__(self, parent):
        super().__init__(parent)
        self.dist_fn = lambda a, b: torch.cdist(a, b).mean(dim=1)


class CosineEdge(Distance):
    MAX_DIST = 0.2

    def __init__(self, parent):
        super().__init__(parent)
        self.dist_fn = torch.nn.modules.distance.CosineSimilarity(dim=0)
