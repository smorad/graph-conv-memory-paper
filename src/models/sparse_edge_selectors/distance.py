import torch


class Distance(torch.nn.Module):
    """Base class for edges based on the similarity between
    latent representations"""

    def __init__(self, max_distance):
        super().__init__()
        self.max_distance = max_distance

    def forward(self, nodes, edge_list, weights, B, T, t):
        """Connect current obs to past obs based on distance of the node features"""
        # Compare B * t to B * (t + T) nodes
        # results in B * (t * (t + T)) comparisons
        # with shape [B, t, t + T]
        # after comparisons, edges will be
        # [B, t, <= t + T]
        cur_node_idxs = torch.arange(T, T + t)
        # TODO: Cur_nodes assumes batches are aligned along temporal dim
        cur_nodes = nodes[:, cur_node_idxs]
        # [B, T]
        dist = self.dist_fn(cur_nodes, nodes)
        b_i_j_idxs = torch.nonzero(dist < self.max_distance).t()
        # a is the cur_nodes and b is all nodes
        a, b = b_i_j_idxs[1:]
        # Add the offset as cur_node_idxs do not start at 0
        a = a + T

        assert a.shape == b.shape
        pairs = torch.stack((a, b)).reshape(2, B, -1).permute(1, 0, 2)
        # Filter out edges pointing to the future, e.g. a < b
        # as well as self edges a == b
        edge_mask = torch.stack(
            (pairs[:, 0, :] > pairs[:, 1, :], pairs[:, 0, :] > pairs[:, 1, :]),  # a,b
            dim=1,
        )
        pairs = pairs[edge_mask].reshape(B, 2, -1)

        edge_list = torch.cat((edge_list, pairs), dim=-1)
        return edge_list, weights


class EuclideanEdge(Distance):
    """Mean per-dimension euclidean distance between obs vectors"""

    def __init__(self, max_distance):
        super().__init__(max_distance)

    def dist_fn(self, a, b):
        return torch.cdist(a, b)


class CosineEdge(Distance):
    """Mean per-dimension cosine distance between obs vectors"""

    def __init__(self, max_distance):
        super().__init__(max_distance)
        self.cs = torch.nn.modules.distance.CosineSimilarity(dim=2)

    def dist_fn(self, a, b):
        a = torch.cat([a.unsqueeze(1)] * b.shape[1], dim=1)
        return self.cs(a, b)


class SpatialEdge(Distance):
    """Euclidean distance representing the physical distance between two observations"""

    def __init__(self, max_distance, pose_slice):
        super().__init__(max_distance)
        self.pose_slice = pose_slice

    def dist_fn(self, a, b):
        a = torch.cat([a.unsqueeze(1)] * b.shape[1], dim=1)
        ra = a[:, :, self.pose_slice]
        rb = b[:, :, self.pose_slice]
        return torch.cdist(ra, rb).mean(dim=1)
