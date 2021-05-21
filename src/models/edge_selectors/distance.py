import torch


class Distance(torch.nn.Module):
    """Base class for edges based on the similarity between
    latent representations"""

    def __init__(self, max_distance, bidirectional=False):
        super().__init__()
        self.max_distance = max_distance
        self.bidirectional = bidirectional

    def forward(self, nodes, adj_mats, edge_weights, num_nodes, B):
        """Connect current obs to past obs based on distance of the node features"""
        B_idx = torch.arange(B)
        curr_nodes = nodes[B_idx, num_nodes[B_idx].squeeze()]
        dists = self.dist_fn(curr_nodes, nodes)
        batch_idxs, node_idxs = torch.where(dists < self.max_distance)
        # Remove entries beyond num_nodes
        # as well as num_nodes because we don't want
        # the self edge
        num_nodes_mask = node_idxs < num_nodes[batch_idxs]
        batch_idxs = batch_idxs.masked_select(num_nodes_mask)
        node_idxs = node_idxs.masked_select(num_nodes_mask)

        adj_mats[batch_idxs, num_nodes[batch_idxs].squeeze(), node_idxs] = 1
        if self.bidirectional:
            adj_mats[batch_idxs, node_idxs, num_nodes[batch_idxs].squeeze()] = 1

        return adj_mats, edge_weights


class EuclideanEdge(Distance):
    """Mean per-dimension euclidean distance between obs vectors"""

    def __init__(self, max_distance):
        super().__init__(max_distance)

    def dist_fn(self, a, b):
        return torch.cdist(a, b).mean(dim=1)


class CosineEdge(Distance):
    """Mean per-dimension cosine distance between obs vectors"""

    def __init__(self, max_distance):
        super().__init__(max_distance)
        self.cs = torch.nn.modules.distance.CosineSimilarity(dim=2)

    def dist_fn(self, a, b):
        a = torch.cat([a.unsqueeze(1)] * b.shape[1], dim=1)
        return self.cs(a, b)


class SpatialEdge(Distance):
    """Euclidean distance representing the physical distance between two observations.
    Uses the slices a_pose_slice and b_pose_slice to extract the respective
    poses from the latent vectors"""

    def __init__(self, max_distance, a_pose_slice, b_pose_slice=None):
        super().__init__(max_distance)
        self.a_pose_slice = a_pose_slice
        if b_pose_slice:
            self.b_pose_slice = b_pose_slice
        else:
            self.b_pose_slice = a_pose_slice

    def dist_fn(self, a, b):
        a = torch.cat([a.unsqueeze(1)] * b.shape[1], dim=1)
        ra = a[:, :, self.a_pose_slice]
        rb = b[:, :, self.b_pose_slice]
        return torch.cdist(ra, rb).mean(dim=1)
