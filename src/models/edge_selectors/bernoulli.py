import torch
import itertools
from ray.rllib.utils.typing import TensorType
from typing import Dict
import numpy as np


class BernoulliEdge(torch.nn.Module):
    """Add temporal bidirectional back edge, but only if we have >1 nodes
    E.g., node_{t} <-> node_{t-1}"""

    def __init__(self, input_size: int = 0, model: torch.nn.Sequential = None):
        super().__init__()
        assert input_size or model, "Must specify either input_size or model"
        if model:
            self.edge_network = model
        else:
            self.edge_network = self.build_edge_network(input_size)
        self.metrics: Dict[str, np.ndarray] = {}

    def sample_random_var(self, p: torch.Tensor) -> torch.Tensor:
        """Given a probability [0,1] p, return a backprop-capable random sample"""
        e = torch.rand(p.shape, device=p.device)
        return torch.sigmoid(
            torch.log(e) - torch.log(1 - e) + torch.log(p) - torch.log(1 - p)
        )

    def to_hard(self, x: torch.Tensor) -> torch.Tensor:
        """Hard trick similar to that used in gumbel softmax in torch.
        Normally, argmax is not differentiable.
        Rounds to {0, 1} in a differentiable fashion"""
        res = torch.stack((1.0 - x, x), dim=0)
        return torch.nn.functional.gumbel_softmax(torch.log(res), hard=True, dim=0)[1]

    def build_edge_network(self, input_size: int) -> torch.nn.Sequential:
        """Builds a network to predict edges.
        Network input: (i || j)
        Network output: p(edge) in [0,1]
        """
        return torch.nn.Sequential(
            torch.nn.Linear(2 * input_size, input_size),
            torch.nn.LeakyReLU(),
            torch.nn.Linear(input_size, 1),
            torch.nn.Sigmoid(),
        )

    def compute_logits(
        self,
        nodes: torch.Tensor,
        num_nodes: torch.Tensor,
        weights: torch.Tensor,
        B: int,
    ):
        """Computes edge probability between current node and all other nodes.
        Returns a modified copy of the weight matrix containing edge probs"""

        B_idx = torch.arange(B)
        N = nodes.shape[1]
        feat = nodes.shape[-1]

        left_nodes = torch.stack([nodes[B_idx, num_nodes[B_idx]]] * N, dim=1)
        right_nodes = nodes
        edge_net_in = torch.cat((left_nodes, right_nodes), dim=-1)
        # Edge network expects [B, feat] but we have [B, N, feat]
        # so flatten to [B, feat] for big perf gainz and unflatten
        batch_in = edge_net_in.view(B * N, 2 * feat)
        batch_out = self.edge_network(batch_in)
        probs = batch_out.view(B, N)

        # Undirected edges
        # TODO: This is not equivariant as nn(a,b) != nn(b,a), run both dirs thru
        # and mean the output
        # TODO: Experiment with directed edges
        # TODO: Vectorize
        # TODO: Do not push all N nodes thru net, only push num_nodes
        for b in B_idx:
            # This does NOT add self edge [num_nodes[b], num_nodes[b]]
            weights[b, num_nodes[b], : num_nodes[b]] = probs[b][: num_nodes[b]]
            weights[b, : num_nodes[b], num_nodes[b]] = probs[b][: num_nodes[b]]

        return weights

    def forward(self, nodes, adj, weights, num_nodes, B):
        """A(i,j) = Ber[phi(i || j), e]

        Modify the nodes/adj_mats/state in-place by reference. Return value
        is not used.
        """

        # a(b,i,j) = gumbel_softmax(phi(n(b,i) || n(b,j))) for i, j < num_nodes
        # First run
        if self.edge_network[0].weight.device != nodes.device:
            self.edge_network = self.edge_network.to(nodes.device)

        # Weights serve as probabilities that we sample from
        # w = weights.clone()
        weights = self.compute_logits(nodes, num_nodes, weights, B)
        sample = self.sample_random_var(weights)
        # a = adj.clone()
        adj = self.to_hard(sample)

        return adj, weights
