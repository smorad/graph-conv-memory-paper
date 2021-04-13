import torch
import itertools
from ray.rllib.utils.typing import TensorType
from typing import Dict
import numpy as np


class BernoulliEdge(torch.nn.Module):
    """Add temporal bidirectional back edge, but only if we have >1 nodes
    E.g., node_{t} <-> node_{t-1}"""

    def __init__(
        self,
        input_size: int = 0,
        model: torch.nn.Sequential = None,
        clamp_range=[0.001, 0.999],
    ):
        super().__init__()
        self.density = torch.tensor(0)
        self.clamp_range = clamp_range
        assert input_size or model, "Must specify either input_size or model"
        if model:
            self.edge_network = model
        else:
            # This MUST be done here
            # if done in forward model does not learn...
            self.edge_network = self.build_edge_network(input_size)

    def sample_random_var(self, p: torch.Tensor) -> torch.Tensor:
        """Given a probability [0,1] p, return a backprop-capable random sample"""
        e = torch.rand(p.shape, device=p.device)
        return torch.sigmoid(
            torch.log(e) - torch.log(1 - e) + torch.log(p) - torch.log(1 - p)
        )

    def sample_hard(self, x: torch.Tensor) -> torch.Tensor:
        """
        Randomly sample a Bernoulli distribution and return the argmax.
        E.g. sample_hard(torch.tensor([0.4, 0.99, 0]) will likely return
        either [0, 1, 0] or [1, 1, 0]. Note that the inputs are expected to
        be probabilities and are clamped to self.clamp_range for numerical
        stability.

        This uses the gumbel_softmax trick to make argmax differentiable
        """

        res = torch.stack((1.0 - x, x), dim=0).clamp(*self.clamp_range)
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

    def detach_loss(self):
        """Returns a copy of the accumulated loss and resets the loss
        and associated gradient to zero. Call this exactly once
        per each backward pass."""
        loss = self.density.clone()
        self.density = torch.tensor(0)
        return loss

    def compute_full_loss(self, nodes: torch.Tensor, B: int):
        """Compute loss over the entire weight node matrix.
        (B, N, feat) by (B, N*N, feat)"""
        N = nodes.shape[1]
        feat = nodes.shape[-1]
        # Right: from [[0, 1, 2], [3,4,5]]
        # to [[0,1,2], [3,4,5], [0,1,2], [3,4,5] ... ]
        left_nodes = nodes.repeat_interleave(N, dim=1)
        # Left: from [[0, 1, 2], [3,4,5]]
        # to [[0,1,2], [0,1,2]... [3,4,5], [3,4,5]]
        right_nodes = nodes.repeat(1, N, 1)
        edge_net_in = torch.cat((left_nodes, right_nodes), dim=-1)
        # Free memory
        del left_nodes
        del right_nodes
        # Edge network expects [B, feat] but we have [B, N, feat]
        # so flatten to [B, feat] for big perf gainz and unflatten
        batch_in = edge_net_in.view(B * N ** 2, 2 * feat)
        batch_out = self.edge_network(batch_in)
        return batch_out.mean()

    def compute_logits(
        self,
        nodes: torch.Tensor,
        num_nodes: torch.Tensor,
        weights: torch.Tensor,
        B: int,
    ):
        """Computes edge probability between current node and all other nodes.
        Returns a modified copy of the weight matrix containing edge probs"""

        # No edges for a single node
        if torch.max(num_nodes) == 0:
            return weights.clamp(*self.clamp_range)

        B_idx = torch.arange(B)
        n = torch.max(num_nodes)
        # N = nodes.shape[1]
        feat = nodes.shape[-1]

        left_nodes = torch.stack([nodes[B_idx, num_nodes[B_idx]]] * n, dim=1)
        right_nodes = nodes[:, :n, :]
        edge_net_in = torch.cat((left_nodes, right_nodes), dim=-1)
        # Edge network expects [B, feat] but we have [B, N, feat]
        # so flatten to [B, feat] for big perf gainz and unflatten
        batch_in = edge_net_in.view(B * n, 2 * feat)
        batch_out = self.edge_network(batch_in)
        probs = batch_out.view(B, n).clamp(*self.clamp_range)
        # print(probs.mean())
        self.density = self.density + probs.mean()  # probs.sum() / num_nodes.sum()

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
            # if num_nodes[b] > 0:
            #    self.density = self.density + probs[b][: num_nodes[b]].mean()

        # Nx2 possible rows
        # self.density = self.density + 2 * batch_out.sum() / ((num_nodes + 1) * 2).sum()
        # probs[b][: num_nodes[b]].sum()

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
        weights = self.compute_logits(nodes, num_nodes, weights, B)
        # sample = self.sample_random_var(weights)
        adj = self.sample_hard(weights)

        return adj, weights
