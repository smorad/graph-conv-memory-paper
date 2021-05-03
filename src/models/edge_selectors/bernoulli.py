import torch
import itertools
from ray.rllib.utils.typing import TensorType
from typing import Dict, Tuple, List
import numpy as np


@torch.jit.script
def up_to_num_nodes_idxs(
    adj: torch.Tensor, num_nodes: torch.Tensor
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Given num_nodes, returns idxs from adj
    up to but not including num_nodes. I.e.
    [batches, 0:num_nodes, num_nodes]. Note the order is
    sorted by (batches, num_nodes, 0:num_nodes) in ascending order"""
    seq_lens = num_nodes.unsqueeze(-1)
    N = adj.shape[-1]
    N_idx = torch.arange(N, device=adj.device).unsqueeze(0)
    N_idx = N_idx.expand(seq_lens.shape[0], N_idx.shape[1])
    # Do not include the current node
    N_idx = torch.nonzero(N_idx < num_nodes.unsqueeze(1))
    assert N_idx.shape[-1] == 2
    batch_idxs = N_idx[:, 0]
    past_idxs = N_idx[:, 1]
    curr_idx = num_nodes[batch_idxs]

    return batch_idxs, past_idxs, curr_idx


@torch.jit.script
def sample_hard(x: torch.Tensor, clamp_range: Tuple[float, float]) -> torch.Tensor:
    """
    Randomly sample a Bernoulli distribution and return the argmax.
    E.g. sample_hard(torch.tensor([0.4, 0.99, 0]) will likely return
    either [0, 1, 0] or [1, 1, 0]. Note that the inputs are expected to
    be probabilities and are clamped to self.clamp_range for numerical
    stability.

    This uses the gumbel_softmax trick to make argmax differentiable
    """

    res = torch.stack((1.0 - x, x), dim=0).clamp(*clamp_range)
    return torch.nn.functional.gumbel_softmax(torch.log(res), hard=True, dim=0)[1]


class BernoulliEdge(torch.nn.Module):
    """Add temporal bidirectional back edge, but only if we have >1 nodes
    E.g., node_{t} <-> node_{t-1}"""

    def __init__(
        self,
        input_size: int = 0,
        model: torch.nn.Sequential = None,
        clamp_range=(0.0001, 0.9999),
        backward_edges: bool = False,
        # gradient_scale: float = 0.5
    ):
        self.clamp_range: Tuple[float] = clamp_range
        self.backward_edges = backward_edges
        super().__init__()
        # for p in self.parameters():
        #    p.register_hook(lambda grad: grad / gradient_scale)
        # init loss
        self.density = torch.tensor(0)
        self.density_numel = torch.tensor(0)
        self.detach_loss()
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

    def build_edge_network(self, input_size: int) -> torch.nn.Sequential:
        """Builds a network to predict edges.
        Network input: (i || j)
        Network output: p(edge) in [0,1]
        """
        return torch.nn.Sequential(
            torch.nn.Linear(2 * input_size, input_size),
            torch.nn.Tanh(),
            torch.nn.Linear(input_size, 1),
            torch.nn.Sigmoid(),
        )

    def detach_loss(self):
        """Returns a copy of the accumulated loss and resets the loss
        and associated gradient to zero. Call this exactly once
        per each backward pass."""
        loss = self.density.clone() / self.density_numel
        self.density = torch.tensor(0)
        self.density_numel = torch.tensor(0)
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

    def update_density(self, probs):
        """Updates the density as a moving mean"""
        # https://math.stackexchange.com/questions/106700/incremental-averageing
        if self.training:
            self.density = self.density + probs.sum()
            self.density_numel = self.density_numel + probs.numel()

    def compute_logits2(
        self,
        nodes: torch.Tensor,
        num_nodes: torch.Tensor,
        weights: torch.Tensor,
        B: int,
    ):
        """Computes edge probability between current node and all other nodes.
        Returns a modified copy of the weight matrix containing edge probs"""
        # No edges for a single node
        if torch.max(num_nodes) < 1:
            return weights.clamp(*self.clamp_range)

        b_idxs, past_idxs, curr_idx = up_to_num_nodes_idxs(weights, num_nodes)
        # curr_idx > past_idxs
        # flows from past_idxs to j
        # so [j, past_idxs]
        curr_nodes = nodes[b_idxs, curr_idx]
        past_nodes = nodes[b_idxs, past_idxs]

        net_in = torch.cat((curr_nodes, past_nodes), dim=-1)
        net_out = self.edge_network(net_in)
        probs = net_out.clamp(*self.clamp_range).squeeze()
        # TODO: weights[:,0] is not populated, why?
        weights[b_idxs, curr_idx, past_idxs] = probs
        self.update_density(probs)
        # self.density = self.density + probs.mean()

        if self.backward_edges:
            net_in = torch.cat((past_nodes, curr_nodes), dim=-1)
            probs = net_out.clamp(*self.clamp_range).squeeze()
            weights[b_idxs, past_idxs, curr_idx] = probs
            self.update_density(probs)
            # self.density = self.density + probs.mean()

        return weights

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
        self.density = self.density + probs.mean()

        # Undirected edges
        # TODO: This is not equivariant as nn(a,b) != nn(b,a), run both dirs thru
        # and mean the output
        # TODO: Experiment with directed edges
        # TODO: Vectorize
        # TODO: Do not push all N nodes thru net, only push num_nodes
        for b in B_idx:
            # This does NOT add self edge [num_nodes[b], num_nodes[b]]
            #
            # For directed edges:
            # DenseGraphConv does Adj @ nodes
            # so A[i, j] corresponds to aggregating i when convolving
            # the jth row
            if self.backward_edges:
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
        weights = self.compute_logits2(nodes, num_nodes, weights, B)
        # TODO: we should only sample up to num_nodes
        b_idxs, past_idxs, curr_idx = up_to_num_nodes_idxs(weights, num_nodes)
        # Only sample entries that exist
        # otherwise we may accumulate zeros
        if len(b_idxs) > 0:
            adj = sample_hard(weights, self.clamp_range)

        return adj, weights
