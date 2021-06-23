import torch
import torch_geometric
from torch_geometric.data import Data, Batch
from typing import List, Tuple, Union, Any, Dict, Callable
import time


def sparse_to_dense(batch: Batch) -> Batch:
    sparse_edges = torch_geometric.utils.to_dense_adj(
        batch.edge_index, batch=batch.batch, max_num_nodes=batch.N
    )[0]
    sparse_nodes = torch_geometric.utils.to_dense_batch(
        x=batch.x, batch=batch.batch, max_num_nodes=batch.N
    )[0]
    dense_batch = Batch(x=sparse_nodes, adj=sparse_edges, N=batch.N, B=batch.B)
    return dense_batch


class SparseToDense(torch.nn.Module):
    """Convert from edge_list to adj. """

    def forward(self, x, edge_index, batch_idx, B, N):
        # TODO: Should handle weights
        x = torch_geometric.utils.to_dense_batch(x=x, batch=batch_idx, max_num_nodes=N)[
            0
        ]
        adj = torch_geometric.utils.to_dense_adj(
            edge_index, batch=batch_idx, max_num_nodes=N
        )[0]
        return x, adj


class DenseToSparse(torch.nn.Module):
    """Convert from adj to edge_list while allowing gradients
    to flow through adj.

    x: shape[B, N+k, feat]
    adj: shape[B, N+k, N+k]
    mask: shape[B, N+k]"""

    def forward(self, x, adj, mask=None):
        assert x.dim() == adj.dim() == 3
        B = x.shape[0]
        N = x.shape[1]
        if mask:
            assert mask.shape == (B, N)
            x_mask = mask.unsqueeze(-1).expand(-1, -1, x.shape[-1])
            adj_mask = mask.unsqueeze(-1).expand(-1, -1, adj.shape[-1])
            x = x * x_mask
            adj = adj * adj_mask
            import pdb

            pdb.set_trace()
            N = mask.shape[1]
        offset, row, col = torch.nonzero(adj > 0).t()
        row += offset * N
        col += offset * N
        edge_index = torch.stack([row, col], dim=0).long()
        x = x.view(B * N, x.shape[-1])
        batch_idx = (
            torch.arange(0, B, device=x.device).view(-1, 1).repeat(1, N).view(-1)
        )

        return x, edge_index, batch_idx


def dense_to_sparse(batch: Batch) -> Batch:
    """Convert from adj to edge_list while allow gradients
    to flow through adj"""

    offset, row, col = torch.nonzero(batch.adj > 0).t()
    # edge_weight = batch.weight[offset, row, col].float()
    row += offset * batch.N
    col += offset * batch.N
    edge_index = torch.stack([row, col], dim=0).long()
    x = batch.x.view(batch.B * batch.N, batch.x.shape[-1])
    batch_idx = (
        torch.arange(0, batch.B, device=batch.x.device)
        .view(-1, 1)
        .repeat(1, batch.N)
        .view(-1)
    )
    sparse_batch = Batch(
        x=x,
        edge_index=edge_index,
        # edge_weight=edge_weight,
        batch=batch_idx,
        B=batch.B,
        N=batch.N,
    )

    return sparse_batch


@torch.jit.script
def overflow(num_nodes: torch.Tensor, N: int):
    return torch.any(num_nodes + 1 > N)


"""
    def forward(self, batch: Batch):
        if self.test == "adj":
            batch.adj = batch.adj * self.grad_test_var
        elif self.test:
            batch.x = batch.x * self.grad_test_var
        if self.sparse:
            # sparse_batch = self.dense_to_sparse(batch)
            return self.forward_sparse(batch)
        else:
            return self.forward_dense(batch)

    def forward_dense(self, batch: Batch) -> Batch:
        # Clone here to avoid backprop error
        output = batch.x.clone()
        for layer in self.layers:
            if type(layer) == self.conv_type:
                # TODO: Multiply edge weights
                output = layer(output, adj=batch.adj)
            else:
                output = layer(output)
        return output

    def forward_sparse(self, batch: Batch) -> Batch:
        output = batch.x.clone()
        for layer in self.layers:
            if type(layer) == self.conv_type:
                output = layer(output, edge_index=batch.edge_index)
            else:
                output = layer(output)
        return torch_geometric.utils.to_dense_batch(
            output, batch=batch.batch, max_num_nodes=batch.N
        )[0]
"""


class DenseGCM(torch.nn.Module):
    """Graph Associative Memory"""

    did_warn = False

    def __init__(
        self,
        gnn: torch.nn.Module,
        preprocessor: torch.nn.Module = None,
        edge_selectors: torch.nn.Module = None,
        graph_size: int = 128,
        pooled: bool = False,
    ):
        super().__init__()

        self.preprocessor = preprocessor
        self.gnn = gnn
        self.graph_size = graph_size
        self.edge_selectors = edge_selectors
        self.pooled = pooled

    def forward(
        self, x, hidden: Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Add a memory x to the graph, and query the memory for it.
        B = batch size
        N = maximum graph size
        Inputs:
            x: [B,feat]
            hidden: (
                nodes: [B,N,feats]
                adj: [B,N,N]
                weights: [B,N,N]
                number_of_nodes_in_graph: [B]
            )
        Outputs:
            m(x): [B,feat]
            hidden: (
                nodes: [B,N,feats]
                adj: [B,N,N]
                weights: [B,N,N]
                number_of_nodes_in_graph: [B]
            )
        """
        nodes, adj, weights, num_nodes = hidden

        assert x.dtype == torch.float32
        assert nodes.dtype == torch.float
        # if self.gnn.sparse:
        #    assert adj.dtype == torch.long
        assert weights.dtype == torch.float
        assert num_nodes.dtype == torch.long
        assert num_nodes.dim() == 1

        N = nodes.shape[1]
        B = x.shape[0]
        B_idx = torch.arange(B)

        assert (
            N == adj.shape[1] == adj.shape[2]
        ), "N must be equal for adj mat and node mat"

        if overflow(num_nodes, N):
            if not self.did_warn:
                print("Overflow detected, wrapping around. Will not warn again")
                self.did_warn = True
            nodes, adj, weights, num_nodes = self.wrap_overflow(
                nodes, adj, weights, num_nodes
            )
        # Add new nodes to the current graph
        # starting at num_nodes
        nodes = nodes.clone()
        nodes[B_idx, num_nodes[B_idx]] = x[B_idx]

        # Do NOT add self edges or they will be counted twice using
        # GraphConv
        if self.edge_selectors:
            adj, weights = self.edge_selectors(
                nodes, adj.clone(), weights.clone(), num_nodes, B
            )

        # Thru network
        if self.preprocessor:
            nodes_in = self.preprocessor(nodes)
        else:
            nodes_in = nodes

        node_feats = self.gnn(nodes_in, adj, weights, B, N)
        if self.pooled:
            # If pooled, we expect only a single output node
            mx = node_feats
        else:
            # Otherwise extract the hidden repr at the current node
            mx = node_feats[B_idx, num_nodes[B_idx]]

        assert torch.all(
            torch.isfinite(mx)
        ), "Got NaN in returned memory, try using tanh activation"

        num_nodes = num_nodes + 1
        return mx, (nodes, adj, weights, num_nodes)

    def wrap_overflow(self, nodes, adj, weights, num_nodes):
        """Call this when the node/adj matrices are full. Deletes the zeroth element
        of the matrices and shifts all the elements up by one, producing a free row
        at the end.

        Returns new nodes, adj, weights, and num_nodes matrices"""
        N = nodes.shape[1]
        overflow_mask = num_nodes + 1 > N
        # Shift node matrix into the past
        # by one and forget the zeroth node
        overflowing_batches = overflow_mask.nonzero().squeeze()
        nodes = nodes.clone()
        adj = adj.clone()
        weights = weights.clone()
        # Zero entries before shifting
        nodes[overflowing_batches, 0] = 0
        adj[overflowing_batches, 0, :] = 0
        adj[overflowing_batches, :, 0] = 0
        weights[overflowing_batches, 0, :] = 0
        weights[overflowing_batches, :, 0] = 0
        # Roll newly zeroed zeroth entry to final entry
        nodes[overflowing_batches] = torch.roll(nodes[overflowing_batches], -1, -2)
        adj[overflowing_batches] = torch.roll(
            adj[overflowing_batches], (-1, -1), (-1, -2)
        )
        weights[overflowing_batches] = torch.roll(
            weights[overflowing_batches], (-1, -1), (-1, -2)
        )

        num_nodes[overflow_mask] = num_nodes[overflow_mask] - 1
        return nodes, adj, weights, num_nodes
