import torch
import torch_geometric
from torch_geometric.data import Data, Batch
from typing import List, Tuple, Union, Any, Dict
import time


class GNN(torch.nn.Module):
    def hidden_block(self, hidden_size, activation, conv_type, attn_heads):
        return [
            conv_type(hidden_size, hidden_size),
            activation(),
            # torch_geometric.nn.BatchNorm(hidden_size),
        ]

    def __init__(
        self,
        input_size: int,
        output_size: int,
        graph_size: int = 128,
        hidden_size: int = 64,
        num_layers: int = 2,
        attn_heads: int = 1,
        conv_type: torch_geometric.nn.MessagePassing = torch_geometric.nn.DenseGCNConv,
        activation: torch.nn.Module = torch.nn.Tanh,  # torch.nn.ReLU,
        sparse: bool = False,
    ):
        super().__init__()
        self.input_size = input_size
        self.sparse = sparse

        first = [
            conv_type(input_size, hidden_size),
            activation(),
            # torch_geometric.nn.BatchNorm(hidden_size),
        ]
        hiddens = []
        for i in range(num_layers):
            hiddens += self.hidden_block(hidden_size, activation, conv_type, attn_heads)
        final = [conv_type(hidden_size, output_size)]

        self.layers = torch.nn.ModuleList([*first, *hiddens, *final])
        self.conv_type = conv_type

    def dense_to_sparse(self, batch: Batch) -> Batch:
        # Convert from adj to edge_list so we can use more types of
        # convs. Edge weight is required to allow gradients to flow back
        # into the adjaceny matrix

        offset, row, col = torch.nonzero(batch.adj > 0).t()
        edge_weight = batch.adj[offset, row, col].float()
        row += offset * batch.N
        col += offset * batch.N
        edge_index = torch.stack([row, col], dim=0).long()
        x = batch.x.view(batch.B * batch.N, batch.x.shape[-1])
        batch_idx = torch.arange(0, batch.B).view(-1, 1).repeat(1, batch.N).view(-1)
        sparse_batch = Batch(
            x=x,
            edge_index=edge_index,
            edge_weight=edge_weight,
            batch=batch_idx,
            B=batch.B,
            N=batch.N,
        )

        return sparse_batch

    def sparse_to_dense(self, batch: Batch) -> Batch:
        sparse_edges = torch_geometric.utils.to_dense_adj(
            batch.edge_index, batch=batch.batch, max_num_nodes=batch.N
        )[0]
        sparse_nodes = torch_geometric.utils.to_dense_batch(
            x=batch.x, batch=batch.batch, max_num_nodes=batch.N
        )[0]
        dense_batch = Batch(x=sparse_nodes, adj=sparse_edges, N=batch.N, B=batch.B)
        return dense_batch

    def forward(self, batch: Batch):
        if self.sparse:
            # sparse_batch = self.dense_to_sparse(batch)
            return self.forward_sparse(batch)
        else:
            return self.forward_dense(batch)

    def forward_dense(self, batch: Batch) -> Batch:
        # Clone here to avoid backprop error
        output = batch.x.clone()
        for layer in self.layers:
            if type(layer) == torch_geometric.nn.BatchNorm:
                orig_shape = output.shape
                collapsed = output.reshape(-1, orig_shape[-1])
                output = layer(collapsed).reshape(orig_shape)
            elif type(layer) == self.conv_type:
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


class DenseGAM(torch.nn.Module):
    """Graph Associative Memory"""

    def __init__(
        self,
        gnn: GNN,
        edge_selectors: List[torch.nn.Module] = [],
        graph_size: int = 128,
    ):
        super().__init__()

        self.gnn = gnn
        self.graph_size = graph_size
        self.edge_selectors = torch.nn.ModuleList(edge_selectors)

    def pack_state(self):
        """Pack state from into a list of torch.tensors"""
        pass

    def unpack_state(self, state):
        """Unpack state from a list of torch.tensors into a torch_geometric.data.Batch"""
        pass

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
        assert adj.dtype == torch.long
        assert weights.dtype == torch.float
        assert num_nodes.dtype == torch.long

        N = nodes.shape[1]
        B = x.shape[0]
        B_idx = torch.arange(B)

        assert (
            N == adj.shape[1] == adj.shape[2]
        ), "N must be equal for adj mat and node mat"

        if torch.any(num_nodes + 1 >= N):
            print(
                f"Warning, ran out of graph space (N={N}, t={num_nodes.max() + 1}). Overwriting node matrix"
            )
            batch_overflow = num_nodes + 1 >= N
            num_nodes[batch_overflow] = 0

        # Add new nodes to the current graph
        # starting at num_nodes
        nodes[B_idx, num_nodes[B_idx]] = x[B_idx]

        # E.g. add self edges and normal weights
        for e in self.edge_selectors:
            adj, weights = e.forward(nodes, adj, weights, num_nodes, B)
        adj[B_idx, num_nodes[B_idx].squeeze()] = 1
        # weights[B_idx, num_nodes[B_idx]] = 1

        # Thru network
        batch = Batch(x=nodes, adj=adj, edge_weight=weights, B=B, N=N)
        if self.gnn.sparse:
            batch = self.gnn.dense_to_sparse(batch)
        node_feats = self.gnn(batch)
        # Why is squeeze required here?
        mx = node_feats[B_idx, num_nodes[B_idx].squeeze()]
        assert torch.all(
            torch.isfinite(mx)
        ), "Got NaN in returned memory, try using tanh activation"

        num_nodes = num_nodes + 1

        return mx, (nodes, adj, weights, num_nodes)
