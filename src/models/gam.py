import torch
import torch_geometric
from torch_geometric.data import Data, Batch
from typing import List, Tuple, Union, Any, Dict


class GNN(torch.nn.Module):
    def hidden_block(self, hidden_size, activation, conv_type, attn_heads):
        return [
            conv_type(hidden_size, hidden_size),
            activation(),
            torch_geometric.nn.BatchNorm(hidden_size),
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
        activation: torch.nn.Module = torch.nn.ReLU,
    ):
        super().__init__()

        first = [
            conv_type(input_size, hidden_size),
            activation(),
            torch_geometric.nn.BatchNorm(hidden_size),
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

        # TODO: Make this work
        raise NotImplementedError()
        """
        offset, row, col = torch.nonzero(batch.adj > 0).t()
        edge_weight = adj[offset, row, col]
        row += offset * n
        col += offset * n
        edge_index = torch.stack([row, col], dim=0)
        x = batch.x.view(batch.the_num_graphs * n, batch.x.shape[-1])
        batch = torch.arange(0, batch.the_num_graphs).view(-1, 1).repeat(1, n).view(-1)

        return batch
        """

    def forward(self, batch: Batch) -> Batch:
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
        T = time batch size
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

        B = x.shape[0]
        B_idx = torch.arange(B)

        # Add new nodes to the current graph
        # starting at num_nodes
        nodes[B_idx, num_nodes[B_idx]] = x[B_idx]

        # E.g. add self edges and normal weights
        for e in self.edge_selectors:
            adj, weights = e.forward(nodes, adj, weights, num_nodes, B)
        # adj[B_idx, num_nodes[B_idx]] = 1
        # weights[B_idx, num_nodes[B_idx]] = 1

        # Thru network
        batch = Batch(x=nodes, adj=adj, edge_weights=weights)
        node_feats = self.gnn(batch)
        mx = node_feats[B_idx, num_nodes[B_idx].squeeze()]

        num_nodes = num_nodes + 1

        return mx, (nodes, adj, weights, num_nodes)


if __name__ == "__main__":
    feats = 11
    batches = 5
    time = 4
    n = 3
    N = 10
    g = GNN(feats, feats)
    s = DenseGAM(g)
    nodes = torch.arange(batches * N * feats, dtype=torch.float).reshape(
        batches, N, feats
    )
    obs = torch.zeros(batches, feats)
    adj = torch.zeros(batches, N, N, dtype=torch.long)
    weights = torch.zeros(batches, N, N)
    num_nodes = torch.zeros(batches, dtype=torch.long)

    torch.autograd.set_detect_anomaly(True)
    """
    out = obs
    for t in range(time):
        hidden = (nodes, adj, weights, num_nodes)
        out, hidden = s.forward(out, hidden)
    # Ensure backprop works
    loss = out.mean()
    loss.backward()
    """
    print(s)

    # Now do it in a loop to make sure grads propagate
    optimizer = torch.optim.SGD(s.parameters(), lr=0.1)
    i = 0
    while True:
        nodes = torch.arange(batches * N * feats, dtype=torch.float).reshape(
            batches, N, feats
        )
        obs = torch.zeros(batches, feats)
        adj = torch.zeros(batches, N, N, dtype=torch.long)
        weights = torch.zeros(batches, N, N)
        num_nodes = torch.zeros(batches, dtype=torch.long)
        s.zero_grad()
        loss = torch.tensor(0)
        for t in range(time):
            hidden = (nodes, adj, weights, num_nodes)
            obs, hidden = s(obs, hidden)
            loss = loss + obs.mean() ** 2

        if i % 10 == 0:
            print(loss)

        loss.backward()
        optimizer.step()
        i += 1
