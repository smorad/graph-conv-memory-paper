from typing import List, Tuple, Any
import torch
import torch_geometric
from torch_geometric_temporal.signal.dynamic_graph_static_signal import (
    DynamicGraphStaticSignal,
)


class SparseGAM(torch.nn.Module):
    """Graph Associative Memory"""

    def __init__(
        self,
        gnn: torch.nn.Module,
        preprocessor: torch.nn.Sequential = None,
        edge_selectors: torch_geometric.nn.Sequential = None,
        graph_size: int = 128,
    ):
        super().__init__()

        self.gnn = gnn
        self.graph_size = graph_size
        self.preprocessor = preprocessor
        self.edge_selectors = edge_selectors

    def build_batch(
        self,
        nodes: torch.Tensor,
        edge_list: torch.Tensor,
        weights: torch.Tensor,
        B: int,
        T: int,
        t: int,
        max_hops: int,
    ) -> torch_geometric.data.Batch:

        # We add all nodes to all timesteps
        # but filter using the edges
        data_list = []

        for b in range(B):
            for tau in range(t):
                new_node_idx = T + tau
                graph_x = nodes[b].narrow(-2, 0, new_node_idx + 1)

                # TODO: Use k_hop_subgraph to improve performance

                # Prune edges if either edge a or b is outside
                # of the current nodes
                edge_mask = torch.sum(edge_list[b] <= new_node_idx, dim=0) == 2
                edge_mask = torch.stack((edge_mask, edge_mask))
                graph_edge = edge_list[b].masked_select(edge_mask).reshape(2, -1)
                if weights is not None:
                    graph_weight = weights[b, torch.arange(graph_edge.shape[-1])]
                    d = torch_geometric.data.Data(
                        x=graph_x,
                        edge_index=graph_edge,
                        edge_attr=graph_weight,
                        B=b,
                        new_idx=new_node_idx,
                        t=tau,
                        T=T,
                    )
                else:
                    d = torch_geometric.data.Data(
                        x=graph_x,
                        edge_index=graph_edge,
                        B=b,
                        new_idx=new_node_idx,
                        t=tau,
                        T=T,
                    )

                data_list.append(d)
        batch = torch_geometric.data.Batch.from_data_list(data_list)
        return batch

    def forward(
        self,
        x: torch.Tensor,
        nodes: torch.Tensor,
        edge_list: torch.Tensor,
        weights: torch.Tensor,
        max_hops: int = -1,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Add memores x to the graph, and return queries at x.
        B = batch size
        T = temporal input size
        t = temporal output size
        k = number of input edges
        j = number of output edges
        Inputs:
            x: [B,t,feat]
            hidden: (
                nodes: [B,T,feats]
                edge_list: [B,2,k]
                weights: [B,k,1]
            )
        Outputs:
            m(x): [B,t,feat]
            hidden: (
                nodes: [B,T+t,feats]
                edge_list: [B,2,j]
                weights: [B,j,1]
            )
        """
        B = nodes.shape[0]
        T = nodes.shape[1]
        t = x.shape[1]

        nodes = torch.cat((nodes, x), dim=-2)

        if self.edge_selectors is not None:
            edge_list, weights = self.edge_selectors(nodes, edge_list, weights, B, T, t)
        # Batching will collapse edges into a single [2,j], nodes into
        # [B * (T + t), feats], etc
        # So make sure we make all the changes we need to
        #  nodes/edges/weights before batching and gnn forward
        batch = self.build_batch(
            nodes, edge_list, weights, B, T, t, max_hops
        ).coalesce()
        # Preprocessor can be used to change obs_size to hidden_size
        # if needed
        if self.preprocessor:
            batch.x = self.preprocessor(batch.x)

        if weights is None:
            out = self.gnn(batch.x, batch.edge_index)
        else:
            out = self.gnn(batch.x, batch.edge_index, batch.edge_attr)

        # Shape [B*t,feat] -> [B,t,feat]
        # mx = batch.x[batch.ptr[1:] - 1].reshape(B, t, batch.x.shape[-1])
        mx = out[batch.ptr[1:] - 1].reshape(B, t, out.shape[-1])
        # Extract nodes
        # nodes = torch_geometric.utils.to_dense_batch(batch.x, batch.batch)[0]

        return mx, nodes, edge_list, weights
