import torch


class TemporalEdge(torch.nn.Module):
    """Add temporal bidirectional back edge, but only if we have >1 nodes
    E.g., node_{t} <-> node_{t-1}"""

    def __init__(self, num_hops=1, bidirectional=False):
        super().__init__()
        self.num_hops = num_hops
        self.bidirectional = bidirectional
        assert num_hops == 1, "Not yet implemented"

    def forward(self, nodes, edge_list, weights, B, T, t):
        # Assumes equal sized full episodes
        # Rather than fill all previous edges (T)
        # only fill out the ones for our newly added nodes (T to T + t)
        a = torch.arange(T, T + t, device=nodes.device)
        b = torch.arange(T + 1, T + t + 1, device=nodes.device)

        new_edges = torch.stack((b, a)).repeat(B, 1, 1)
        if self.bidirectional:
            out_edge = torch.stack((a, b)).repeat(B, 1, 1)
            new_edges = torch.cat((new_edges, out_edge), dim=-1)

        # TODO: We have edge 0,1 at t0 with single node
        all_edges = torch.cat((edge_list, new_edges), dim=-1)
        return all_edges, weights
