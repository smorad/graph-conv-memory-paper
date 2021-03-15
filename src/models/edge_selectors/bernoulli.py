import torch
import itertools


class BernoulliEdge(torch.nn.Module):
    """Add temporal bidirectional back edge, but only if we have >1 nodes
    E.g., node_{t} <-> node_{t-1}"""

    def __init__(self, parent):
        super().__init__()
        self.parent = parent
        self.build_edge_network()

    def build_edge_network(self):
        """Builds a network to predict edges.
        Input: (i || j)
        Output: [p(edge), 1-p(edge)] in R^2
        """
        self.edge_network = torch.nn.Sequential(
            torch.nn.Linear(2 * self.parent.obs_dim, self.parent.obs_dim),
            torch.nn.ReLU(),
            # Output is [yes_edge, no_edge]
            torch.nn.Linear(self.parent.obs_dim, 2),
            # We want output to be -inf, inf so do not
            # use ReLU as final activation
        )

    def compute_logits(self, node_view, adj_view, num_node, state):
        """Compute logits using a forward pass of the edge_network for a single graph.
        Logits will be of shape [nodes, 2], where the last dim
        corresponds to logits: [yes_edge, no_edge].

        The computation occurs between the currently added node and
        all previously added nodes. Results are stored in state."""
        left_half_mat = node_view[num_node].repeat(node_view.shape[0], 1)
        right_half_mat = node_view
        # Shape [nodes, 2*node_feats]
        network_in = torch.cat((left_half_mat, right_half_mat), dim=1)
        # Batch inference, returns shape [nodes, 2]
        logits = self.edge_network(network_in)
        # Undirected edges, cache edge network forward passes
        state[: num_node + 1, num_node] = logits
        state[num_node, : num_node + 1] = logits

    def get_initial_state(self, states, adj_mats, B):
        for mat in adj_mats:
            states.append(
                torch.zeros((*mat.shape, 2), dtype=torch.float, device=mat.device)
            )
        self.edge_network = self.edge_network.to(adj_mats[0].device)

        return states

    def forward(self, nodes, adj_mats, num_nodes, state, B):
        """A(i,j) = Ber[phi(i || j), e]"""
        # a(b,i,j) = gumbel_softmax(phi(n(b,i) || n(b,j))) for i, j < num_nodes
        # First run
        if len(state) == 0:
            state = self.get_initial_state(state, adj_mats, B)

        node_views, adj_views = self.parent.get_views(nodes, adj_mats, num_nodes)

        # Compute logits for all nodes to curr node
        # and place results in state adj matrix
        for b in range(B):
            # Concatenate left half of stacked current_node
            # with right half of all current nodes
            self.compute_logits(
                node_views[b], adj_views[b], num_nodes[b].squeeze(), state[b]
            )

            # Now resample the entire logits adj mat using gumbel max trick
            probs = torch.nn.functional.gumbel_softmax(state[b], hard=True, dim=-1)
            adj_views[b] = probs
