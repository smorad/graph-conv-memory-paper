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

    def compute_prob_adj(self):
        """Compute a probabilistic adjacency matrix,
        where each A(i,j) = phi(i,j). Caching nn results
        in this way is much faster. We can randomly sample A(i,j)
        from bernoulli later"""
        pass

    def get_initial_state(self, adj_mats, B):
        pass

    def forward(self, nodes, adj_mats, num_nodes, state, B):
        """A(i,j) = Ber[phi(i || j), e]"""
        # a(b,i,j) = gumbel_softmax(phi(n(b,i) || n(b,j))) for i, j < num_nodes
        if not state:
            state = self.get_initial_state(adj_mats, B)

        for b in range(B):
            ii, jj = zip(*itertools.combinations(adj_mats[b].shape, 2))
            # TODO: Need to cache adj computation
            import pdb

            pdb.set_trace()

        """
        # (i || j) mat
        cat_mat = torch.zeros((batch, *nodes.shape[:-1], 2 *  nodes.shape[-1]))
        for b in range(batch):
            # View of the submatrices
            # our matrix has a fixed shape, but we are only concerned up to
            # the num_nodes'th element
            #
            # shape [num_nodes, feat]
            nodeview = nodes[b].narrow(dim=0, start=0, length=num_nodes[b]+1)
            # shape [num_nodes, num_nodes]
            adjview = adj_mats[b].narrow(dim=0, start=0, length=num_nodes[b]+1).narrow(dim=1, start=0, length=num_nodes[b]+1)
            # ((i0, i0, ... i1, i1, ... ... in, in), (j0, j0, ... ... jn, jn)
            cat_vects = tuple(zip(*tuple(itertools.permutations(nodeview, 2))))
            # i, j nodes
            ii = torch.stack(cat_vects[0])
            jj = torch.stack(cat_vects[1])
            # Reshape to row == (i || j)
            ii = i.reshape((i.shape[0] // 2, i.shape[1]))
            jj = j.reshape((j.shape[0] // 2, j.shape[1]))
        """

        """
        for b in range(batch):
            for i in range(num_nodes[b] + 1):
                for j in range(num_nodes[b] + 1):
                    cat_vect = torch.cat((nodes[b, i], nodes[b, j]))
                    p = self.edge_network(cat_vect)
                    # Gumbel expects logits, not probs
                    z = nn.functional.gumbel_softmax(p, hard=True)
                    adj_mats[b, i, j] = z[0]
        """
