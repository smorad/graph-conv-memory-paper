import unittest
import torch
import ray_graph


class TestAdj(unittest.TestCase):
    g = ray_graph.RayObsGraph

    def setUp(self):
        self.adj = torch.zeros(2, 3, 3, dtype=torch.float32)
        self.nodes = torch.zeros(2, 3, 4, dtype=torch.float32)
        self.num_nodes = torch.tensor([[1, 2]]).long().T

    def test_add_self_edge(self):
        self.g.add_self_edge(None, self.adj, self.num_nodes),
        sol = torch.tensor(
            [[[0, 0, 0], [0, 1, 0], [0, 0, 0]], [[0, 0, 0], [0, 0, 0], [0, 0, 1]]]
        ).float()

        if torch.any(self.adj != sol):
            self.fail(f"\nactual:\n {self.adj}\nexpected:\n {sol}")

    def test_add_backedge(self):
        self.g.add_backedge(None, self.adj, self.num_nodes),
        sol = torch.tensor(
            [[[0, 0, 0], [0, 0, 0], [0, 0, 0]], [[0, 0, 0], [0, 0, 1], [0, 1, 0]]],
        )

        if torch.any(self.adj != sol):
            self.fail(f"\nactual:\n {self.adj}\nexpected:\n {sol}")

    def test_densify_graph(self):
        self.g.densify_graph(None, self.adj, self.num_nodes),
        sol = torch.tensor(
            [[[1, 1, 0], [1, 1, 0], [0, 0, 0]], [[1, 1, 1], [1, 1, 1], [1, 1, 1]]],
        ).float()

        if torch.any(self.adj != sol):
            self.fail(f"\nactual:\n {self.adj}\nexpected:\n {sol}")

    """
    def test_positional_encoding(self):
        self.g.add_time_positional_encoding(None, self.nodes, self.num_nodes)

        embed = torch.range(0, self.nodes.shape[-1] - 1)
        sol = torch.sin(self.nodes / (10000 ** (2 * embed / embed.shape[0])))

        if torch.any(self.nodes != sol):
            self.fail(f"\nactual:\n {self.nodes}\nexpected:\n {sol}")
    """

    def test_index_select(self):
        nodes0 = torch.arange(24).reshape(2, 3, 4)
        nodes1 = torch.arange(24).reshape(2, 3, 4)

        flat = torch.ones(2, 4, dtype=torch.long)
        outs = []

        for batch in range(self.nodes.shape[0]):
            outs.append(nodes0[batch, self.num_nodes[batch]])
            nodes0[batch, self.num_nodes[batch]] = flat[batch]

        # It's critical both these vectors are 1D
        idx_0 = torch.arange(self.num_nodes.shape[0])
        idx_1 = self.num_nodes.squeeze()
        nodes1[idx_0, idx_1] = flat[idx_0]

        if torch.any(nodes0 != nodes1):
            self.fail(f"\nactual:\n {nodes1}\nexpected:\n {nodes0}")


if __name__ == "__main__":
    unittest.main()
