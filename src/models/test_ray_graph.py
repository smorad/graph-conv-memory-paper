import unittest
import torch
import ray_graph


class TestAdj(unittest.TestCase):
    g = ray_graph.RayObsGraph

    def setUp(self):
        self.adj = torch.zeros(2, 3, 3, dtype=torch.float32)
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


if __name__ == "__main__":
    unittest.main()
