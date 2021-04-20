import unittest
import torch
from sparse_gam import SparseGAM
import torch_geometric
from sparse_edge_selectors.temporal import TemporalEdge
from sparse_edge_selectors.distance import EuclideanEdge


class Ident(torch.nn.Module):
    def forward(self, x, edge):
        return x


class Sum(torch_geometric.nn.MessagePassing):
    def forward(self, x, edge):
        return self.propagate(edge, x=x)

    def message(self, x_i, x_j):
        return x_i + x_j.sum(dim=0)


class TestSparseGAM(unittest.TestCase):
    def setUp(self):
        B = 3
        T = 4
        feat = 5
        t = 2

        self.nodes = torch.arange(1, B * T * feat + 1).reshape(B, T, feat)
        # Self edges
        # a = torch.arange(T)
        # edges = torch.meshgrid(a, a)
        a = torch.arange(T, dtype=torch.long)
        edges = torch.stack((a, a)).unsqueeze(0)
        # B, 2, T
        self.edges = edges.repeat(B, 1, 1)
        self.xs = torch.zeros(B, t, feat)

        self.gnn = torch_geometric.nn.Sequential(
            "x, edge_index", [(Ident(), "x, edge_index -> x")]
        )

        self.gam = SparseGAM(self.gnn)

    def test_simple(self):
        out, nodes, edge_list, weights = self.gam(self.xs, self.nodes, self.edges, None)
        desired = torch.zeros(3, 2, 5)
        if torch.any(out != desired):
            self.fail(f"{out} != {desired}")

    def test_first_run(self):
        self.nodes = torch.tensor([]).reshape(3, 0, 5)
        self.edges = torch.tensor([], dtype=torch.long).reshape(3, 2, 0)
        out, nodes, edge_list, weights = self.gam(self.xs, self.nodes, self.edges, None)

        desired_shape = (3, 2, 5)
        if nodes.shape != desired_shape:
            self.fail(f"{nodes.shape} != {desired_shape}")


class TestTemporalEdge(unittest.TestCase):
    def setUp(self):
        B = 3
        T = 4
        feat = 5
        t = 2

        self.nodes = torch.arange(1, B * T * feat + 1).reshape(B, T, feat)
        # Self edges
        self.edges = torch.tensor([], dtype=torch.long).reshape(3, 2, 0)
        # B, 2, T
        self.xs = torch.zeros(B, t, feat)

        self.gnn = torch_geometric.nn.Sequential(
            "x, edge_index", [(Sum(), "x, edge_index -> x")]
        )

        self.edge_selector = torch_geometric.nn.Sequential(
            "nodes, edge_list, weights, B, T, t",
            [
                (
                    TemporalEdge(1),
                    "nodes, edge_list, weights, B, T, t -> edge_list, weights",
                )
            ],
        )

        self.gam = SparseGAM(self.gnn, edge_selectors=self.edge_selector)

    def test_simple(self):
        out, nodes, edge_list, weights = self.gam(self.xs, self.nodes, self.edges, None)

        desired = torch.tensor(
            [
                [5, 6],
                [4, 5],
            ],
            dtype=torch.long,
        ).repeat(3, 1, 1)
        if torch.any(edge_list != desired):
            self.fail(f"{edge_list} != {desired}")


class TestDistanceEdge(unittest.TestCase):
    def setUp(self):
        B = 3
        T = 4
        feat = 5
        t = 2

        self.nodes = torch.arange(1, B * T * feat + 1).reshape(B, T, feat)
        self.edges = torch.tensor([], dtype=torch.long).reshape(3, 2, 0)
        # B, 2, T
        self.xs = torch.zeros(B, t, feat)

        self.gnn = torch_geometric.nn.Sequential(
            "x, edge_index", [(Sum(), "x, edge_index -> x")]
        )

        self.edge_selector = torch_geometric.nn.Sequential(
            "nodes, edge_list, weights, B, T, t",
            [
                (
                    EuclideanEdge(1),
                    "nodes, edge_list, weights, B, T, t -> edge_list, weights",
                )
            ],
        )

        self.gam = SparseGAM(self.gnn, edge_selectors=self.edge_selector)

    def test_simple(self):
        self.nodes[:, 0] = 0
        out, nodes, edge_list, weights = self.gam(self.xs, self.nodes, self.edges, None)

        desired = torch.tensor(
            [
                [4, 5, 5],
                [0, 0, 4],
            ],
            dtype=torch.long,
        ).repeat(3, 1, 1)
        if torch.any(edge_list != desired):
            self.fail(f"{edge_list} != {desired}")

    def test_staggered(self):
        self.nodes[0, 0] = 0
        self.nodes[1, 1] = 0
        self.nodes[2, 2] = 0
        out, nodes, edge_list, weights = self.gam(self.xs, self.nodes, self.edges, None)

        desired = torch.tensor(
            [
                [
                    [4, 5, 5],
                    [0, 0, 4],
                ],
                [
                    [4, 5, 5],
                    [1, 1, 4],
                ],
                [[4, 5, 5], [2, 2, 4]],
            ],
            dtype=torch.long,
        )
        if torch.any(edge_list != desired):
            self.fail(f"{edge_list} != {desired}")


if __name__ == "__main__":
    unittest.main()
