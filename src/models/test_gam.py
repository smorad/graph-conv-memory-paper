import unittest
import torch
from gam import GNN, DenseGAM
import torch_geometric
from edge_selectors.temporal import TemporalBackedge
from edge_selectors.distance import EuclideanEdge, CosineEdge
from edge_selectors.dense import DenseEdge
from edge_selectors.bernoulli import BernoulliEdge
import torchviz


class TestDenseGAME2E(unittest.TestCase):
    def setUp(self):
        torch.autograd.set_detect_anomaly(True)
        feats = 11
        batches = 5
        N = 10
        self.g = GNN(
            input_size=feats,
            graph_size=N,
            hidden_size=feats,
            activation=torch.nn.ReLU,
            conv_type=torch_geometric.nn.DenseGraphConv,
        )
        for layer in self.g.layers:
            if layer.__class__ == self.g.conv_type:
                layer.lin_root.weight[:] = torch.diag(
                    torch.ones(layer.lin_root.weight.shape[-1])
                )
                layer.lin_root.bias[:] = 0.0
                layer.lin_rel.weight[:] = torch.diag(
                    torch.ones(layer.lin_root.weight.shape[-1])
                )
        self.s = DenseGAM(self.g)

        self.nodes = torch.zeros((batches, N, feats), dtype=torch.float)
        self.all_obs = [
            1 * torch.ones(batches, feats),
            2 * torch.ones(batches, feats),
            3 * torch.ones(batches, feats),
        ]
        self.adj = torch.zeros(batches, N, N)
        self.weights = torch.ones(batches, N, N)
        self.num_nodes = torch.zeros(batches, dtype=torch.long)

    def test_e2e_self_edge(self):
        (nodes, adj, weights, num_nodes) = (
            self.nodes,
            self.adj,
            self.weights,
            self.num_nodes,
        )
        # First iter
        # Zeroth row of graph should be 1111...
        # Output should be 1111...
        obs = self.all_obs[0].clone()
        out, (nodes, adj, weights, num_nodes) = self.s(
            obs, (nodes, adj, weights, num_nodes)
        )
        if torch.any(out != self.all_obs[0]):
            self.fail(f"out: {out} != {self.all_obs[0]}")

        desired_nodes = torch.cat(
            (self.all_obs[0].unsqueeze(1), torch.zeros(5, 9, 11)), dim=1
        )
        if torch.any(nodes != desired_nodes):
            self.fail(f"out: {nodes} != {desired_nodes}")

        # Second iter
        # Rows 1111 and 2222...
        # Output should be 2222
        obs = self.all_obs[1].clone()
        out, (nodes, adj, weights, num_nodes) = self.s(
            obs, (nodes, adj, weights, num_nodes)
        )
        if torch.any(out != self.all_obs[1]):
            self.fail(f"out: {out} != {self.all_obs[1]}")

        # Third iter
        # Rows 1111 and 2222 and 3333...
        # Output should be 3333
        obs = self.all_obs[2].clone()
        out, (nodes, adj, weights, num_nodes) = self.s(
            obs, (nodes, adj, weights, num_nodes)
        )
        if torch.any(out != self.all_obs[2]):
            self.fail(f"out: {out} != {self.all_obs[1]}")


class TestDenseGAM(unittest.TestCase):
    def setUp(self):
        torch.autograd.set_detect_anomaly(True)
        feats = 11
        batches = 5
        N = 10
        self.g = GNN(input_size=feats, graph_size=N, hidden_size=feats)
        self.s = DenseGAM(self.g)

        # Now do it in a loop to make sure grads propagate
        self.optimizer = torch.optim.Adam(self.s.parameters(), lr=0.005)

        self.nodes = torch.arange(batches * N * feats, dtype=torch.float).reshape(
            batches, N, feats
        )
        self.obs = torch.ones(batches, feats)
        self.adj = torch.zeros(batches, N, N)
        self.weights = torch.ones(batches, N, N)
        self.num_nodes = torch.zeros(batches, dtype=torch.long)

    def test_grad_prop(self):
        self.g = GNN(11, 11, 11, test=True)
        self.s = DenseGAM(self.g)
        out, (nodes, adj, weights, num_nodes) = self.s(
            self.obs, (self.nodes, self.adj, self.weights, self.num_nodes)
        )
        loss = torch.norm(out)
        dot = torchviz.make_dot(loss, params=dict(self.s.named_parameters()))
        # Make sure gradients make it all the way thru node_feats
        self.assertTrue("grad_test_var" in dot.source)

    def test_zeroth_entry(self):
        # Ensure first obs ends up in nodes matrix
        _, (nodes, adj, weights, num_nodes) = self.s(
            self.obs, (self.nodes, self.adj, self.weights, self.num_nodes)
        )
        if torch.any(nodes[:, 0] != self.obs):
            self.fail(f"{nodes[:,0]} != {self.obs}")
        # Ensure only self edges
        adj = torch.zeros_like(self.adj, dtype=torch.long)
        if torch.any(self.adj != adj):
            self.fail(f"adj: {adj} != {self.adj}")

    def test_first_entry(self):
        _, (nodes, adj, weights, num_nodes) = self.s(
            self.obs, (self.nodes, self.adj, self.weights, self.num_nodes)
        )
        _, (nodes, adj, weights, num_nodes) = self.s(
            self.obs, (nodes, adj, weights, num_nodes)
        )
        if torch.any(nodes[:, 1] != self.obs):
            self.fail(f"{nodes[:,0]} != {self.obs}")

    def test_propagation(self):
        out, (nodes, adj, weights, num_nodes) = self.s(
            self.obs, (self.nodes, self.adj, self.weights, self.num_nodes)
        )
        if torch.all(out == self.obs):
            self.fail(f"{out} == {self.obs}")

    def test_dense_learn(self):
        feats = 11
        batches = 5
        T = 4
        N = 10
        losses = []
        for i in range(20):
            nodes = torch.arange(batches * N * feats, dtype=torch.float).reshape(
                batches, N, feats
            )
            obs = torch.ones(batches, feats)
            adj = torch.zeros(batches, N, N, dtype=torch.long)
            weights = torch.ones(batches, N, N)
            num_nodes = torch.zeros(batches, dtype=torch.long)

            self.s.zero_grad()
            hidden = (nodes, adj, weights, num_nodes)
            for t in range(T):
                obs, hidden = self.s(obs, hidden)

            loss = torch.norm(obs)
            loss.backward()
            losses.append(loss)

            self.optimizer.step()

        if not losses[-1] < losses[0]:
            self.fail(f"Final loss {losses[-1]} not better than init loss {losses[0]}")


class TestSparseGAM(unittest.TestCase):
    def setUp(self):
        feats = 11
        batches = 5
        N = 10
        self.g = GNN(
            input_size=feats,
            graph_size=N,
            hidden_size=feats,
            sparse=True,
            conv_type=torch_geometric.nn.GCNConv,
        )
        self.s = DenseGAM(self.g)

        # Now do it in a loop to make sure grads propagate
        self.optimizer = torch.optim.Adam(self.s.parameters(), lr=0.005)

        self.nodes = torch.arange(batches * N * feats, dtype=torch.float).reshape(
            batches, N, feats
        )
        self.obs = torch.ones(batches, feats)
        self.adj = torch.zeros(batches, N, N, dtype=torch.long)
        self.weights = torch.ones(batches, N, N)
        self.num_nodes = torch.zeros(batches, dtype=torch.long)

    def test_zeroth_entry(self):
        # Ensure first obs ends up in nodes matrix
        _, (nodes, adj, weights, num_nodes) = self.s(
            self.obs, (self.nodes, self.adj, self.weights, self.num_nodes)
        )
        if torch.any(nodes[:, 0] != self.obs):
            self.fail(f"{nodes[:,0]} != {self.obs}")

    def test_first_entry(self):
        _, (nodes, adj, weights, num_nodes) = self.s(
            self.obs, (self.nodes, self.adj, self.weights, self.num_nodes)
        )
        _, (nodes, adj, weights, num_nodes) = self.s(
            self.obs, (nodes, adj, weights, num_nodes)
        )
        if torch.any(nodes[:, 1] != self.obs):
            self.fail(f"{nodes[:,0]} != {self.obs}")

    def test_dense_to_sparse(self):
        B = self.nodes.shape[0]
        N = self.nodes.shape[1]
        dense_batch = torch_geometric.data.Batch(
            x=self.nodes, adj=self.adj, edge_weight=self.weights, B=B, N=N
        )
        sparse_batch = self.g.dense_to_sparse(dense_batch)
        new_nodes = torch_geometric.utils.to_dense_batch(
            x=sparse_batch.x, batch=sparse_batch.batch, max_num_nodes=sparse_batch.N
        )[0]
        new_adj = torch_geometric.utils.to_dense_adj(
            edge_index=sparse_batch.edge_index,
            batch=sparse_batch.batch,
            max_num_nodes=sparse_batch.N,
        )[0]

        if torch.any(new_nodes != dense_batch.x):
            self.fail(f"x: {new_nodes} != {dense_batch.x}")

        if torch.any(new_adj != dense_batch.adj):
            self.fail(f"adj: {new_adj} != {dense_batch.adj}")

    def test_dense_to_sparse_sparse_to_dense(self):
        B = self.nodes.shape[0]
        N = self.nodes.shape[1]
        batch = torch_geometric.data.Batch(
            x=self.nodes, adj=self.adj, edge_weight=self.weights, B=B, N=N
        )
        sparse_batch = self.g.dense_to_sparse(batch)
        dense_batch = self.g.sparse_to_dense(sparse_batch)

        if torch.any(dense_batch.x != batch.x):
            self.fail(f"x: {dense_batch.x} != {batch.x}")

        if torch.any(dense_batch.adj != batch.adj):
            self.fail(f"adj: {dense_batch.adj} != {batch.adj}")

    """
    def test_dense_to_sparse_sparse_to_dense_with_weights(self):
        B = self.nodes.shape[0]
        N = self.nodes.shape[1]
        self.weights = torch.rand(B,N,N)
        batch = torch_geometric.data.Batch(x=self.nodes, adj=self.adj, edge_weight=self.weights, B=B, N=N)
        sparse_batch = self.g.dense_to_sparse(batch)
        dense_batch = self.g.sparse_to_dense(sparse_batch)

        if torch.any(dense_batch.x != batch.x):
            self.fail(f'x: {dense_batch.x} != {batch.x}')

        if torch.any(dense_batch.adj!= batch.adj):
            self.fail(f'adj: {dense_batch.adj} != {batch.adj}')
        if torch.any(dense_batch.edge_weight != batch.edge_weight):
            self.fail(f'weight: {dense_batch.weight} != {batch.weight}')
    """

    def test_propagation(self):
        out, (nodes, adj, weights, num_nodes) = self.s(
            self.obs, (self.nodes, self.adj, self.weights, self.num_nodes)
        )
        if torch.all(out == self.obs):
            self.fail(f"{out} == {self.obs}")

    def test_sparse_learn(self):
        feats = 11
        batches = 5
        T = 4
        N = 10
        losses = []
        for i in range(20):
            nodes = torch.arange(batches * N * feats, dtype=torch.float).reshape(
                batches, N, feats
            )
            obs = torch.ones(batches, feats)
            adj = torch.zeros(batches, N, N, dtype=torch.long)
            weights = torch.ones(batches, N, N)
            num_nodes = torch.zeros(batches, dtype=torch.long)

            self.s.zero_grad()
            hidden = (nodes, adj, weights, num_nodes)
            for t in range(T):
                obs, hidden = self.s(obs, hidden)

            loss = torch.norm(obs)
            loss.backward()
            losses.append(loss)

            self.optimizer.step()

        self.assertTrue(losses[-1] < losses[0])


class TestTemporalEdge(unittest.TestCase):
    def setUp(self):
        feats = 11
        batches = 5
        N = 10
        self.g = GNN(feats, feats)
        self.s = DenseGAM(self.g, edge_selectors=[TemporalBackedge(num_hops=1)])

        # Now do it in a loop to make sure grads propagate
        self.optimizer = torch.optim.Adam(self.s.parameters(), lr=0.005)

        self.nodes = torch.arange(batches * N * feats, dtype=torch.float).reshape(
            batches, N, feats
        )
        self.obs = torch.ones(batches, feats)
        self.adj = torch.zeros(batches, N, N, dtype=torch.long)
        self.weights = torch.ones(batches, N, N)
        self.num_nodes = torch.zeros(batches, dtype=torch.long)

    def test_two_nodes(self):
        _, (nodes, adj, weights, num_nodes) = self.s(
            self.obs, (self.nodes, self.adj, self.weights, self.num_nodes)
        )
        _, (nodes, adj, weights, num_nodes) = self.s(
            self.obs, (nodes, adj, weights, num_nodes)
        )
        tgt_adj = torch.zeros_like(adj, dtype=torch.long)
        tgt_adj[:, 0, 1] = 1
        tgt_adj[:, 1, 0] = 1
        # Also add self edges
        if torch.any(tgt_adj != self.adj):
            self.fail(f"{tgt_adj} != {self.adj}")


class TestDistanceEdge(unittest.TestCase):
    def setUp(self):
        feats = 11
        batches = 5
        N = 10
        self.g = GNN(feats, feats)
        self.s = DenseGAM(self.g, edge_selectors=[EuclideanEdge(max_distance=1)])

        self.nodes = torch.zeros(batches, N, feats, dtype=torch.float)
        self.obs = torch.zeros(batches, feats)
        self.adj = torch.zeros(batches, N, N, dtype=torch.long)
        self.weights = torch.ones(batches, N, N)
        self.num_nodes = torch.ones(batches, dtype=torch.long)

    def test_zero_dist(self):
        # Start num_nodes = 1
        _, (nodes, adj, weights, num_nodes) = self.s(
            self.obs, (self.nodes, self.adj, self.weights, self.num_nodes)
        )
        tgt_adj = torch.zeros_like(adj, dtype=torch.long)
        # It adds self edge
        tgt_adj[:, 1, 1] = 1
        tgt_adj[:, 0, 1] = 1
        tgt_adj[:, 1, 0] = 1

        # TODO: Ensure not off by one
        if torch.any(tgt_adj != adj):
            self.fail(f"{tgt_adj} != {self.adj}")

    def test_one_dist(self):
        # Start num_nodes = 1
        self.obs = torch.ones_like(self.obs)
        _, (nodes, adj, weights, num_nodes) = self.s(
            self.obs, (self.nodes, self.adj, self.weights, self.num_nodes)
        )
        tgt_adj = torch.zeros_like(adj, dtype=torch.long)
        # Adds self edge
        tgt_adj[:, 1, 1] = 1
        if torch.any(tgt_adj != adj):
            self.fail(f"{tgt_adj} != {self.adj}")

    def test_cosine(self):
        self.s = DenseGAM(self.g, edge_selectors=[CosineEdge(max_distance=1)])
        _, (nodes, adj, weights, num_nodes) = self.s(
            self.obs, (self.nodes, self.adj, self.weights, self.num_nodes)
        )


class TestDenseEdge(unittest.TestCase):
    def setUp(self):
        feats = 11
        batches = 5
        N = 10
        self.g = GNN(feats, feats)
        self.s = DenseGAM(self.g, edge_selectors=[DenseEdge()])

        self.nodes = torch.zeros(batches, N, feats, dtype=torch.float)
        self.obs = torch.zeros(batches, feats)
        self.adj = torch.zeros(batches, N, N, dtype=torch.long)
        self.weights = torch.ones(batches, N, N)
        self.num_nodes = torch.zeros(batches, dtype=torch.long)

    def test_two_obs(self):
        # Start num_nodes = 1
        _, (nodes, adj, weights, num_nodes) = self.s(
            self.obs, (self.nodes, self.adj, self.weights, self.num_nodes)
        )
        _, (nodes, adj, weights, num_nodes) = self.s(
            self.obs, (nodes, adj, weights, num_nodes)
        )
        tgt_adj = torch.zeros_like(adj, dtype=torch.long)
        # It adds self edge
        tgt_adj[:, 1, 1] = 1
        tgt_adj[:, 0, 0] = 1
        tgt_adj[:, 1, 0] = 1
        tgt_adj[:, 0, 1] = 1

        # TODO: Ensure not off by one
        if torch.any(tgt_adj != adj):
            self.fail(f"{tgt_adj} != {self.adj}")


class TestBernoulliEdge(unittest.TestCase):
    def setUp(self):
        torch.autograd.set_detect_anomaly(True)
        feats = 11
        batches = 5
        N = 10
        self.g = GNN(input_size=feats, graph_size=N, hidden_size=feats, test="adj")
        self.s = DenseGAM(self.g, edge_selectors=[BernoulliEdge(11)])

        # Now do it in a loop to make sure grads propagate
        self.optimizer = torch.optim.Adam(self.s.parameters(), lr=0.005)

        self.nodes = torch.arange(batches * N * feats, dtype=torch.float).reshape(
            batches, N, feats
        )
        self.obs = torch.ones(batches, feats)
        self.adj = torch.zeros(batches, N, N)
        self.weights = torch.ones(batches, N, N)
        self.num_nodes = torch.zeros(batches, dtype=torch.long)

    def test_grad_prop(self):
        out, (nodes, adj, weights, num_nodes) = self.s(
            self.obs, (self.nodes, self.adj, self.weights, self.num_nodes)
        )
        loss = torch.norm(out)
        dot = torchviz.make_dot(loss, params=dict(self.s.named_parameters()))
        # Make sure gradients make it all the way thru node_feats
        self.assertTrue("grad_test_var" in dot.source)


if __name__ == "__main__":
    unittest.main()
