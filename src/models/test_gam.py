import unittest
import torch
import gam
from gam import DenseGAM
import torch_geometric
from edge_selectors.temporal import TemporalBackedge
from edge_selectors.distance import EuclideanEdge, CosineEdge, SpatialEdge
from edge_selectors.dense import DenseEdge
from edge_selectors.bernoulli import BernoulliEdge
import torchviz


class TestGAMDirection(unittest.TestCase):
    def setUp(self):
        torch.autograd.set_detect_anomaly(True)
        feats = 11
        batches = 1
        N = 10
        conv_type = torch_geometric.nn.DenseGraphConv
        self.g = torch_geometric.nn.Sequential(
            "x, adj, weights, B, N",
            [
                (conv_type(feats, feats), "x, adj -> x"),
                (torch.nn.ReLU()),
            ],
        )
        for layer in list(self.g.modules())[1]:
            if layer.__class__ == conv_type:
                layer.lin_root.weight = torch.nn.Parameter(
                    torch.diag(torch.zeros(layer.lin_root.weight.shape[-1]))
                )
                layer.lin_root.bias = torch.nn.Parameter(
                    torch.zeros_like(layer.lin_root.bias)
                )
                layer.lin_rel.weight = torch.nn.Parameter(
                    torch.diag(torch.ones(layer.lin_root.weight.shape[-1]))
                )
        self.s = DenseGAM(self.g)

        self.nodes = torch.arange((batches * N * feats), dtype=torch.float).reshape(
            batches, N, feats
        )
        self.all_obs = [
            1 * torch.ones(batches, feats),
            2 * torch.ones(batches, feats),
            3 * torch.ones(batches, feats),
        ]
        self.adj = torch.zeros(batches, N, N)
        self.weights = torch.ones(batches, N, N)
        self.num_nodes = torch.zeros(batches, dtype=torch.long)

    def test_gam_direction(self):
        # Only get neighbor
        self.adj[:, 0, 3] = 1
        # list(self.g.modules())[1][0].lin_rel(torch.matmul(self.adj, self.nodes))

        out, (nodes, adj, weights, num_nodes) = self.s(
            self.all_obs[0], (self.nodes, self.adj, self.weights, self.num_nodes)
        )
        # neighbor
        # flows from 3 => 0, neighbor => root
        # root = i, neighbor = j
        # j should be < i
        desired = torch.arange(3 * 11, 4 * 11, dtype=torch.float)
        if not torch.all(self.nodes[0, 3] == desired):
            self.fail(f"{self.nodes[0,3]} != {desired}")


class TestDenseGAME2E(unittest.TestCase):
    def setUp(self):
        torch.autograd.set_detect_anomaly(True)
        feats = 11
        batches = 5
        N = 10
        conv_type = torch_geometric.nn.DenseGraphConv
        self.g = torch_geometric.nn.Sequential(
            "x, adj, weights, B, N",
            [
                (conv_type(feats, feats), "x, adj -> x"),
                (torch.nn.ReLU()),
                (conv_type(feats, feats), "x, adj -> x"),
                (torch.nn.ReLU()),
            ],
        )
        for layer in list(self.g.modules())[1]:
            if layer.__class__ == conv_type:
                layer.lin_root.weight = torch.nn.Parameter(
                    torch.diag(torch.ones(layer.lin_root.weight.shape[-1]))
                )
                layer.lin_root.bias = torch.nn.Parameter(
                    torch.zeros_like(layer.lin_root.bias)
                )
                layer.lin_rel.weight = torch.nn.Parameter(
                    torch.diag(torch.ones(layer.lin_root.weight.shape[-1]))
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
        conv_type = torch_geometric.nn.DenseGCNConv
        self.g = torch_geometric.nn.Sequential(
            "x, adj, weights, B, N",
            [
                (conv_type(feats, feats), "x, adj -> x"),
                (torch.nn.ReLU()),
                (conv_type(feats, feats), "x, adj -> x"),
                (torch.nn.ReLU()),
            ],
        )
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
        self.g.grad_test_var = torch.nn.Parameter(torch.tensor([1.0]))
        self.nodes = self.nodes * self.g.grad_test_var
        self.assertTrue(self.nodes.requires_grad)
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

    def test_num_nodes_entry(self):
        _, (nodes, adj, weights, num_nodes) = self.s(
            self.obs, (self.nodes, self.adj, self.weights, self.num_nodes)
        )
        B_idx = torch.arange(num_nodes.shape[0])
        if torch.any(nodes[B_idx, num_nodes - 1] != self.obs):
            self.fail(f"{nodes[:,num_nodes]} != {self.obs}")

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
        conv_type = torch_geometric.nn.DenseGCNConv
        self.g = torch_geometric.nn.Sequential(
            "x, adj, weights, B, N",
            [
                (conv_type(feats, feats), "x, adj -> x"),
                (torch.nn.ReLU()),
                (conv_type(feats, feats), "x, adj -> x"),
                (torch.nn.ReLU()),
            ],
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
        sparse_batch = gam.dense_to_sparse(dense_batch)
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
        sparse_batch = gam.dense_to_sparse(batch)
        dense_batch = gam.sparse_to_dense(sparse_batch)

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
        conv_type = torch_geometric.nn.DenseGCNConv
        self.g = torch_geometric.nn.Sequential(
            "x, adj, weights, B, N",
            [
                (conv_type(feats, feats), "x, adj -> x"),
                (torch.nn.ReLU()),
                (conv_type(feats, feats), "x, adj -> x"),
                (torch.nn.ReLU()),
            ],
        )
        self.s = DenseGAM(self.g, edge_selectors=TemporalBackedge(num_hops=1))

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
        # tgt_adj[:, 0, 1] = 1
        tgt_adj[:, 1, 0] = 1
        # Also add self edges
        if torch.any(tgt_adj != adj):
            self.fail(f"{tgt_adj} != {adj}")


class TestDistanceEdge(unittest.TestCase):
    def setUp(self):
        feats = 11
        batches = 5
        N = 10
        conv_type = torch_geometric.nn.DenseGCNConv
        self.g = torch_geometric.nn.Sequential(
            "x, adj, weights, B, N",
            [
                (conv_type(feats, feats), "x, adj -> x"),
                (torch.nn.ReLU()),
                (conv_type(feats, feats), "x, adj -> x"),
                (torch.nn.ReLU()),
            ],
        )
        self.s = DenseGAM(self.g, edge_selectors=EuclideanEdge(max_distance=1))

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
        if torch.any(tgt_adj != adj):
            self.fail(f"{tgt_adj} != {self.adj}")

    def test_cosine(self):
        self.s = DenseGAM(self.g, edge_selectors=CosineEdge(max_distance=1))
        _, (nodes, adj, weights, num_nodes) = self.s(
            self.obs, (self.nodes, self.adj, self.weights, self.num_nodes)
        )


class TestDenseEdge(unittest.TestCase):
    def setUp(self):
        feats = 11
        batches = 5
        N = 10
        conv_type = torch_geometric.nn.DenseGCNConv
        self.g = torch_geometric.nn.Sequential(
            "x, adj, weights, B, N",
            [
                (conv_type(feats, feats), "x, adj -> x"),
                (torch.nn.ReLU()),
                (conv_type(feats, feats), "x, adj -> x"),
                (torch.nn.ReLU()),
            ],
        )
        self.s = DenseGAM(self.g, edge_selectors=DenseEdge())

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
        conv_type = torch_geometric.nn.DenseGCNConv
        self.g = torch_geometric.nn.Sequential(
            "x, adj, weights, B, N",
            [
                (conv_type(feats, feats), "x, adj -> x"),
                (torch.nn.ReLU()),
                (conv_type(feats, feats), "x, adj -> x"),
                (torch.nn.ReLU()),
            ],
        )
        self.s = DenseGAM(self.g, edge_selectors=BernoulliEdge(feats))

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
        self.g.grad_test_var = torch.nn.Parameter(torch.tensor([1.0]))
        self.nodes = self.nodes * self.g.grad_test_var
        self.assertTrue(self.nodes.requires_grad)

        out, (nodes, adj, weights, num_nodes) = self.s(
            self.obs, (self.nodes, self.adj, self.weights, self.num_nodes)
        )
        # First run has no gradient as no edges to be made
        out, (nodes, adj, weights, num_nodes) = self.s(
            self.obs, (nodes, adj, weights, num_nodes)
        )

        loss = torch.norm(out)
        dot = torchviz.make_dot(loss, params=dict(self.s.named_parameters()))
        # Make sure gradients make it all the way thru node_feats
        self.assertTrue("grad_test_var" in dot.source)

    def test_grad_prop2(self):
        out, (nodes, adj, weights, num_nodes) = self.s(
            self.obs, (self.nodes, self.adj, self.weights, self.num_nodes)
        )
        # First run has no gradient as no edges to be made
        out, (nodes, adj, weights, num_nodes) = self.s(
            self.obs, (nodes, adj, weights, num_nodes)
        )
        adj, weights = self.s.edge_selectors(nodes, adj, weights, num_nodes, 5)
        self.assertTrue(adj.grad_fn, "Adj has no gradient")
        self.assertTrue(weights.grad_fn, "Weight has no gradient")

    def test_backwards(self):
        nodes, adj, weights, num_nodes = (
            self.nodes,
            self.adj,
            self.weights,
            self.num_nodes + 1,
        )
        out, (nodes, adj, weights, num_nodes) = self.s(
            self.obs, (nodes, adj, weights, num_nodes)
        )
        self.s.edge_selectors.zero_grad()
        nodes = torch.rand_like(self.nodes) * 0.00001
        adj, weights = self.s.edge_selectors(nodes, adj, weights, num_nodes, 5)
        adj.mean().backward()
        self.optimizer.step()

    def test_reg_loss(self):
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
            adj = torch.zeros(batches, N, N)
            weights = torch.ones(batches, N, N)
            num_nodes = torch.zeros(batches, dtype=torch.long)

            if i == 0:
                continue

            hidden = (nodes, adj, weights, num_nodes)
            for t in range(T):
                obs, hidden = self.s(obs, hidden)

            loss = self.s.edge_selectors.compute_full_loss(nodes, nodes.shape[0])
            loss.backward()
            losses.append(loss)

            self.optimizer.step()
            # Must zero grad AND reset density
            self.optimizer.zero_grad()

        if not losses[-1] < losses[0]:
            self.fail(f"Final loss {losses[-1]} not better than init loss {losses[0]}")

    def test_logit_index(self):
        # Given 3 nodes, make sure we compare node 3 to nodes 1,2
        out, (nodes, adj, weights, num_nodes) = self.s(
            self.obs, (self.nodes, self.adj, self.weights, self.num_nodes)
        )
        # First run has no gradient as no edges to be made
        out, (nodes, adj, weights, num_nodes) = self.s(
            self.obs, (nodes, adj, weights, num_nodes)
        )
        adj, weights = self.s.edge_selectors(nodes, adj, weights, num_nodes, 5)
        self.assertTrue(adj.grad_fn, "Adj has no gradient")
        self.assertTrue(weights.grad_fn, "Weight has no gradient")

    def test_validate_logits(self):
        # TODO: Fix test
        return
        nodes = self.nodes
        adj = self.adj
        weights = self.weights
        num_nodes = self.num_nodes
        adj, weights = self.s.edge_selectors.compute_logits(
            nodes, weights.clone(), num_nodes, 5
        )
        adj2, weights2 = self.s.edge_selectors.compute_logits2(
            nodes, weights.clone(), num_nodes, 5
        )
        if torch.any(adj != adj2):
            self.fail(f"{adj} != {adj2}")

        if torch.any(weights != weights):
            self.fail(f"{weights} != {weights2}")


class TestSpatialEdge(unittest.TestCase):
    def setUp(self):
        feats = 11
        batches = 5
        N = 10
        conv_type = torch_geometric.nn.DenseGCNConv
        self.g = torch_geometric.nn.Sequential(
            "x, adj, weights, B, N",
            [
                (conv_type(feats, feats), "x, adj -> x"),
                (torch.nn.ReLU()),
                (conv_type(feats, feats), "x, adj -> x"),
                (torch.nn.ReLU()),
            ],
        )
        self.slice = slice(0, 2)
        self.s = DenseGAM(self.g, edge_selectors=SpatialEdge(1, self.slice))

        self.nodes = torch.zeros(batches, N, feats, dtype=torch.float)
        self.obs = torch.zeros(batches, feats)
        self.adj = torch.zeros(batches, N, N, dtype=torch.float)
        self.weights = torch.ones(batches, N, N)
        self.num_nodes = torch.ones(batches, dtype=torch.long)

    def test_zero_dist(self):
        # Start num_nodes = 1
        self.nodes[:] = 1
        self.nodes[:, 0:2, self.slice] = 0
        _, (nodes, adj, weights, num_nodes) = self.s(
            self.obs, (self.nodes, self.adj, self.weights, self.num_nodes)
        )
        tgt_adj = torch.zeros_like(adj, dtype=torch.long)
        B_idx = torch.arange(num_nodes.shape[0])
        # It adds self edge
        # tgt_adj[:, 0, 1] = 1
        # tgt_adj[:, 1, 0] = 1
        tgt_adj[B_idx, num_nodes[B_idx] - 1, 0] = 1

        B_idx = torch.arange(num_nodes.shape[0])

        # TODO: Ensure not off by one
        if torch.any(tgt_adj != adj):
            self.fail(f"{tgt_adj} != {adj}")

    def test_one_dist(self):
        # Start num_nodes = 1
        self.nodes[:, 0, self.slice] = 1
        _, (nodes, adj, weights, num_nodes) = self.s(
            self.obs, (self.nodes, self.adj, self.weights, self.num_nodes)
        )
        tgt_adj = torch.zeros_like(adj, dtype=torch.long)
        # It adds self edge

        # TODO: Ensure not off by one
        if torch.any(tgt_adj != adj):
            self.fail(f"{tgt_adj} != {adj}")


if __name__ == "__main__":
    unittest.main()
