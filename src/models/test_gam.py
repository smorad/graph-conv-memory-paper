import unittest
import torch
from gam import GNN, DenseGAM
import torch_geometric


class TestDenseGAM(unittest.TestCase):
    def setUp(self):
        feats = 11
        batches = 5
        N = 10
        self.g = GNN(feats, feats)
        self.s = DenseGAM(self.g)

        # Now do it in a loop to make sure grads propagate
        self.optimizer = torch.optim.Adam(self.s.parameters(), lr=0.005)

        self.nodes = torch.arange(batches * N * feats, dtype=torch.float).reshape(
            batches, N, feats
        )
        self.obs = torch.zeros(batches, feats)
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
        for i in range(3):
            nodes = torch.arange(batches * N * feats, dtype=torch.float).reshape(
                batches, N, feats
            )
            obs = torch.zeros(batches, feats)
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

        self.assertTrue(losses[2] < losses[0])


class TestSparseGAM(unittest.TestCase):
    def setUp(self):
        feats = 11
        batches = 5
        N = 10
        self.g = GNN(feats, feats, sparse=True, conv_type=torch_geometric.nn.GCNConv)
        self.s = DenseGAM(self.g)

        # Now do it in a loop to make sure grads propagate
        self.optimizer = torch.optim.Adam(self.s.parameters(), lr=0.005)

        self.nodes = torch.arange(batches * N * feats, dtype=torch.float).reshape(
            batches, N, feats
        )
        self.obs = torch.zeros(batches, feats)
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
        return
        out, (nodes, adj, weights, num_nodes) = self.s(
            self.obs, (self.nodes, self.adj, self.weights, self.num_nodes)
        )
        if torch.all(out == self.obs):
            self.fail(f"{out} == {self.obs}")

    def test_sparse_learn(self):
        return
        feats = 11
        batches = 5
        T = 4
        N = 10
        losses = []
        for i in range(3):
            nodes = torch.arange(batches * N * feats, dtype=torch.float).reshape(
                batches, N, feats
            )
            obs = torch.zeros(batches, feats)
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

        self.assertTrue(losses[2] < losses[0])


if __name__ == "__main__":
    unittest.main()
