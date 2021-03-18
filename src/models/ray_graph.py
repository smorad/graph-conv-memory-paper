import torch
import torch.autograd.profiler as profiler
import numpy as np
import gym
from torch import nn
import torch.nn.functional as F
from typing import Union, Dict, List, Tuple, Any
import ray
from ray.rllib.models.torch.torch_modelv2 import TorchModelV2
from ray.rllib.models.modelv2 import ModelV2, restore_original_dimensions
from ray.rllib.models.torch.fcnet import FullyConnectedNetwork as TorchFC
from ray.rllib.models.torch.misc import SlimFC
from ray.rllib.models.torch.recurrent_net import RecurrentNetwork
from ray.rllib.utils.typing import ModelConfigDict, TensorType
from ray.rllib.policy.sample_batch import SampleBatch
from ray.rllib.policy.view_requirement import ViewRequirement
from ray.rllib.policy.rnn_sequencing import add_time_dimension
from torchviz import make_dot

import torch_geometric
from torch_geometric.data import Data, Batch


class RayObsGraph(TorchModelV2, nn.Module):
    DEFAULT_CONFIG = {
        # Maximum number of nodes in a graph
        "graph_size": 32,
        # Maximum GCN forward passes per node, results in
        # receptive field of gcn_num_layers * gcn_num_passes
        "gcn_num_passes": 1,
        # Size of latent vector coming out of GNN
        # before being fed to logits/vf layer(s)
        "gcn_output_size": 256,
        # Size of the hidden layers in the GNN
        "gcn_hidden_size": 256,
        # Number of layers in the GCN, must be >= 2
        "gcn_num_layers": 2,
        # If using GAT, number of attention heads per layer
        "gcn_num_attn_heads": 1,
        # Graph convolution layer class
        "gcn_conv_type": torch_geometric.nn.DenseGCNConv,
        # Activation function for GNN layers
        "gcn_act_type": torch.nn.ReLU,
        "use_batch_norm": False,
        # Methodologies for building edges
        # can be [temporal, dense, knn-mse, knn-cos]
        "edge_selectors": [],
        # For debug visualization
        "export_gradients": False,
    }

    def __init__(
        self,
        obs_space: gym.spaces.Space,
        action_space: gym.spaces.Space,
        num_outputs: int,
        model_config: ModelConfigDict,
        name: str,
        **custom_model_kwargs,
    ):
        nn.Module.__init__(self)
        super(RayObsGraph, self).__init__(
            obs_space, action_space, num_outputs, model_config, name
        )
        self.num_outputs = num_outputs
        self.obs_dim = gym.spaces.utils.flatdim(obs_space)
        self.act_dim = gym.spaces.utils.flatdim(action_space)

        self.cfg = dict(self.DEFAULT_CONFIG, **custom_model_kwargs)
        self.build_network(self.cfg)
        print("GNN network is:", self.gnn)

        self.edge_selectors = [e(self) for e in self.cfg["edge_selectors"]]

        assert (
            model_config["max_seq_len"] <= self.cfg["graph_size"]
        ), "max_seq_len cannot be more than graph size"

        self.cur_val = None
        self.fwd_iters = 0
        self.grad_dots: Dict[str, Any] = {}

    def build_network(self, cfg):
        """Builds the GNN and MLPs based on config"""
        layer_sizes = (
            np.ones((cfg["gcn_num_layers"] + 1,), dtype=np.int32)
            * cfg["gcn_hidden_size"]
        )
        num_heads = (
            np.ones((cfg["gcn_num_layers"],), dtype=np.int32)
            * cfg["gcn_num_attn_heads"]
        )
        # These must be exact
        layer_sizes[0] = self.obs_dim
        layer_sizes[-1] = cfg["gcn_output_size"]
        num_heads = np.insert(num_heads, 0, 1)
        num_heads[-1] = 1

        layers = {}
        for layer in range(cfg["gcn_num_layers"]):
            in_size = layer_sizes[layer]
            out_size = layer_sizes[layer + 1]
            in_heads = num_heads[layer]
            out_heads = num_heads[layer + 1]

            if cfg["gcn_conv_type"] == torch_geometric.nn.GATConv:
                layers[f"graph_{layer}"] = cfg["gcn_conv_type"](
                    int(in_heads * in_size), out_size, out_heads
                )
            else:
                layers[f"graph_{layer}"] = cfg["gcn_conv_type"](in_size, out_size)

            if cfg["use_batch_norm"] and layer < cfg["gcn_num_layers"] - 1:
                layers[f"batchnorm_{layer}"] = torch.nn.BatchNorm1d(out_size)

            layers[f"act_{layer}"] = cfg["gcn_act_type"]()

        self.gnn = nn.ModuleDict(layers)

        self.logit_branch = SlimFC(
            in_size=cfg["gcn_output_size"],
            out_size=self.num_outputs,
            activation_fn=None,
            initializer=torch.nn.init.xavier_uniform_,
        )

        self.value_branch = SlimFC(
            in_size=cfg["gcn_output_size"],
            out_size=1,
            activation_fn=None,
            initializer=torch.nn.init.xavier_uniform_,
        )

    def get_initial_state(self, sparse=False):
        # TODO: Try using torch.sparse_coo layout
        # it's likely the conversion to numpy in rllib
        # breaks this
        if sparse:
            edges = torch.sparse_coo_tensor(
                (self.cfg["graph_size"], self.cfg["graph_size"]), dtype=torch.long
            )
            nodes = torch.sparse_coo_tensor((self.cfg["graph_size"], self.obs_dim))
        else:
            edges = torch.zeros(
                (self.cfg["graph_size"], self.cfg["graph_size"]), dtype=torch.long
            )
            nodes = torch.zeros((self.cfg["graph_size"], self.obs_dim))

        num_nodes = torch.tensor([0], dtype=torch.long)
        state = [num_nodes, nodes, edges]

        edge_selector_states = [
            e.get_initial_state(nodes, edges)
            for e in self.edge_selectors
            if hasattr(e, "get_initial_state")
        ]
        # Edge selector states cannot be a list, must be TensorType
        if len(edge_selector_states) != 0:
            state += [*edge_selector_states]

        return state

    def value_function(self):
        assert self.cur_val is not None, "must call forward() first"
        return self.cur_val

    def densify_graph(self, adj_mats, num_nodes):
        """Connect all nodes to all other nodes. In other words,
        the adjacency matrix is all 1's for a specific row and column
        if the node for that column exists. Also creates a self edge."""
        batch = adj_mats.shape[0]
        for b in range(batch):
            adj_mats[b, : num_nodes[b] + 1, : num_nodes[b] + 1] = 1

    def add_time_positional_encoding(self, nodes, num_nodes):
        """Add a 1D cosine/sine positional embedding to the current node"""
        batch = nodes.shape[0]
        i = torch.range(0, nodes.shape[-1] - 1)
        for b in range(batch):
            pe = num_nodes[b] / (10000 ** (2 * i / i.shape[0]))
            if num_nodes[b] // 2 == 0:
                pe = torch.sin(pe)
            else:
                pe = torch.cos(pe)

            nodes[b] = nodes[b] + pe

    def print_adj_heatmap(self, adj_mats):
        """Prints the mean values of each cell in the adjacency matrix
        for the batch. Call once at the end of the batch train step."""
        mean = torch.mean(adj_mats.float(), dim=0)
        torch.set_printoptions(profile="full")
        print(mean)
        torch.set_printoptions(profile="default")

    def make_pose_relative(self, nodes, num_nodes):
        """The GPS observation is in global coordinates. To ensure locality,
        make all node poses relative to the coordinate frame
        of the current observation"""
        pass

    def get_views(self, nodes, adj_mats, num_nodes):
        """Returns reduced views of the nodes and adj_mats matrices
        so we don't see any of the empty space. It is much faster to use
        a single matrix for the entire batch than multiple small matrices.
        Views allow fast and readable operations on this big tensor."""
        node_views = []
        adj_views = []

        batch = nodes.shape[0]
        for b in range(batch):
            node_views.append(
                nodes[b].narrow(dim=0, start=0, length=num_nodes[b].item() + 1)
            )
            adj_views.append(
                adj_mats[b]
                .narrow(dim=0, start=0, length=num_nodes[b].item() + 1)
                .narrow(dim=1, start=0, length=num_nodes[b].item() + 1)
            )
        return node_views, adj_views

    def add_grad_dot(self, tensor, name):
        if self.cfg["export_gradients"] and self.training:
            self.grad_dots[f"{name}_iter_{self.fwd_iters}"] = make_dot(
                tensor.clone(), params=dict(list(self.named_parameters()))
            )

    def export_dots(self):
        if self.cfg["export_gradients"] and self.training:
            for k, v in self.grad_dots.items():
                v.render(f"/tmp/{k}")
            self.grad_dots.clear()

    def gnn_forward(self, nodes, adj):
        """Given a PyG batch, push it through
        the GNN and return the output at the
        just-added node"""
        out = nodes
        for fwd_pass in range(self.cfg["gcn_num_passes"]):
            for name, layer in self.gnn.items():
                if "act" in name:
                    out = layer(out)
                elif "graph" in name:
                    # Edge weights are required to allow gradient backprop
                    # thru dense->sparse conversion
                    out = layer(out, adj)
                elif "batchnorm" in name:
                    out = layer(out)
                else:
                    raise NotImplementedError(
                        'Graph config only recognizes "graph" and "act" layers'
                    )
        self.add_grad_dot(out, "after_gnn_conv")
        assert self.cfg["gcn_output_size"] == out.shape[-1]

        return out

    def to_sparse(self):
        pass

    def forward(
        self,
        input_dict: Dict[str, TensorType],
        state: List[TensorType],
        seq_lens: TensorType,
    ) -> Tuple[TensorType, List[TensorType]]:

        flat = input_dict["obs_flat"]
        # Batch and Time
        # Forward expects outputs as [B, T, logits]
        B = len(seq_lens)
        T = flat.shape[0] // B

        logits = torch.zeros(B, T, self.num_outputs, device=flat.device)
        values = torch.zeros(B, T, 1, device=flat.device)
        # Deconstruct batch into batch and time dimensions: [B, T, feats]
        flat = torch.reshape(flat, [-1, T] + list(flat.shape[1:]))
        num_nodes, nodes, adj_mats, *edge_selector_states = state

        num_nodes = num_nodes.long()
        adj_mats = adj_mats.long()

        # We only have one training pass per batch
        # this is because instead of propagating state,
        # we are simply passed `obs` with a time dimension
        for t in range(T):
            # Views will be [(num_nodes, features)...], [(num_nodes, num_nodes)...]
            # each of length batch
            node_views, adj_views = self.get_views(nodes, adj_mats, num_nodes)
            # Flat will be shape [B, feat] for inference and
            # shape [B, T, feat] for training
            # For training, we want loss across the entire sequence
            if len(flat.shape) > 2:
                flat = flat[:, t, :].squeeze(1)

            # We have limited space reserved, rewrite at the zeroth entry if we run out
            # of space
            graph_overflows = num_nodes == self.cfg["graph_size"]
            if torch.any(graph_overflows):
                print(
                    "warning: ran out of graph space, overwriting old nodes (seq_len, graph_size):",
                    T,
                    self.cfg["graph_size"],
                    "(You can ignore this for the first backwards pass)",
                )
                num_nodes[graph_overflows] = 0

            # Apply edge selectors `forward`
            edge_state_idx = 0
            for e in self.edge_selectors:
                if hasattr(e, "get_initial_state"):
                    e(
                        nodes,
                        adj_mats,
                        num_nodes,
                        edge_selector_states[edge_state_idx],
                        B,
                    )
                    edge_state_idx += 1
                else:
                    e(nodes, adj_mats, num_nodes, B)

            # Do forwards in batch mode to be more efficient
            gnn_out = self.gnn_forward(nodes, adj_mats)
            self.add_grad_dot(gnn_out, "after_gnn_forward")
            # Extract output at the node we are interested in
            # Rather than use view, use num_nodes so if we run out of graph space
            # we look at the newly placed zeroth node instead of the old nth node
            # target_idxs = torch.stack((torch.arange(B, device=flat.device), num_nodes.squeeze()))
            # node_feats = torch.cat([gnn_out[b, num_nodes[b]] for b in range(B)])
            node_feats = gnn_out[
                torch.arange(B, device=flat.device), num_nodes.squeeze()
            ]
            self.add_grad_dot(node_feats, "after_feat_extraction")

            # Update graph with new node
            num_nodes = num_nodes + 1

            # Outputs
            # vtrace drops the last obs [-1] for vtrace
            # otherwise it should be in order [t0, t1, ... tn]
            logits[:, -1] = self.logit_branch(node_feats)
            values[:, -1] = self.value_branch(node_feats)

        logits = logits.reshape((B * T, self.num_outputs))
        self.add_grad_dot(logits, "final_logits")
        values = values.reshape((B * T, 1))

        self.cur_val = values.squeeze(1)
        self.fwd_iters += 1

        # For metric logging purposes
        self.density = adj_mats.detach().float().mean().item()

        state = [num_nodes, nodes, adj_mats]
        if len(edge_selector_states) != 0:
            state += [*edge_selector_states]

        self.export_dots()
        return logits, state
