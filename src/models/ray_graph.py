import torch
import torch.autograd.profiler as profiler
import numpy as np
import gym
from torch import nn
import torch.nn.functional as F
from typing import Union, Dict, List, Tuple, Any
import ray
from ray.rllib.models.torch.torch_modelv2 import TorchModelV2
from ray.rllib.models.modelv2 import (
    ModelV2,
    restore_original_dimensions,
    flatten,
    _unpack_obs,
)
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
from models.gam import GNN, DenseGAM


class RayObsGraph(TorchModelV2, nn.Module):
    DEFAULT_CONFIG = {
        # Maximum number of nodes in a graph
        "graph_size": 32,
        # Maximum GCN forward passes per node, results in
        # receptive field of gcn_hidden_layers * gcn_num_passes
        "gcn_num_passes": 1,
        # Size of latent vector coming out of GNN
        # before being fed to logits/vf layer(s)
        "gcn_output_size": 256,
        # Size of the hidden layers in the GNN
        "gcn_hidden_size": 256,
        # Number of layers in the GCN, must be >= 2
        "gcn_hidden_layers": 2,
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
        TorchModelV2.__init__(
            self, obs_space, action_space, num_outputs, model_config, name
        )
        nn.Module.__init__(self)
        # super(RayObsGraph, self).__init__(
        #    obs_space, action_space, num_outputs, model_config, name
        # )
        self.num_outputs = num_outputs
        self.obs_dim = gym.spaces.utils.flatdim(obs_space)
        self.act_dim = gym.spaces.utils.flatdim(action_space)

        self.cfg = dict(self.DEFAULT_CONFIG, **custom_model_kwargs)
        self.edge_selectors = [e(self) for e in self.cfg["edge_selectors"]]
        self.build_network(self.cfg)
        print("Full network is:", self)

        assert (
            model_config["max_seq_len"] <= self.cfg["graph_size"]
        ), "max_seq_len cannot be more than graph size"

        self.cur_val = None
        self.fwd_iters = 0
        self.grad_dots: Dict[str, Any] = {}

    def build_network(self, cfg):
        """Builds the GNN and MLPs based on config"""
        gnn = GNN(
            input_size=self.obs_dim,
            output_size=cfg["gcn_output_size"],
            graph_size=cfg["graph_size"],
            hidden_size=cfg["gcn_hidden_size"],
            num_layers=cfg["gcn_hidden_layers"],
            conv_type=cfg["gcn_conv_type"],
            activation=cfg["gcn_act_type"],
        )
        self.gam = DenseGAM(gnn, edge_selectors=self.edge_selectors)

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

    def get_initial_state(self):
        # TODO: Try using torch.sparse_coo layout
        # it's likely the conversion to numpy in rllib
        # breaks this
        edges = torch.zeros(
            (self.cfg["graph_size"], self.cfg["graph_size"]), dtype=torch.long
        )
        nodes = torch.zeros((self.cfg["graph_size"], self.obs_dim))
        weights = torch.zeros(
            (self.cfg["graph_size"], self.cfg["graph_size"]), dtype=torch.long
        )

        num_nodes = torch.zeros([1], dtype=torch.long)
        state = [nodes, edges, weights, num_nodes]

        return state

    def value_function(self):
        assert self.cur_val is not None, "must call forward() first"
        return self.cur_val

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

    def get_flat_views(self, obs, flat):
        """Given the obs dict and flat obs, return a dict of corresponding views
        to the same data in the flat vector."""
        ends = {}
        starts = {}

        start = 0
        for key, data in obs.items():
            starts[key] = start
            start += data.shape[-1]

        end = 0
        for key, data in obs.items():
            end += data.shape[-1]
            ends[key] = end

        flat_views = {key: flat[:, starts[key] : ends[key]] for key in obs}
        return flat_views

    def get_flat_idxs(self, obs, flat):
        """Given the obs dict and flat obs, return a dict of [start, end) slices
        that index the obs keys into the flat vector"""
        ends = {}
        starts = {}

        start = 0
        for key, data in obs.items():
            starts[key] = start
            start += data.shape[-1]

        end = 0
        for key, data in obs.items():
            end += data.shape[-1]
            ends[key] = end

        return starts, ends

    def make_pose_relative(self, obs, nodes, num_nodes):
        """The GPS observation is in global coordinates. To ensure locality,
        make all node poses relative to the coordinate frame
        of the current observation"""
        B = nodes.shape[0]
        start, stop = self.get_flat_idxs(obs, nodes)
        gps_s = slice(start["gps"], stop["gps"])
        compass_s = slice(start["compass"], stop["compass"])
        origin_nodes = nodes[torch.arange(B), num_nodes.squeeze()]
        nodes[:, :, gps_s] = nodes[:, :, gps_s] - origin_nodes[:, None, gps_s]
        nodes[:, :, compass_s] = (
            nodes[:, :, compass_s] - origin_nodes[:, None, compass_s]
        )

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

    def forward(
        self,
        input_dict: Dict[str, TensorType],
        state: List[TensorType],
        seq_lens: TensorType,
    ) -> Tuple[TensorType, List[TensorType]]:

        torch.autograd.set_detect_anomaly(True)
        flat = input_dict["obs_flat"]
        # Batch and Time
        # Forward expects outputs as [B, T, logits]
        B = len(seq_lens)
        T = flat.shape[0] // B

        logits = torch.zeros(B, T, self.num_outputs, device=flat.device)
        values = torch.zeros(B, T, 1, device=flat.device)
        # Deconstruct batch into batch and time dimensions: [B, T, feats]
        flat = torch.reshape(flat, [-1, T] + list(flat.shape[1:]))
        nodes, adj_mats, weights, num_nodes = state

        num_nodes = num_nodes.long()
        adj_mats = adj_mats.long()

        for t in range(T):
            hidden = (nodes, adj_mats, weights, num_nodes)
            out, hidden = self.gam(flat[:, t, :], hidden)

            # Outputs
            logits[:, -1] = self.logit_branch(out)
            values[:, -1] = self.value_branch(out)

        logits = logits.reshape((B * T, self.num_outputs))
        self.add_grad_dot(logits, "final_logits")
        values = values.reshape((B * T, 1))

        self.cur_val = values.squeeze(1)
        self.fwd_iters += 1

        state = list(hidden)
        self.export_dots()

        return logits, state
