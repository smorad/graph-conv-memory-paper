import warnings
import numpy as np

warnings.filterwarnings("ignore", category=np.VisibleDeprecationWarning)
import torch
import time
import torch.autograd.profiler as profiler
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
from ray.rllib.models.torch.misc import SlimFC, normc_initializer
from ray.rllib.models.torch.recurrent_net import RecurrentNetwork
from ray.rllib.utils.typing import ModelConfigDict, TensorType
from ray.rllib.policy.sample_batch import SampleBatch
from ray.rllib.policy.view_requirement import ViewRequirement
from ray.rllib.policy.rnn_sequencing import add_time_dimension
from ray.rllib.utils.torch_ops import one_hot

from torchviz import make_dot

import torch_geometric
from torch_geometric.data import Data, Batch
from models.sparse_gcm import SparseGCM
from models.edge_selectors.bernoulli import BernoulliEdge
import pydot
import visdom


class RaySparseObsGraph(TorchModelV2, nn.Module):
    DEFAULT_CONFIG = {
        # Input size to the GNN. Make sure your first gnn layer
        # has this many input channels
        "gnn_input_size": 64,
        # Number of output channels of the GNN. This feeds into the logits
        # and value function layers
        "gnn_output_size": 64,
        # GNN model that takes x, edge_index, weights
        # Note that input will be reshaped by a linear layer
        # to gcn_input_size
        "gnn": torch_geometric.nn.Sequential(
            "x, edge_index, weights",
            [
                (torch_geometric.nn.GraphConv(64, 64), "x, edge_index -> x"),
                torch.nn.Tanh(),
                (torch_geometric.nn.GraphConv(64, 64), "x, edge_index -> x"),
                torch.nn.Tanh(),
            ],
        ),
        # Methodologies for building edges
        # This should be a torch.nn.Module
        # use torch_geometric.nn.Sequential to chain modules
        "edge_selectors": None,
        # Whether the prev action should be placed in the observation nodes
        "use_prev_action": False,
        # Add regularization loss based on weight matrix.
        # This only makes sense if using the BernoulliEdge.
        "regularize": False,
        "regularization_coeff": 1e-5,
        # Do not apply regularization when graph density
        # drops below this value. This prevents the density
        # from going all the way to zero
        "regularization_min": 0.1,
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
        self.num_outputs = num_outputs
        self.obs_dim = gym.spaces.utils.flatdim(obs_space)
        self.act_space = action_space
        self.act_dim = gym.spaces.utils.flatdim(action_space)

        for k in custom_model_kwargs:
            assert k in self.DEFAULT_CONFIG, f"Invalid config key {k}"
        self.cfg = dict(self.DEFAULT_CONFIG, **custom_model_kwargs)
        self.input_dim = self.obs_dim
        if self.cfg["use_prev_action"]:
            self.input_dim += self.act_dim
            self.view_requirements["prev_actions"] = ViewRequirement(
                "actions", space=self.action_space, shift=-1
            )

        self.build_network(self.cfg)
        print("Full network is:", self)

        self.cur_val = None
        self.fwd_iters = 0
        self.grad_dots: Dict[str, Any] = {}
        self.visdom_mets: Dict[str, Dict[str, np.ndarray]] = {}
        # if self.cfg["export_gradients"]:
        self.visdom = visdom.Visdom("http://localhost", port=5050)

    def build_network(self, cfg):
        """Builds the MLPs based on config"""
        fc = torch.nn.Linear(self.input_dim, cfg["gnn_input_size"])
        self.gcm = SparseGCM(
            gnn=cfg["gnn"],
            preprocessor=fc,
            edge_selectors=self.cfg["edge_selectors"],
        )

        self.logit_branch = SlimFC(
            in_size=cfg["gnn_output_size"],
            out_size=self.num_outputs,
            activation_fn=None,
            initializer=normc_initializer(0.01),  # torch.nn.init.xavier_uniform_,
        )

        self.value_branch = SlimFC(
            in_size=cfg["gnn_output_size"],
            out_size=1,
            activation_fn=None,
            initializer=normc_initializer(0.01),  # torch.nn.init.xavier_uniform_,
        )

    def get_initial_state(self):
        # B,2,k
        # edges = torch.tensor([], dtype=torch.long).reshape(2, 0)
        edges = torch.zeros([], dtype=torch.long).reshape(2, 0)
        # B,T,feat
        nodes = torch.tensor([]).reshape(0, self.input_dim)
        # B,1,k
        weights = torch.tensor([], dtype=torch.float).reshape(1, 0)

        state = [nodes, edges, weights]

        return state

    def value_function(self):
        assert self.cur_val is not None, "must call forward() first"
        return self.cur_val

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

    def make_pose_relative(self, obs, nodes, flat):
        """The GPS observation is in global coordinates. To ensure locality,
        make all node poses relative to the coordinate frame
        of the current observation"""
        start, stop = self.get_flat_idxs(obs, nodes)
        gps_s = slice(start["gps"], stop["gps"])
        compass_s = slice(start["compass"], stop["compass"])
        nodes[:, :, gps_s] = nodes[:, :, gps_s] - flat[:, None, gps_s]
        nodes[:, :, compass_s] = nodes[:, :, compass_s] - flat[:, None, compass_s]
        flat = flat - flat  # set curr obs to (0,0)
        return nodes, flat

    def add_grad_dot(self, tensor, name):
        if self.cfg["export_gradients"] and self.training:
            self.grad_dots[f"{name}_iter_{self.fwd_iters}"] = make_dot(
                tensor.clone(), params=dict(list(self.named_parameters()))
            )

    def export_dots(self):
        if self.cfg["export_gradients"] and self.training:
            for k, v in self.grad_dots.items():
                path = f"/tmp/{k}"
                v.render(path, format="svg")
                self.visdom.svg(svgfile=path + ".svg", opts={"caption": k, "title": k})
            print("Exported dots to visdom")

    def get_num_comp_graph_nodes(self):
        """Prints the number of nodes in the torch computational graph. Use
        this to ensure we don't leak gradients from previous passes"""
        if self.cfg["export_gradients"] and self.training:
            for k, v in self.grad_dots.items():
                self.visdom_mets["line"][
                    k.split("_iter")[0] + "_comp_graph_nodes"
                ] = np.array([len(v.body)])

    def adj_heatmap(self, adj):
        if self.training:
            if "heat" not in self.visdom_mets:
                self.visdom_mets["heat"] = {}
            key = f'adj_heatmap-{self.cfg["edge_selectors"]}'
            if key not in self.visdom_mets["heat"]:
                self.visdom_mets["heat"][key] = np.zeros(
                    shape=adj.shape[1:], dtype=np.float
                )
            self.visdom_mets["heat"][key] += adj.sum(dim=0).detach().cpu().numpy()

    def pose_adj_scatter(self, adj, num_nodes):
        if self.training:
            if "scatter" not in self.visdom_mets:
                self.visdom_mets["scatter"] = {}
            key = f'pose_scatter-{self.cfg["edge_selectors"]}'
            if key not in self.visdom_mets["scatter"]:
                self.visdom_mets[key] = np.zeros(shape=adj.shape[1:], dtype=np.float)
            self.visdom_mets[key] = adj.sum(dim=0).detach().cpu().numpy()

    def report_densities(self, adj, weight):
        if self.training:
            if "line" not in self.visdom_mets:
                self.visdom_mets["line"] = {}
            key = f'adj_density-{self.cfg["edge_selectors"]}'
            if key not in self.visdom_mets["line"]:
                self.visdom_mets["line"][key] = np.zeros((1,))
            adj_det = adj.detach()
            self.visdom_mets["line"][key] = (
                adj_det.float().mean().cpu().numpy().reshape(-1)
            )

            key = f'weight_density-{self.cfg["edge_selectors"]}'
            if key not in self.visdom_mets["line"]:
                self.visdom_mets["line"][key] = np.zeros((1,))
            self.visdom_mets["line"][key] = (
                weight.detach().float().mean().cpu().numpy().reshape(-1)
            )

    def forward(
        self,
        input_dict: Dict[str, TensorType],
        state: List[TensorType],
        seq_lens: TensorType,
    ) -> Tuple[TensorType, List[TensorType]]:

        if self.cfg["use_prev_action"]:
            prev_acts = one_hot(input_dict["prev_actions"].float(), self.act_space)
            prev_acts = prev_acts.reshape(-1, self.act_dim)
            flat = torch.cat((input_dict["obs_flat"], prev_acts), dim=-1)
        else:
            flat = input_dict["obs_flat"]
        # Batch and Time
        # Forward expects outputs as [B, T, logits]
        self.device = flat.device
        # print(seq_lens)
        B = len(seq_lens)
        T = flat.shape[0] // B

        # logits = torch.zeros(B, T, self.num_outputs, device=flat.device)
        # values = torch.zeros(B, T, 1, device=flat.device)
        # outs = torch.zeros(B, T, self.cfg["gnn_input_size"], device=flat.device)
        # Deconstruct batch into batch and time dimensions: [B, T, feats]
        flat = torch.reshape(flat, [-1, T] + list(flat.shape[1:]))
        nodes, edges, weights = state
        edges = edges.long()

        # Push thru pre-gcm layers
        out, nodes, edges, weights = self.gcm(flat, nodes, edges, weights)

        # Collapse batch and time for more efficient forward
        out_view = out.view(B * T, self.cfg["gnn_output_size"])
        logits = self.logit_branch(out_view)
        values = self.value_branch(out_view)
        # print(B, T, nodes.shape)

        self.cur_val = values.squeeze(1)
        self.fwd_iters += 1

        # Old num_nodes shape
        state = [nodes, edges, weights]
        return logits, state

    def custom_loss(
        self, policy_loss: List[TensorType], loss_inputs: Dict[str, TensorType]
    ) -> List[TensorType]:

        if not self.cfg["regularize"]:
            return policy_loss

        [bern_edge] = [
            e for e in self.cfg["edge_selectors"] if isinstance(e, BernoulliEdge)
        ]

        # L_0 regularization loss for bernoulli
        edge_density = bern_edge.detach_loss()
        reg_loss = self.cfg["regularization_coeff"] * edge_density
        self.add_grad_dot(reg_loss, "reg_loss")
        # For next time
        # reg_loss = self.cfg["regularization_coeff"] * edge_density
        # policy_loss[0] = policy_loss[0] * (1 + reg_loss)
        policy_loss[0] = policy_loss[0] + reg_loss

        return policy_loss
