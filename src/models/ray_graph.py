import torch
import torch.autograd.profiler as profiler
import numpy as np
import gym
from torch import nn
import torch.nn.functional as F
from typing import Union, Dict, List, Tuple
import ray
from ray.rllib.models.torch.torch_modelv2 import TorchModelV2
from ray.rllib.models.modelv2 import ModelV2, restore_original_dimensions
from ray.rllib.models.torch.fcnet import FullyConnectedNetwork as TorchFC
from ray.rllib.models.torch.misc import SlimFC
from ray.rllib.models.torch.recurrent_net import RecurrentNetwork
from ray.rllib.utils.typing import ModelConfigDict, TensorType
from ray.rllib.policy.sample_batch import SampleBatch
from ray.rllib.policy.view_requirement import ViewRequirement

import torch_geometric
from torch_geometric.data import Data, Batch

## What I've found
## if we disable trajectory view:
##      we must implement get_initial_state
##      the length of the state must remain constant (matching get_initial_state)
##      Changing the shape of the state tensors RESETS them to get_initial_state
##      State is retained if the shape remains constant
##
##      Therefore, we must preallocate the full adjacency and node matrix in get_initial_state
##      if we would like build a graph
##
## With trajectory view enabled (default):
##      State propagates if the shape remains constant
##      View requirements shift DO NOT follow python syntax
##      shift="-500:0" will return a shape 500 zero-padded vector
##      with shape [B, i, space] where i is the index from 0-500
##      note that this counts backwards (i.e. dict[:,-1,:] will be populated first)
##
##      With empty list in get_initial_state, state will not propagate

## The solution is to use trajectory view with shift == max_timesteps
## nodes are stored as observations in the trajectory view
## edges must be propagated using get_initial_state and state, likely via adjacency matrix
## because it must have a fixed size


class RayObsGraph(TorchModelV2, nn.Module):
    DEFAULT_CONFIG = {
        # Maximum number of nodes in a graph
        "graph_size": 32,
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
        "gcn_conv_type": torch_geometric.nn.GCNConv,
        # Activation function for GNN layers
        "gcn_act_type": torch.nn.ReLU,
        # Methodologies for building edges
        # can be [temporal, dense, knn-mse, knn-cos]
        "edge_selectors": ["temporal"],
    }

    EDGE_SELECTORS = ["temporal", "dense", "pose", "knn-mse", "knn-cos"]

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

        assert all(
            [e in self.EDGE_SELECTORS for e in self.cfg["edge_selectors"]]
        ), "Invalid edge selectors"
        assert (
            model_config["max_seq_len"] <= self.cfg["graph_size"]
        ), "max_seq_len cannot be more than graph size"

        self.cur_val = None

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

    def get_initial_state(self):
        edges = torch.zeros(
            (self.cfg["graph_size"], self.cfg["graph_size"]), dtype=torch.long
        )
        nodes = torch.zeros((self.cfg["graph_size"], self.obs_dim))
        num_nodes = torch.tensor([0], dtype=torch.long)
        return [num_nodes, nodes, edges]

    def value_function(self):
        assert self.cur_val is not None, "must call forward() first"
        return self.cur_val

    def add_backedge(self, adj_mats, num_nodes):
        """Add temporal bidirectional back edge, but only if we have >1 nodes
        E.g., node_{t} <-> node_{t-1}"""
        batch = adj_mats.shape[0]

        batch_idxs = torch.arange(batch)
        # batch_idxs = num_nodes.squeeze()
        curr_node_idxs = num_nodes[batch_idxs].squeeze()
        # Zeroth node cant have backedge
        mask = curr_node_idxs > 1
        batch_idxs = batch_idxs.masked_select(mask)
        curr_node_idxs = curr_node_idxs.masked_select(mask)

        adj_mats[batch_idxs, curr_node_idxs, curr_node_idxs - 1] = 1
        adj_mats[batch_idxs, curr_node_idxs - 1, curr_node_idxs] = 1

    def densify_graph(self, adj_mats, num_nodes):
        """Connect all nodes to all other nodes. In other words,
        the adjacency matrix is all 1's for a specific row and column
        if the node for that column exists. Also creates a self edge."""
        batch = adj_mats.shape[0]
        for b in range(batch):
            adj_mats[b, : num_nodes[b] + 1, : num_nodes[b] + 1] = 1

    def add_self_edge(self, adj_mats, num_nodes):
        """Add a self edge to the latest node in the graph
        (graph[num_node] <-> graph[num_node])"""
        batch = adj_mats.shape[0]
        batch_idxs = torch.arange(batch)
        curr_node_idxs = num_nodes[batch_idxs].squeeze()
        adj_mats[batch_idxs, curr_node_idxs, curr_node_idxs] = 1

    def add_knn_edges(self, nodes, adj_mats, num_nodes, dist_measure):
        raise NotImplementedError()

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

    def add_pose_edges(self, nodes, adj_mats, num_nodes):
        pass

    def make_pose_relative(self, nodes, num_nodes):
        """The GPS observation is in global coordinates. To ensure locality,
        make all node poses relative to the coordinate frame
        of the current observation"""
        """
        sensor_keys = list(obs.keys())
        gps_idx = sensor_keys.index('gps')
        compass_idx = sensor_keys.index('compass')
        # Obs will be of size [Batch, feat]
        sensor_sizes = [v.shape[1] for v in obs.values()]
        gps_offset = sum([sensor_sizes[i] for i in range(gps_idx)])
        gps_dim = sensor_sizes[gps_idx]

        compass_offset = sum([sensor_sizes[i] for i in range(compass_idx)])
        compass_dim = sensor_sizes[compass_idx]

        # Shape [Batch, feat]
        # Do NOT use -= as it differs from x = x-y
        # it is in-place and breaks torch autograd

        gps[:,gps_offset : gps_offset + gps_dim + 1]
        compass[:, compass_offset : compass_offset + compass_dim + 1]
        """
        pass

    def forward(
        self,
        input_dict: Dict[str, TensorType],
        state: List[TensorType],
        seq_lens: TensorType,
    ) -> Tuple[TensorType, List[TensorType]]:

        flat = input_dict["obs_flat"]
        # From attention net code
        # likely where the bug in gtrxl is for our case too
        # Batch and Time
        # Forward expects outputs as [B, T, logits]
        B = len(seq_lens)
        T = flat.shape[0] // B

        logits = torch.zeros(B, T, self.num_outputs, device=flat.device)
        values = torch.zeros(B, T, 1, device=flat.device)
        # Deconstruct batch into batch and time dimensions: [B, T, feats]
        flat = torch.reshape(flat, [-1, T] + list(flat.shape[1:]))
        # We only care about latest obs, but we get seq_lens[i] worth of obs
        # TODO: IMPORTANT - ensure we want idx 0 and not -1
        # It looks like input is ordered [t-n, t-n+1, ... t]
        # but output is ordered [t, t-1, ... t-n]
        # make sure this is actually correct
        num_nodes, nodes, adj_mats = state
        num_nodes = num_nodes.long()
        adj_mats = adj_mats.long()

        # We only have one training pass per batch
        # this is because instead of propagating state,
        # we are simply passed `obs` with a time dimension
        for t in range(T):
            # Flat will be shape [B, feat] for inference and
            # shape [B, T, feat] for training
            # For training, we want loss across the entire sequence
            if len(flat.shape) > 2:
                flat = flat[:, t, :].squeeze(1)

            """
            # We have limited space reserved, rewrite at the zeroth entry if we run out
            # of space
            graph_overflows = num_nodes == self.graph_size - 1
            if torch.any(graph_overflows):
                print("error: ran out of graph space, starting from zero")
                num_nodes[graph_overflows] = 0
            """

            # Add new nodes to graph at the next free index (num_nodes)
            batch_idxs = torch.arange(num_nodes.shape[0])
            curr_node_idxs = num_nodes.squeeze()
            nodes[batch_idxs, curr_node_idxs] = flat[batch_idxs]

            # Add self edge
            # TODO: Is this working as expected?
            # Probably not
            # adj_mats[:, num_nodes, num_nodes] = 1
            self.add_self_edge(adj_mats, num_nodes)

            # Add other edges based on config
            if "temporal" in self.cfg["edge_selectors"]:
                # Add edge to obs at t-1
                self.add_backedge(adj_mats, num_nodes)
            if "pose" in self.cfg["edge_selectors"]:
                pass
            if "dense" in self.cfg["edge_selectors"]:
                # Add every possible edge to graph
                self.densify_graph(adj_mats, num_nodes)
            if "knn-mse" in self.cfg["edge_selectors"]:
                # Neighborhoods based on mse distance
                self.add_knn_edges(adj_mats, nodes, num_nodes, F.mse_loss)
            if "knn-cos" in self.cfg["edge_selectors"]:
                # Neigborhoods based on cosine similarity
                self.add_knn_edges(adj_mats, nodes, num_nodes, F.cosine_similarity)

            # GCN uses sparse edgelist
            # For batch mode, torch_geometric expects a specific format
            # see https://pytorch-geometric.readthedocs.io/en/latest/notes/introduction.html#mini-batches
            edge_lists = [
                torch_geometric.utils.dense_to_sparse(adj_mat)[0]
                for adj_mat in adj_mats
            ]
            in_batch = Batch.from_data_list(
                [
                    Data(x=nodes[i], edge_index=edge_lists[i])
                    for i in range(len(edge_lists))
                ]
            )
            # Push graph through GNN
            out = in_batch.x
            for name, layer in self.gnn.items():
                if "act" in name:
                    out = layer(out)
                elif "graph" in name:
                    out = layer(out, in_batch.edge_index)
                else:
                    raise NotImplementedError(
                        'Graph config only recognizes "graph" and "act" layers'
                    )

            # torch_geometric will have collapsed dims[0,1] into dim[0]
            # reconstruct but use gcn output feature size
            # After reshape, out is [Batch, node, gcn_outsize (aka feat)]
            out = torch.reshape(out, (*nodes.shape[:-1], self.cfg["gcn_output_size"]))
            # We only care about observation at our newly added node
            # Index each batch by num_nodes (the node we just inserted) to get
            # the target node for each batch
            target_node_out = torch.stack(
                [out[i, tgt_n, :].squeeze(0) for i, tgt_n in enumerate(num_nodes)]
            )

            # Update graph with new node
            num_nodes = num_nodes + 1

            # Outputs
            # vtrace drops the last obs [-1] for vtrace
            # otherwise it should be in order [t0, t1, ... tn]
            logits[:, t] = self.logit_branch(target_node_out)
            values[:, t] = self.value_branch(target_node_out)

        logits = logits.reshape((B * T, self.num_outputs))
        values = values.reshape((B * T, 1))

        self.cur_val = values.squeeze(1)

        return logits, state

    def forward_debug(
        self,
        input_dict: Dict[str, TensorType],
        state: List[TensorType],
        seq_lens: TensorType,
    ) -> Tuple[TensorType, List[TensorType]]:

        obs = input_dict["obs_flat"]
        batch_size = obs.shape[0]

        logits = self.simple(obs)

        """
        # Test if shape changes ok
        print('input', state[0].shape, state[0].max())
        state[0] = torch.cat( (state[0], torch.ones((batch_size, 1), device=state[0].device)))
        print('output', state[0].shape, state[0].max())
        """

        """
        # Test if state propagates
        state[0] += 5
        print(state[0].shape, state[0].max())
        """

        """
        # Test trajectory view
        print(input_dict['prev_obs'].shape)
        """

        """
        # Test trajectory view padding
        print(torch.nonzero(input_dict['prev_obs']).shape)
        """
        # Outputs:
        # logits: [batch_size, num_outputs]
        # cur_val: [batch_size,]
        # logits = torch.zeros((batch_size, self.num_outputs)).to("cuda")
        self.cur_val = torch.zeros((batch_size, 1)).squeeze(1).to("cuda")
        # self.value_branch(out).squeeze(1).to("cuda")

        return logits, state
