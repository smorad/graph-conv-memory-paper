import torch
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
from torch_geometric.data import Data

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
    def __init__(
        self,
        obs_space: gym.spaces.Space,
        action_space: gym.spaces.Space,
        num_outputs: int,
        model_config: ModelConfigDict,
        name: str,
    ):
        nn.Module.__init__(self)
        super(RayObsGraph, self).__init__(
            obs_space, action_space, num_outputs, model_config, name
        )
        self.num_outputs = num_outputs
        self.graph_size = 128
        self.t_dist = 4
        self.gcn_outsize = 1024
        self.gcn_h_size = 256
        self.edge_predictor_h_size = 256
        self.obs_dim = gym.spaces.utils.flatdim(obs_space)
        self.act_dim = gym.spaces.utils.flatdim(action_space)

        """
        self.view_requirements["prev_obs"] = ViewRequirement(
            data_col='obs',
            shift="-{self.graph_size}:0",
            space=obs_space,
            used_for_training=True,
            used_for_compute_actions=True)
        """
        self.simple = SlimFC(in_size=self.obs_dim, out_size=self.num_outputs)

        self.gcn0 = torch_geometric.nn.GraphConv(self.obs_dim, self.gcn_h_size)
        # Kind of like the insect robots
        # Action probs could be sum of gcn and image policy
        # When unclear follow map, else if in front follow obs policy
        self.gcn1 = torch_geometric.nn.GraphConv(self.gcn_h_size, self.gcn_outsize)

        self.logit_branch = SlimFC(
            in_size=self.gcn_outsize,
            out_size=self.num_outputs,
            activation_fn=None,
            initializer=torch.nn.init.xavier_uniform_,
        )

        self.value_branch = SlimFC(
            in_size=self.gcn_outsize,
            out_size=1,
            activation_fn=None,
            initializer=torch.nn.init.xavier_uniform_,
        )

        self.cur_val = None

        self.edge_predictor = nn.Sequential(
            nn.Linear(2 * self.gcn_outsize, self.edge_predictor_h_size),
            nn.ReLU(),
            nn.Linear(self.edge_predictor_h_size, 1),
            nn.Tanh(),
        )

    def get_initial_state(self):
        edges = torch.zeros(
            (self.graph_size, self.graph_size), dtype=torch.long, device="cuda"
        )
        nodes = torch.zeros((self.graph_size, self.obs_dim), device="cuda")
        num_nodes = torch.tensor([0], dtype=torch.long, device="cuda")
        return [num_nodes, nodes, edges]

    def value_function(self):
        assert self.cur_val is not None, "must call forward() first"
        return self.cur_val

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
        # Deconstruct batch into batch and time dimensions
        flat = torch.reshape(flat, [-1, T] + list(flat.shape[1:]))

        b_num_nodes, b_nodes, b_adj_mats = state
        # rllib casts state tensors to float32 for some reason...
        b_adj_mats = b_adj_mats.long()
        b_num_nodes = b_num_nodes.long()

        torch.autograd.set_detect_anomaly(True)

        batch_logits = torch.zeros(B, T, self.num_outputs, device=flat.device)
        batch_values = torch.zeros(B, T, 1, device=flat.device)

        """
        print('batch_size', B)
        print('time_size', T)
        print('num_nodes', b_num_nodes.shape)
        print('seq_lens', seq_lens)
        """

        for b in range(B):
            num_nodes = b_num_nodes[b]
            nodes = b_nodes[b]
            adj_mat = b_adj_mats[b]
            # Nodes come in as [t, t-1, ... t-n]

            # Add self edge and t-1 edge
            adj_mat[num_nodes, num_nodes] = 1
            if num_nodes > 0:
                adj_mat[num_nodes, num_nodes - 1] = 1
                adj_mat[num_nodes - 1, num_nodes] = 1

            # Add latest node to graph
            # Zeroth time seems to be the most recent
            # TODO WARNING: This fixes backprop issue but we are bypassing
            # computation graph for this assignment
            # see https://stackoverflow.com/questions/53819383/how-to-assign-a-new-value-to-a-pytorch-variable-without-breaking-backpropagation
            nodes[num_nodes].data = flat[b][0]

            edge_list, _ = torch_geometric.utils.dense_to_sparse(adj_mat)
            # TODO: GCN causing in-place issue
            # nodes.data fixes it but then we lost gradient info...
            out = self.gcn0(nodes, edge_list).relu()
            out = self.gcn1(out, edge_list).relu()  # 128x1024
            # We only care about our current pos/node
            # TODO: Ensure we actually care about only a specific node
            # Instead of convolving to size 1
            node_out = out[num_nodes]
            # node_out = torch.zeros((1024,),device=flat.device)

            batch_logits[b][0] = self.logit_branch(node_out)
            batch_values[b][0] = self.value_branch(node_out)
            num_nodes = num_nodes + 1

        # TODO: ENSURE WE CAN SEND ZEROS FOR OTHER TIMESTEPS
        batch_logits = batch_logits.reshape((B * T, self.num_outputs))
        batch_values = batch_values.reshape((B * T, 1))

        self.cur_val = batch_values.squeeze(1)

        # Outputs:
        # logits: [batch_size, num_outputs]
        # cur_val: [batch_size,]
        # logits = torch.zeros((batch_size, self.num_outputs)).to("cuda")
        # self.cur_val = torch.zeros((batch_size, 1)).squeeze(1).to('cuda')
        # self.value_branch(out).squeeze(1).to("cuda")

        return batch_logits, state

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
