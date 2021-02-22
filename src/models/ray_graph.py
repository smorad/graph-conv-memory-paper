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

import torch_geometric
from torch_geometric.data import Data


class RayObsGraph(RecurrentNetwork, nn.Module):
    garbage_state: bool

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
        self.t_dist = 4
        self.gcn_outsize = 1024
        self.gcn_h_size = 256
        self.edge_predictor_h_size = 256
        self.obs_dim = gym.spaces.utils.flatdim(obs_space)
        self.act_dim = gym.spaces.utils.flatdim(action_space)
        self.gcn0 = torch_geometric.nn.GCNConv(self.obs_dim, self.gcn_h_size)
        # Kind of like the insect robots
        # Action probs could be sum of gcn and image policy
        # When unclear follow map, else if in front follow obs policy
        self.gcn1 = torch_geometric.nn.GCNConv(self.gcn_h_size, self.gcn_outsize)

        """
        # TODO: Copy our model setup
        self.logit_branch = nn.Sequential(
            nn.Linear(self.gcn_outsize, num_outputs)
        )

        self.value_branch = nn.Sequential(
            nn.Linear(self.gcn_outsize, 1),
        )
        """
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

    def value_function(self):
        assert self.cur_val is not None, "must call forward() first"
        return self.cur_val

    def variables(
        self, as_dict: bool = False
    ) -> Union[List[TensorType], Dict[str, TensorType]]:
        p = list(self.parameters())
        if as_dict:
            return {k: p[i] for i, k in enumerate(self.state_dict().keys())}
        return p

    def trainable_variables(
        self, as_dict: bool = False
    ) -> Union[List[TensorType], Dict[str, TensorType]]:
        if as_dict:
            return {
                k: v for k, v in self.variables(as_dict=True).items() if v.requires_grad  # type: ignore
            }
        return [v for v in self.variables() if v.requires_grad]  # type: ignore

    def add_node(self, state: TensorType, flat_obs: TensorType, seq_lens):
        """Adds the current observation as a node in
        the graph, and builds edges to t-1 and itself

        state is [[Batch, [[in_edges], [out_edges]], [Batch, nodes]]
        flat_obs is [Batch, feature]

        Returns updated state: [edges, nodes]

        """
        # Add current obs as node and edge
        # pytorch geometric data format
        # State = [edge_list, node_features]

        batch_size = flat_obs.shape[0]
        self_edge = torch.zeros(
            (batch_size, 2, 1), device=flat_obs.device, dtype=torch.long
        )
        node = flat_obs.to(flat_obs.device)
        # t == 0
        if state == []:
            return self_edge, node.unsqueeze(1)

        # t = 1..n
        # Add self connection and t-1 connection
        # state == [Batch, num_nodes, num_features]
        nodes: TensorType = torch.cat((nodes, node.unsqueeze(1)))
        edges: TensorType = torch.cat((edges, self_edge))
        """
        if edges.shape[1] > 1:
            back_edges = torch.tensor([
                [nodes.shape[1], nodes.shape[1] - 1],
                [nodes.shape[1] - 1, node.shape[1]
            ])
            new_edges = torch.cat((new_edges, back_edges))

        new_node = flat_obs.unsqueeze(0)

        edges = torch.cat((edges, new_edges))
        nodes = torch.cat((nodes, new_node))
        """
        state = [edges, nodes]
        return state

    def predict_edges(self, nodes, idx):
        """Predicts useful edges for node at idx"""
        pred_edges = []
        tgt_node = nodes[idx]
        for n_idx in range(nodes.shape[0]):
            pred = self.predictor(torch.cat(tgt_node, nodes[n_idx]))
            if pred > 0:
                pred.append([idx, n_idx])
                pred.append([n_idx, idx])
        return torch.Tensor(pred_edges)

    def get_initial_state(self):
        # We cannot return empty here
        # so create these placeholders instead
        return []
        nodes = torch.empty(
            (1, self.obs_dim), device="cuda"
        )  # torch.tensor([], device='cuda')
        edges = torch.empty((2, 1), device="cuda", dtype=torch.long)
        self.garbage_state = True
        return [edges, nodes]
        # return []

    def forward(
        self,
        input_dict: Dict[str, TensorType],
        state: List[TensorType],
        seq_lens: TensorType,
    ) -> Tuple[TensorType, List[TensorType]]:
        """Adds time dimension to batch before sending inputs to forward_rnn().

        You should implement forward_rnn() in your subclass."""
        flat_inputs = input_dict["obs_flat"]
        if isinstance(seq_lens, np.ndarray):
            seq_lens = torch.Tensor(seq_lens).int()

        output, new_state = self.forward_rnn(flat_inputs, state, seq_lens)
        output = torch.reshape(output, [-1, self.num_outputs])
        return output, new_state

    def forward_rnn(
        self, inputs: TensorType, state: List[TensorType], seq_lens: TensorType
    ):
        """
        inputs (dict): Observation tensor with shape [B, obs_size].
        state (list): List of state tensors, each with shape [B, size].
        seq_lens (Tensor): 1D tensor holding input sequence lengths.
            Note: len(seq_lens) == B.

        Returns:
            (outputs, new_state): The model output tensor of shape
                [B, T, num_outputs] and the list of new state tensors each with
                shape [B, size].
        """
        obs = inputs
        edges, nodes = self.add_node(state, obs, seq_lens)
        # Load into form for torch geometric
        # x, edge_index = Data(nodes, edges)
        self = self.to("cpu")
        nodes = nodes.to("cpu")
        edges = edges.to("cpu")
        batches = nodes.shape[0]
        # TODO: Use more efficient batch notation
        out = torch.empty((batches, self.gcn_outsize), device="cpu")
        for b in range(batches):
            x = nodes[b]
            # TODO: This is float, figure out what is casting it from int to float
            ei = edges[b].long()
            x = F.relu(self.gcn0(x, ei))
            x = F.dropout(x)
            x = F.relu(self.gcn1(x, ei))
            out[b] = x

        state = [edges, nodes]
        # Rllib expects outputs on gpu
        logits = self.logit_branch(out).to("cuda")
        self.cur_val = self.value_branch(out).squeeze(1).to("cuda")
        print(
            "inputs",
            inputs.shape,
            "obs",
            obs.shape,
            "nodes",
            nodes.shape,
            "logits",
            logits.shape,
        )

        # out = torch.zeros((obs["gps"].shape[0], 5)).to(obs["gps"].device)
        # Return [batch, action_space]

        return logits, state

    def custom_loss(
        self, policy_loss: List[torch.Tensor], loss_inputs
    ) -> List[torch.Tensor]:
        return policy_loss

    def metrics(self):
        return {}
