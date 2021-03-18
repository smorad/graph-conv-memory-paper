import torch
import gym
from torch import nn
from typing import Union, Dict, List, Tuple, Any
import ray
from ray.rllib.models.torch.torch_modelv2 import TorchModelV2
from ray.rllib.models.torch.misc import SlimFC
from ray.rllib.utils.typing import ModelConfigDict, TensorType

import os
from dnc import DNC


class DNCMemory(TorchModelV2, nn.Module):
    DEFAULT_CONFIG = {
        "dnc_model": DNC,
        "hidden_size": 128,
        "num_layers": 1,
        "num_hidden_layers": 2,
        "read_heads": 4,
        "nr_cells": 32,
        "cell_size": 16,
    }

    MEMORY_KEYS = [
        "memory",
        "link_matrix",
        "precedence",
        "read_weights",
        "write_weights",
        "usage_vector",
    ]

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
        super(DNCMemory, self).__init__(
            obs_space, action_space, num_outputs, model_config, name
        )
        self.num_outputs = num_outputs
        self.obs_dim = gym.spaces.utils.flatdim(obs_space)
        self.act_dim = gym.spaces.utils.flatdim(action_space)

        self.cfg = dict(self.DEFAULT_CONFIG, **custom_model_kwargs)
        self.cur_val = None
        self.fwd_iters = 0

        self.logit_branch = SlimFC(
            in_size=self.cfg["cell_size"],
            out_size=self.num_outputs,
            activation_fn=None,
            initializer=torch.nn.init.xavier_uniform_,
        )

        self.value_branch = SlimFC(
            in_size=self.cfg["cell_size"],
            out_size=1,
            activation_fn=None,
            initializer=torch.nn.init.xavier_uniform_,
        )

        self.dnc: Union[None, DNC] = None

    def get_initial_state(self):
        ctrl_hidden = [
            torch.zeros(2, self.cfg["hidden_size"]),
            torch.zeros(2, self.cfg["hidden_size"]),
        ]
        m = self.cfg["nr_cells"]
        r = self.cfg["read_heads"]
        w = self.cfg["cell_size"]
        memory = [
            torch.zeros(m, w),  # memory
            torch.zeros(1, m, m),  # link_matrix
            torch.zeros(1, m),  # precedence
            torch.zeros(r, m),  # read_weights
            torch.zeros(1, m),  # write_weights
            torch.zeros(m),  # usage_vector
        ]

        read_vecs = torch.zeros(w * r)

        state = [*ctrl_hidden, read_vecs, *memory]
        assert len(state) == 9
        return state

    def value_function(self):
        assert self.cur_val is not None, "must call forward() first"
        return self.cur_val

    def unpack_state(self, state: List[TensorType], B: int, T: int):
        """Given a list of tensors, reformat for self.dnc input"""
        assert len(state) == 9, "Failed to verify unpacked state"
        ctrl_hidden: List[Tuple[TensorType, TensorType]] = [
            (
                state[0].permute(1, 0, 2).contiguous(),
                state[1].permute(1, 0, 2).contiguous(),
            )
        ]
        read_vecs: TensorType = state[2]
        memory: List[TensorType] = state[3:]
        memory_dict: Dict[str, TensorType] = dict(zip(self.MEMORY_KEYS, memory))

        return ctrl_hidden, memory_dict, read_vecs

    def pack_state(
        self,
        ctrl_hidden: List[Tuple[TensorType, TensorType]],
        memory_dict: Dict[str, TensorType],
        read_vecs: TensorType,
    ):
        """Given the dnc output, pack it into a list of tensors
        for rllib state. Order is ctrl_hidden, read_vecs, memory_dict"""
        state = []
        ctrl_hidden = [
            ctrl_hidden[0][0].permute(1, 0, 2),
            ctrl_hidden[0][1].permute(1, 0, 2),
        ]
        state += ctrl_hidden  # len 2
        state.append(read_vecs)  # len 3
        state += memory_dict.values()  # len 9
        assert len(state) == 9, "Failed to verify packed state"
        return state

    def validate_unpack(self, dnc_output, unpacked_state):
        # Validate correct shapes
        s_ctrl_hidden, s_memory_dict, s_read_vecs = unpacked_state
        ctrl_hidden, memory_dict, read_vecs = dnc_output

        for i in range(len(ctrl_hidden)):
            for j in range(len(ctrl_hidden[i])):
                assert (
                    s_ctrl_hidden[i][j].shape == ctrl_hidden[i][j].shape
                ), f"Controller state mismatch: got {s_ctrl_hidden[i][j].shape} should be {ctrl_hidden[i][j].shape}"

        for k in memory_dict:
            assert (
                s_memory_dict[k].shape == memory_dict[k].shape
            ), f"Memory state mismatch at key {k}: got {s_memory_dict[k].shape} should be {memory_dict[k].shape}"

        assert (
            s_read_vecs.shape == read_vecs.shape
        ), f"Read state mismatch: got {s_read_vecs.shape} should be {read_vecs.shape}"

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

        # First run
        if self.dnc is None:
            (ctrl_hidden, read_vecs, memory_dict) = (None, None, None)
            gpu_id = flat.device.index if flat.device.index is not None else -1
            self.dnc = self.cfg["dnc_model"](
                input_size=self.obs_dim,
                hidden_size=self.cfg["hidden_size"],
                num_layers=self.cfg["num_layers"],
                read_heads=self.cfg["read_heads"],
                cell_size=self.cfg["cell_size"],
                nr_cells=self.cfg["nr_cells"],
                gpu_id=gpu_id,
            )
            output, (ctrl_hidden, memory_dict, read_vecs) = self.dnc(
                flat, (ctrl_hidden, memory_dict, read_vecs)
            )

            self.validate_unpack(
                (ctrl_hidden, memory_dict, read_vecs), self.unpack_state(state, B, T)
            )

        else:
            ctrl_hidden, memory_dict, read_vecs = self.unpack_state(state, B, T)
            output, (ctrl_hidden, memory_dict, read_vecs) = self.dnc(
                flat, (ctrl_hidden, memory_dict, read_vecs)
            )

        packed_state = self.pack_state(ctrl_hidden, memory_dict, read_vecs)

        # Compute action/value from output
        for t in range(T):
            logits[:, t] = self.logit_branch(output[:, t])
            values[:, t] = self.value_branch(output[:, t])

        logits = logits.reshape((B * T, self.num_outputs))
        values = values.reshape((B * T, 1))

        self.cur_val = values.squeeze(1)

        return logits, packed_state
