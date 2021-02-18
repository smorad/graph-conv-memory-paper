import torch
import gym
from torch import nn
from typing import Union, Dict, List
import ray
from ray.rllib.models.torch.torch_modelv2 import TorchModelV2
from ray.rllib.models.modelv2 import ModelV2, restore_original_dimensions
from ray.rllib.utils.typing import ModelConfigDict, TensorType
from ray.rllib.models.torch.fcnet import FullyConnectedNetwork as TorchFC


from models.ae import CNNAutoEncoder


class RayAE(TorchModelV2, CNNAutoEncoder):
    def __init__(
        self,
        obs_space: gym.spaces.Space,
        action_space: gym.spaces.Space,
        num_outputs: int,
        model_config: ModelConfigDict,
        name: str,
    ):
        super().__init__(obs_space, action_space, num_outputs, model_config, name)
        # TODO: We should inherit from VAE instead of set to member var
        # super(VAE).__init__()
        CNNAutoEncoder.__init__(self)
        self.sem_loss_fn = nn.CosineEmbeddingLoss(reduction="mean")
        # self.sem_loss_fn = nn.BCELoss(reduction='mean')
        self.depth_loss_fn = nn.MSELoss(reduction="mean")
        # self.depth_loss_fn = nn.BCELoss(reduction='mean')

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

    def forward(self, input_dict, state, seq_lens):
        """Compute autoencoded image. Note the returned "logits"
        are random, as we want random actions"""
        self.curr_ae_input = self.to_img(input_dict)
        # TODO figure out why inheritance is such shit
        self.curr_ae_output = CNNAutoEncoder.forward(self, self.curr_ae_input)
        obs = input_dict["obs"]
        self._curr_value = torch.zeros((obs["gps"].shape[0],)).to(obs["gps"].device)
        out = torch.zeros((obs["gps"].shape[0], 5)).to(obs["gps"].device)
        # Return [batch, action_space]
        return out, state

    def value_function(self):
        return self._curr_value

    def to_img(self, input_dict):
        """Build obs into an image tensor for feeding to nn"""
        semantic = input_dict["obs"]["semantic"]
        depth = input_dict["obs"]["depth"]
        # [batch, channel, cols, rows]
        batch, channels, cols, rows = (  # type: ignore
            semantic.shape[0],
            semantic.shape[1] + 1,
            *semantic.shape[2:],  # type : ignore
        )

        semantic_tgt_channel = semantic.shape[1]
        depth_channel = semantic_tgt_channel  # + 1

        x = torch.zeros(
            (batch, channels, cols, rows), dtype=torch.float32, device=semantic.device
        )
        x[:, 0:semantic_tgt_channel] = semantic
        # x[:, semantic_channels:semantic_tgt_channel] = sem_tgt
        x[:, depth_channel] = torch.squeeze(depth)
        return x

    def custom_loss(
        self, policy_loss: List[torch.Tensor], loss_inputs
    ) -> List[torch.Tensor]:
        # How important is depth compared to semantic
        # depth_to_sem_ratio = 1 / 4
        """
        cos_sem = self.sem_loss_fn(
            self.curr_ae_output[:, :-1, :, :],
            self.curr_ae_input[:, :-1, :, :],
            torch.ones(self.curr_ae_input.shape[-2:], device=self.curr_ae_input.device)
        )
        """
        if not hasattr(self, "sem_tgt"):
            self.sem_tgt = torch.ones(
                self.curr_ae_input.shape[-2:], device=self.curr_ae_input.device
            )

        self.sem_loss = self.sem_loss_fn(
            self.curr_ae_output[:, :-1, :, :],
            self.curr_ae_input[:, :-1, :, :],
            self.sem_tgt,
        )

        self.depth_loss = self.depth_loss_fn(
            self.curr_ae_output[:, -1, :, :],
            self.curr_ae_input[:, -1, :, :],
        )
        # self.depth_loss = MSE_sem * sem_factor + MSE_depth * depth_to_sem_ratio
        self.combined_loss = self.sem_loss + self.depth_loss
        # For compatibility with vae logger
        print(
            f"Losses: Sem: {self.sem_loss} Depth: {self.depth_loss} Combined: {self.combined_loss.item()}"
        )

        return [self.combined_loss]

    def metrics(self):
        return {
            "depth_loss": self.depth_loss.detach().item(),
            "semantic_loss": self.sem_loss.detach().item(),
            "combined_loss": self.combined_loss.detach().item(),
        }
