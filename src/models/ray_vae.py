import torch
import gym
from torch import nn
from typing import Union, Dict, List
from ray.rllib.models.torch.torch_modelv2 import TorchModelV2
from ray.rllib.models.modelv2 import ModelV2, restore_original_dimensions
from ray.rllib.utils.typing import ModelConfigDict, TensorType
from ray.rllib.models.torch.fcnet import FullyConnectedNetwork as TorchFC


from models.vae import VAE


class RayVAE(TorchModelV2, VAE):
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
        VAE.__init__(self)

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
        self.curr_ae_output, self.curr_mu, self.curr_logvar = VAE.forward(
            self, self.curr_ae_input
        )

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
            semantic.shape[1] + 1 + 1,
            *semantic.shape[2:],  # type : ignore
        )

        semantic_tgt_channel = semantic.shape[1]
        depth_channel = semantic_tgt_channel + 1

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
        if torch.abs(policy_loss[0]) > 1e-8:
            print("Warning -- nonzero policy loss: {policy_loss[0]}, zeroing it out...")

        # BCE = nn.functional.binary_cross_entropy(
        #        self.curr_ae_output, self.curr_ae_input, size_average=False).to(policy_loss[0].device)
        MSE = nn.functional.mse_loss(
            self.curr_ae_output, self.curr_ae_input, reduction="sum"
        )

        # see Appendix B from VAE paper:
        # Kingma and Welling. Auto-Encoding Variational Bayes. ICLR, 2014
        # 0.5 * sum(1 + log(sigma^2) - mu^2 - sigma^2)
        KLD = -0.5 * torch.mean(
            1 + self.curr_logvar - self.curr_mu.pow(2) - self.curr_logvar.exp()
        )
        combined = MSE + KLD - policy_loss[0]
        print(f"Loss: {combined.item()}")

        return [combined]
