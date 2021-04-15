import torch
import gym
from torch import nn
from typing import Union, Dict, List, Any
import ray
from ray.rllib.models.torch.torch_modelv2 import TorchModelV2
from ray.rllib.models.modelv2 import ModelV2, restore_original_dimensions
from ray.rllib.utils.typing import ModelConfigDict, TensorType
from ray.rllib.models.torch.fcnet import FullyConnectedNetwork as TorchFC


from models.depth_vae import DepthVAE

# from models.rgbd_resnet18_vae import VAE


DEFAULTS = {"z_dim": 64, "depth_weight": 1, "rgb_weight": 1, "elbo_beta": 1}


class DepthRayVAE(TorchModelV2, DepthVAE):
    def __init__(
        self,
        obs_space: gym.spaces.Space,
        action_space: gym.spaces.Space,
        num_outputs: int,
        model_config: ModelConfigDict,
        name: str,
        **custom_model_kwargs,
    ):
        super().__init__(obs_space, action_space, num_outputs, model_config, name)
        self.cfg = dict(DEFAULTS, **custom_model_kwargs)
        # VAE.__init__(self, z_dim=self.cfg["z_dim"], nc=4)
        DepthVAE.__init__(self, z_dim=self.cfg["z_dim"])
        self.rgb_loss_fn = nn.MSELoss(reduction="mean")
        self.depth_loss_fn = nn.MSELoss(reduction="mean")
        self.act_space = gym.spaces.utils.flatdim(action_space)
        self.visdom_imgs: Dict[str, Any] = {}

    def forward(self, input_dict, state, seq_lens):
        """Compute autoencoded image. Note the returned "logits"
        are random, as we want random actions"""
        self.curr_ae_input = self.to_img(input_dict)
        # if torch.any(self.curr_ae_input > 0):
        #        ray.util.pdb.set_trace()
        self.curr_ae_output, self.curr_mu, self.curr_logvar = DepthVAE.forward(
            self, self.curr_ae_input
        )
        batch = input_dict["obs_flat"].shape[0]
        device = input_dict["obs_flat"].device
        self.device = device
        self._curr_value = torch.zeros((batch,), device=device)
        out = torch.zeros((batch, self.act_space), device=device)

        self.visdom_imgs.clear()
        di = self.curr_ae_input[:64].cpu().detach()
        do = self.curr_ae_output[:64].cpu().detach()
        self.visdom_imgs[f"depth_in-{self.cfg}"] = torch.cat(
            (di, di, di), dim=1
        ).numpy()
        self.visdom_imgs[f"depth_out-{self.cfg}"] = torch.cat(
            (do, do, do), dim=1
        ).numpy()
        # Return [batch, action_space]
        return out, state

    def value_function(self):
        return self._curr_value

    def to_img(self, input_dict):
        """Build obs into an image tensor for feeding to nn"""
        depth = input_dict["obs"]["depth"]
        # [batch, channel, cols, rows]
        # To B, dim, h, w
        depth = depth.permute(0, 3, 1, 2)
        """
        x = torch.zeros(
            (batch, channels, cols, rows), dtype=torch.float32, device=rgb.device
        )
        x[:, 0:semantic_tgt_channel] = semantic
        # x[:, semantic_channels:semantic_tgt_channel] = sem_tgt
        x[:, depth_channel] = torch.squeeze(depth)
        """
        return depth

    def custom_loss(
        self, policy_loss: List[torch.Tensor], loss_inputs
    ) -> List[torch.Tensor]:

        self.sem_loss = torch.tensor([0], device=self.curr_ae_output.device)

        self.depth_loss = self.cfg["depth_weight"] * self.depth_loss_fn(
            self.curr_ae_output[:, -1, :, :],
            self.curr_ae_input[:, -1, :, :],
        )
        self.recon_loss = self.sem_loss + self.depth_loss
        self.kld_loss = self.cfg["elbo_beta"] * DepthVAE.kld_loss(
            self, self.curr_mu, self.curr_logvar
        )
        self.combined_loss = self.recon_loss + self.kld_loss

        return [self.combined_loss]

    def metrics(self):
        return {
            "depth_loss": self.depth_loss.detach().item(),
            "semantic_loss": self.sem_loss.detach().item(),
            "combined_loss": self.combined_loss.detach().item(),
        }
