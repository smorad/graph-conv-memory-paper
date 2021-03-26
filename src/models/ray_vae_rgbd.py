import torch
import gym
from torch import nn
from typing import Union, Dict, List, Any
import ray
from ray.rllib.models.torch.torch_modelv2 import TorchModelV2
from ray.rllib.models.modelv2 import ModelV2, restore_original_dimensions
from ray.rllib.utils.typing import ModelConfigDict, TensorType
from ray.rllib.models.torch.fcnet import FullyConnectedNetwork as TorchFC


from models.rgbd_vae import RGBDVAE

# from models.rgbd_resnet18_vae import VAE


DEFAULTS = {"z_dim": 64, "depth_weight": 1, "rgb_weight": 1, "elbo_beta": 1}


class RGBDRayVAE(TorchModelV2, RGBDVAE):
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
        RGBDVAE.__init__(self, z_dim=self.cfg["z_dim"])
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
        self.curr_ae_output, self.curr_mu, self.curr_logvar = RGBDVAE.forward(
            self, self.curr_ae_input
        )
        batch = input_dict["obs_flat"].shape[0]
        device = input_dict["obs_flat"].device
        self._curr_value = torch.zeros((batch,), device=device)
        out = torch.zeros((batch, self.act_space), device=device)

        self.visdom_imgs.clear()
        self.visdom_imgs[f"rgb_in-{self.cfg}"] = (
            self.curr_ae_input[:64, 0:3].cpu().detach().numpy()
        )
        self.visdom_imgs[f"rgb_out-{self.cfg}"] = (
            self.curr_ae_output[:64, 0:3].cpu().detach().numpy()
        )
        di = self.curr_ae_input[:64, 3].cpu().detach()
        do = self.curr_ae_output[:64, 3].cpu().detach()
        self.visdom_imgs[f"depth_in-{self.cfg}"] = torch.stack(
            (di, di, di), dim=1
        ).numpy()
        self.visdom_imgs[f"depth_out-{self.cfg}"] = torch.stack(
            (do, do, do), dim=1
        ).numpy()
        # Return [batch, action_space]
        return out, state

    def value_function(self):
        return self._curr_value

    def to_img(self, input_dict):
        """Build obs into an image tensor for feeding to nn"""
        rgb = input_dict["obs"]["rgb"]
        depth = input_dict["obs"]["depth"]
        # input shape B, h, w, dim
        rgb = rgb / 255.0
        assert rgb.min() >= 0.0 and rgb.max() <= 1.0
        x = torch.cat((rgb, depth), dim=-1)
        # To B, dim, h, w
        x = x.permute(0, 3, 1, 2)
        return x

    def custom_loss(
        self, policy_loss: List[torch.Tensor], loss_inputs
    ) -> List[torch.Tensor]:

        self.sem_loss = self.cfg["rgb_weight"] * self.rgb_loss_fn(
            self.curr_ae_output[:, :-1, :, :],
            self.curr_ae_input[:, :-1, :, :],
            # self.sem_tgt,
        )

        self.depth_loss = self.cfg["depth_weight"] * self.depth_loss_fn(
            self.curr_ae_output[:, -1, :, :],
            self.curr_ae_input[:, -1, :, :],
        )
        self.recon_loss = self.sem_loss + self.depth_loss
        self.kld_loss = self.cfg["elbo_beta"] * RGBDVAE.kld_loss(
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
