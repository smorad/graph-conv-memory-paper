import torch
import gym
from torch import nn
from typing import Union, Dict, List
import ray
from ray.rllib.models.torch.torch_modelv2 import TorchModelV2
from ray.rllib.models.modelv2 import ModelV2, restore_original_dimensions
from ray.rllib.utils.typing import ModelConfigDict, TensorType
from ray.rllib.models.torch.fcnet import FullyConnectedNetwork as TorchFC


from models.vae import VAE


DEFAULTS = {"z_dim": 128, "depth_weight": 1, "semantic_weight": 1, "elbo_beta": 1}


class FocalLoss(nn.Module):
    def __init__(self, alpha=0.75, gcmma=2.0):
        self.alpha = alpha
        self.gcmma = gcmma
        super(FocalLoss, self).__init__()

    def forward(self, inputs, targets, smooth=1):
        # flatten label and prediction tensors
        inputs = inputs.flatten()
        targets = targets.flatten()

        # first compute binary cross-entropy
        BCE = nn.functional.binary_cross_entropy(inputs, targets, reduction="mean")
        BCE_EXP = torch.exp(-BCE)
        focal_loss = self.alpha * (1 - BCE_EXP) ** self.gcmma * BCE

        return focal_loss


class DiceLoss(nn.Module):
    def __init__(self, weight=None, size_average=True):
        super(DiceLoss, self).__init__()

    def forward(self, inputs, targets, smooth=1):
        # flatten label and prediction tensors
        inputs = inputs.flatten()
        targets = targets.flatten()

        intersection = (inputs * targets).sum()
        dice = (2.0 * intersection + smooth) / (inputs.sum() + targets.sum() + smooth)

        return 1 - dice


class DiceBCELoss(nn.Module):
    def __init__(self, weight=None, size_average=True):
        super(DiceBCELoss, self).__init__()

    def forward(self, inputs, targets, smooth=1):
        # flatten label and prediction tensors
        inputs = inputs.flatten()
        targets = targets.flatten()

        intersection = (inputs * targets).sum()
        dice_loss = 1 - (2.0 * intersection + smooth) / (
            inputs.sum() + targets.sum() + smooth
        )
        BCE = nn.functional.binary_cross_entropy(inputs, targets, reduction="mean")
        Dice_BCE = BCE + dice_loss

        return Dice_BCE


class RayVAE(TorchModelV2, VAE):
    def __init__(
        self,
        obs_space: gym.spaces.Space,
        action_space: gym.spaces.Space,
        num_outputs: int,
        model_config: ModelConfigDict,
        name: str,
        **custom_model_kwargs
    ):
        super().__init__(obs_space, action_space, num_outputs, model_config, name)
        self.cfg = dict(DEFAULTS, **custom_model_kwargs)
        VAE.__init__(self, z_dim=self.cfg["z_dim"])
        self.sem_loss_fn = (
            DiceBCELoss()
        )  # FocalLoss() #self.weighted_sem_loss #nn.CosineEmbeddingLoss(reduction="mean")
        self._sem_loss_fn = nn.BCELoss(reduction="none")
        self.depth_loss_fn = nn.MSELoss(reduction="mean")
        self.act_space = gym.spaces.utils.flatdim(action_space)
        self.export_video = model_config.get("export_video", False)

    def weighted_sem_loss(self, x, tgt):
        """Semantic loss weighted by category appearance"""
        loss_per_channel = self._sem_loss_fn(x, tgt).mean((0, 2, 3))
        loss = torch.sum(loss_per_channel * self.channel_weights)
        return loss

    def compute_semantic_weights(self, input_tensor, e=1e-8, beta=0.99):
        """Computes channel weights for the batch, such that less-common objects
        contribute more to loss than common objects. See Class Balanced Loss
        for the theory behind this."""
        # Sum across batches and pixels, but not channels!
        px_per_channel = input_tensor.sum((0, 2, 3))
        effective_num = 1.0 - torch.pow(beta, px_per_channel) + e
        weights = (1.0 - beta) / effective_num
        weights = weights / torch.sum(weights)

        return weights

    def forward(self, input_dict, state, seq_lens):
        """Compute autoencoded image. Note the returned "logits"
        are random, as we want random actions"""
        self.curr_ae_input = self.to_img(input_dict)
        # self.channel_weights = self.compute_semantic_weights(self.curr_ae_input[:,:-1,:,:])
        self.curr_ae_output, self.curr_mu, self.curr_logvar = VAE.forward(
            self, self.curr_ae_input
        )
        batch = input_dict["obs_flat"].shape[0]
        device = input_dict["obs_flat"].device
        self._curr_value = torch.zeros((batch,), device=device)
        out = torch.zeros((batch, self.act_space), device=device)

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
        if not hasattr(self, "sem_tgt"):
            self.sem_tgt = torch.ones(
                self.curr_ae_input.shape[-2:], device=self.curr_ae_input.device
            )

        self.sem_loss = self.cfg["semantic_weight"] * self.sem_loss_fn(
            self.curr_ae_output[:, :-1, :, :],
            self.curr_ae_input[:, :-1, :, :],
            # self.sem_tgt,
        )

        self.depth_loss = self.cfg["depth_weight"] * self.depth_loss_fn(
            self.curr_ae_output[:, -1, :, :],
            self.curr_ae_input[:, -1, :, :],
        )
        self.recon_loss = self.sem_loss + self.depth_loss
        self.kld_loss = self.cfg["elbo_beta"] * VAE.kld_loss(
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
