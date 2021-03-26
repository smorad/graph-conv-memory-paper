from cfg import base
from preprocessors.autoencoder.vae import PPVAE

CFG = base.CFG


CFG["ray"]["env_config"]["preprocessors"]["semantic_and_depth_autoencoder"] = PPVAE
