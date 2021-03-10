import os
from cfg import train_vae


cfg_dir = os.path.abspath(os.path.dirname(__file__))

CFG = train_vae.CFG
CFG["ray"]["env_config"][
    "hab_cfg_path"
] = f"{cfg_dir}/objectnav_mp3d_train_val_mini.yaml"
CFG["ray"]["num_workers"] = 0
CFG["tune"]["stop"] = {"training_iteration": 1}
CFG["ray"]["train_batch_size"] = 8
CFG["ray"]["rollout_fragment_length"] = 8
