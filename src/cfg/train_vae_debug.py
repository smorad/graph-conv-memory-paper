from cfg import train_vae

CFG = train_vae.CFG
CFG["ray"]["num_workers"] = 0
CFG["tune"]["stop"] = {"training_iteration": 1}
CFG["ray"]["train_batch_size"] = 8
CFG["ray"]["rollout_fragment_length"] = 8
