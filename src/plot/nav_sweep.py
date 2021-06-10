{
    "x": "timesteps_total",
    "x_label": "Training Timestep",
    "y_label": "Mean Reward per Train Batch",
    "y": "episode_reward_mean",
    "ci": 90,
    "range": (0.12, 0.52),
    "title": None,
    "smooth": 10,
    "group_category": "$|z|$",
    "trial_category": "GCM Prior",
    "num_samples": 500,
    "output": "/tmp/plots/gcm_sweep.pdf",
    "legend_offset": 0.9,
    "limit_line": None,
    "use_latex": True,

    "experiment_groups": [
        {
            "group_title": "$8$",
            "replay": 1,
            "data_prefix": "/Users/smorad/data/corl_2021_exp/navigation_sweep/h8/",
            "data": [
                {
                    "title": "None",
                    "trial_paths": ["none/*/progress.csv"],
                },
                {
                    "title": "Temporal",
                    "trial_paths": ["temporal/*/progress.csv"]
                },
                {
                    "title": "Spatial",
                    "trial_paths": ["spatial/*/progress.csv"]
                },
                {
                    "title": "Latent Sim.",
                    "trial_paths": ["vae/*/progress.csv"]
                },
            ]
        },
        {
            "group_title": "$16$",
            "replay": 1,
            "data_prefix": "/Users/smorad/data/corl_2021_exp/navigation_sweep/h16/",
            "data": [
                {
                    "title": "None",
                    "trial_paths": ["none/*/progress.csv"],
                },
                {
                    "title": "Temporal",
                    "trial_paths": ["temporal/*/progress.csv"]
                },
                {
                    "title": "Spatial",
                    "trial_paths": ["spatial/*/progress.csv"]
                },
                {
                    "title": "Latent Sim.",
                    "trial_paths": ["vae/*/progress.csv"]
                },
            ]
        },
        {
            "group_title": "$32$",
            "replay": 1,
            "data_prefix": "/Users/smorad/data/corl_2021_exp/navigation_sweep/h32/",
            "data": [
                {
                    "title": "None",
                    "trial_paths": ["none/*/progress.csv"],
                },
                {
                    "title": "Temporal",
                    "trial_paths": ["temporal/*/progress.csv"]
                },
                {
                    "title": "Spatial",
                    "trial_paths": ["spatial/*/progress.csv"]
                },
                {
                    "title": "Latent Sim.",
                    "trial_paths": ["vae/*/progress.csv"]
                },
            ]
        },
    ]
}
