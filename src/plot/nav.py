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
    "trial_category": "Core Module",
    "num_samples": 500,
    "output": "/tmp/plots/nav.pdf",
    "legend_offset": 0.9,
    "limit_line": None,
    "use_latex": True,

    "experiment_groups": [
        {
            "group_title": "$8$",
            "replay": 1,
            "data_prefix": "/Users/smorad/data/corl_2021_exp/navigation/h8/",
            "data": [
                {
                    "title": "GCM",
                    "trial_paths": ["gcm/*/progress.csv"],
                },
                {
                    "title": "GTrXL",
                    "trial_paths": ["gtrxl_1t/*/progress.csv"]
                },
                {
                    "title": "LSTM",
                    "trial_paths": ["lstm/*/progress.csv"]
                },
                {
                    "title": "DNC",
                    "trial_paths": ["dnc/*/progress.csv"]
                },
                {
                    "title": "MLP",
                    "trial_paths": ["mlp/*/progress.csv"]
                },
            ]
        },
        {
            "group_title": "$16$", 
            "replay": 1,
            "data_prefix": "/Users/smorad/data/corl_2021_exp/navigation/h16/",
            "data": [
                {
                    "title": "GCM",
                    "trial_paths": ["gcm/*/progress.csv"],
                },
                {
                    "title": "GTrXL",
                    "trial_paths": ["gtrxl_1t/*/progress.csv"]
                },
                {
                    "title": "LSTM",
                    "trial_paths": ["lstm/*/progress.csv"]
                },
                {
                    "title": "DNC",
                    "trial_paths": ["dnc/*/progress.csv"]
                },
                {
                    "title": "MLP",
                    "trial_paths": ["mlp/*/progress.csv"]
                },
            ]
        },
        {
            "group_title": "$32$", 
            "replay": 1,
            "data_prefix": "/Users/smorad/data/corl_2021_exp/navigation/h32/",
            "data": [
                {
                    "title": "GCM",
                    "trial_paths": ["gcm/*/progress.csv"],
                },
                {
                    "title": "GTrXL",
                    "trial_paths": ["gtrxl_1t/*/progress.csv"]
                },
                {
                    "title": "LSTM",
                    "trial_paths": ["lstm/*/progress.csv"]
                },
                {
                    "title": "DNC",
                    "trial_paths": ["dnc/*/progress.csv"]
                },
                {
                    "title": "MLP",
                    "trial_paths": ["mlp/*/progress.csv"]
                },
            ]
        },
    ]
}
