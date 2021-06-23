{
    "x": "timesteps_total",
    "x_label": "Training Timestep",
    "y_label": "Mean Reward per Train Batch",
    "y": "episode_reward_mean",
    "ci": 90,
    "range": (0.15, 1.05),
    "title": None,
    "smooth": 10,
    "group_category": "$n$",
    "trial_category": "Memory Module",
    "num_samples": 500,
    "output": "/tmp/plots/memory.pdf",
    "legend_offset": 0.88,
    "limit_line": None,
    "use_latex": True,

    # Each experiment group has its own plot
    "experiment_groups": [
        {
            "group_title": "$16$", 
            "replay": 0,
            "data_prefix": "/Users/smorad/data/corl_2021_exp/memory/8_cards/",
            # Each data is a collection of trials
            # Each trial is just an identical run, from which we compute mean/stddev
            "data": [
                {
                    "title": "MLP",
                    "trial_paths": ["mlp/*/progress.csv"]
                },
                {
                    "title": "LSTM",
                    "trial_paths": ["lstm/*/progress.csv"]
                },
                {
                    "title": "GTrXL",
                    "trial_paths": ["gtrxl/*/progress.csv"]
                },
                {
                    "title": "DNC",
                    "trial_paths": ["dnc/*/progress.csv"]
                },
                {
                    "title": "GCM",
                    "trial_paths": ["gcm/*/progress.csv"],
                },
            ]
        },
        {
            "group_title": "$20$", 
            "replay": 0,
            "data_prefix": "/Users/smorad/data/corl_2021_exp/memory/10_cards/",
            # Each data is a collection of trials
            # Each trial is just an identical run, from which we compute mean/stddev
            "data": [
                {
                    "title": "MLP",
                    "trial_paths": ["mlp/*/progress.csv"]
                },
                {
                    "title": "LSTM",
                    "trial_paths": ["lstm/*/progress.csv"]
                },
                {
                    "title": "GTrXL",
                    "trial_paths": ["gtrxl/*/progress.csv"]
                },
                {
                    "title": "DNC",
                    "trial_paths": ["dnc/*/progress.csv"]
                },
                {
                    "title": "GCM",
                    "trial_paths": ["gcm/*/progress.csv"],
                },
            ]
        },
        {
            "group_title": "$24$", 
            "replay": 0,
            "data_prefix": "/Users/smorad/data/corl_2021_exp/memory/12_cards/",
            # Each data is a collection of trials
            # Each trial is just an identical run, from which we compute mean/stddev
            "data": [
                {
                    "title": "MLP",
                    "trial_paths": ["mlp/*/progress.csv"]
                },
                {
                    "title": "LSTM",
                    "trial_paths": ["lstm/*/progress.csv"]
                },
                {
                    "title": "GTrXL",
                    "trial_paths": ["gtrxl/*/progress.csv"]
                },
                {
                    "title": "DNC",
                    "trial_paths": ["dnc/*/progress.csv"]
                },
                {
                    "title": "GCM",
                    "trial_paths": ["gcm/*/progress.csv"],
                },
            ]
        },
    ]
}
