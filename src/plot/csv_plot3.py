import numpy as np
import argparse
import pandas as pd
import seaborn as sb
import itertools
import functools
import glob
import matplotlib.pyplot as plt
from matplotlib import rc


"""
Config should be python dict:
"""

memory_config = {
    "x": "timesteps_total",
    "x_label": "Training Timestep",
    "y_label": "Mean Reward per Train Batch",
    "y": "episode_reward_mean",
    "range": (0.15, 1.05),
    "title": None,
    "smooth": 10,
    "group_category": "$n$",
    "trial_category": "Core Module",
    "num_samples": 500,
    "output": "/tmp/plots/memory.pdf",
    "legend_offset": 0.9,
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
                    "title": "GCM",
                    "trial_paths": ["gcm/*/progress.csv"],
                },
                {
                    "title": "GTrXL",
                    "trial_paths": ["gtrxl/*/progress.csv"]
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
            "group_title": "$20$", 
            "replay": 0,
            "data_prefix": "/Users/smorad/data/corl_2021_exp/memory/10_cards/",
            # Each data is a collection of trials
            # Each trial is just an identical run, from which we compute mean/stddev
            "data": [
                {
                    "title": "GCM",
                    "trial_paths": ["gcm/*/progress.csv"],
                },
                {
                    "title": "GTrXL",
                    "trial_paths": ["gtrxl/*/progress.csv"]
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
            "group_title": "$24$", 
            "replay": 0,
            "data_prefix": "/Users/smorad/data/corl_2021_exp/memory/12_cards/",
            # Each data is a collection of trials
            # Each trial is just an identical run, from which we compute mean/stddev
            "data": [
                {
                    "title": "GCM",
                    "trial_paths": ["gcm/*/progress.csv"],
                },
                {
                    "title": "GTrXL",
                    "trial_paths": ["gtrxl/*/progress.csv"]
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

nav_config = {
    "x": "timesteps_total",
    "x_label": "Training Timestep",
    "y_label": "Mean Reward per Train Batch",
    "y": "episode_reward_mean",
    "range": (0.12, 0.52),
    "title": None,
    "smooth": 10,
    "group_category": "$|h|$",
    "trial_category": "Core Module",
    "num_samples": 500,
    "output": "/tmp/plots/nav.pdf",
    "legend_offset": 0.9,

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


def main():
    #cfg = memory_config
    cfg = nav_config

    if cfg.get("use_latex", False):
        rc('text', usetex=True)
    exps = []
    for exp_group in cfg["experiment_groups"]:
        group_exps = []
        for data in exp_group["data"]:
            
            # Mean/stddev trials with identical params
            run_data = []
            if exp_group.get('data_prefix', False):
                trial_paths = [glob.glob(f"{exp_group['data_prefix']}/{p}") for p in data["trial_paths"]]
            else:
                trial_paths = [glob.glob(p) for p in data["trial_paths"]]
            # Flatten
            trial_paths = list(itertools.chain(*trial_paths))
            for trial_path in trial_paths:
                run = pd.read_csv(trial_path, usecols=[cfg["x"], cfg["y"]])
                run.set_index(cfg["x"])
                run_data.append(run)

            # Resample so all data is the same size and frequency for seaborn
            # stddev computation
            if len(run_data) == 0:
                run_str = data
                print(f'Warning, run data for {data} is empty, skipping...')
                continue
            max_size = max([r[cfg['x']].max() for r in run_data])
            new_idx = np.linspace(0, max_size, cfg['num_samples'])
            for i in range(len(run_data)):
                run_data[i] = run_data[i].set_index(cfg['x'])
                run_data[i] = run_data[i].reindex(run_data[i].index.union(new_idx))
                run_data[i] = run_data[i].interpolate('index').loc[new_idx]
                # Smooth
                run_data[i][cfg['y']] = run_data[i][cfg['y']].rolling(cfg['smooth'], min_periods=1).mean()
                
            runs = pd.concat(run_data)
            runs['trial_name'] = data['title']
            group_exps.append(runs)

        group = pd.concat(group_exps)
        group['group_title'] = exp_group['group_title']
        group.index = group.index * (exp_group['replay'] + 1)
        #res = group.groupby('trial_name')[cfg['y']].transform(lambda s: s.rolling(cfg['smooth'], min_periods=1).mean())
        #group[cfg['y']] = group[cfg['y']].rolling(cfg['smooth'], min_periods=1).mean()
        exps.append(group)

    df = pd.concat(exps)

    # Now rename everything before plotting
    df = df.rename(columns={
        cfg['y']: cfg['y_label'],
        'trial_name': cfg['trial_category'],
        'group_title': cfg['group_category'],
        })
    df.index = df.index.rename(cfg['x_label'])

    # plotting 
    sb.set_theme()
    sb.set_context('talk')
    sb.set_palette('colorblind')
    plot = sb.relplot(
        data=df,
        x=df.index,
        y=cfg['y_label'],
        hue=cfg['trial_category'],
        col=cfg['group_category'],
        kind='line',
        linewidth=1.5,
        ci=90#'sd',
    )

    if cfg.get('title', False):
        plot.fig.suptitle(cfg['title'], y=0.93)
    if cfg.get('domain', False):
        plot.set(xlim=cfg['domain'])
    if cfg.get('range', False):
        plot.set(ylim=cfg['range'])
    if cfg.get('limit_line', False):
        for a in plot.axes[0]:
            a.axhline(cfg['limit_line'], ls='--', c='red')


    plt.tight_layout(pad=0.5)
    # Legend outisde of plots
    if cfg.get('legend_offset', False):
        plt.subplots_adjust(right=cfg['legend_offset'])
        plot.legend.set_bbox_to_anchor([1, 0.5])
    if cfg.get('output'):
        plt.savefig(cfg['output'])
    plt.show()


if __name__ == "__main__":
    main()
