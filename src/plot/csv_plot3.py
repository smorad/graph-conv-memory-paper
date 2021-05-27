import numpy as np
import argparse
import pandas as pd
import seaborn as sb
import itertools
import functools
import glob
import matplotlib.pyplot as plt
from matplotlib import rc
rc('text', usetex=True)


"""
Config should be python dict:
"""

example_config = {
    "x": "timesteps_total",
    "x_label": "Training Timestep",
    "y_label": "Mean Reward per Batch",
    "y": "episode_reward_mean",
    "title": None,
    "smooth": 20,
    "group_category": "$n$",
    "trial_category": "Model",
    "num_samples": 1000,
    "output": "/tmp/plots/my_plot.pdf",
    "legend_offset": 0.9,

    # Each experiment group has its own plot
    "experiment_groups": [
        {
            "group_title": "$16$", #r"$n=16$",
            "replay": 0,
            # Each data is a collection of trials
            # Each trial is just an identical run, from which we compute mean/stddev
            "data": [
                {
                    "title": "GCM",
                    "trial_paths": glob.glob("/Users/smorad/data/corl_2021/memory/*/*GraphConv*/progress.csv"),
                },
                {
                    "title": "LSTM",
                    "trial_paths": glob.glob("/Users/smorad/data/corl_2021/memory/8_card/*use_lstm*/progress.csv"),
                },
            ]
        },
        {
            "group_title": "$32$", #r"$n=16$",
            "replay": 0,
            # Each data is a collection of trials
            # Each trial is just an identical run, from which we compute mean/stddev
            "data": [
                {
                    "title": "GCM",
                    "trial_paths": glob.glob("/Users/smorad/data/corl_2021/memory/*/*GraphConv*/progress.csv"),
                },
                {
                    "title": "LSTM",
                    "trial_paths": glob.glob("/Users/smorad/data/corl_2021/memory/*/*use_lstm*/progress.csv"),
                },
            ]
        },
    ]
}


def main():
    cfg = example_config

    exps = []
    for exp_group in cfg["experiment_groups"]:
        group_exps = []
        for data in exp_group["data"]:
            
            # Mean/stddev trials with identical params
            run_data = []
            for trial_path in data["trial_paths"]:
                run = pd.read_csv(trial_path, usecols=[cfg["x"], cfg["y"]])
                run.set_index(cfg["x"])
                run_data.append(run)

            # Resample so all data is the same size and frequency for seaborn
            # stddev computation
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
        ci='sd',
    )

    if cfg.get('title', False):
        plot.fig.suptitle(args.title, y=0.93)
    if cfg.get('domain', False):
        plot.set(xlim=args.domain)
    if cfg.get('range', False):
        plot.set(ylim=args.range)

    plt.tight_layout()
    # Legend outisde of plots
    if cfg.get('legend_offset', False):
        plt.subplots_adjust(right=cfg['legend_offset'])
        plot.legend.set_bbox_to_anchor([1, 0.5])
    if cfg.get('output'):
        plt.savefig(cfg['output'])
    plt.show()


if __name__ == "__main__":
    main()
