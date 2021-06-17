import ast
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
Config should be python dict. Example config:
"""
{
    # The x axis csv column
    "x": "agent_timesteps_total",
    # The label of x axis in the plot
    "x_label": "Training Timestep",
    # Y axis plot label
    "y_label": "Mean Reward per Train Batch",
    # Y axis csv column
    "y": "episode_reward_mean",
    # Range of plot as tuple (ymin, ymax)
    # Set to none to use default range
    "range": (20, 205),
    # Domain of plot, set to none to use default
    "domain": (0, 1e7),
    # Confidence interval, plotted along with mean
    # can use 'sd' for standard deviation instead
    "ci": 90,
    # Plot supertitle
    "title": None,
    # Smoothing window. Set this higher if your plot is jagged and you want it smooth
    # set to one for no smoothing
    "smooth": 10,
    # Label category spanned by the experiment groups
    # e.g. current config will produce 3 plots with titles
    # |z| = 8, |z| = 16, |z| = 32
    "group_category": "$|z|$",
    # The category of each line within the experiment
    # e.g. this will produce a legend titled Core Module with GCM, GTrXL, etc. entries
    "trial_category": "Core Module",
    # We resample the CSV data. This the number of datapoints used in the plot.
    # Increase for slower runtime but more accurate plots
    "num_samples": 500,
    # Where to save the output plot
    "output": "/tmp/plots/cartpole_200.pdf",
    # Offsets the legend in the x axis. This allows you
    # to move the legend outside of the plot, so it doesn't cover
    # the lines. Set to None for default behavior
    "legend_offset": 0.9,
    # If you'd like some dashed red line (like minimum success rate or something)
    # Set to None to not use it
    "limit_line": 195,
    # Whether latex should be used for rendering. You need to install all the latex bits
    # for this to work. Then use dollar notation for LaTeX math mode, e.g. "$|z|$"
    "use_latex": True,
    # Here is where we specify the plot data
    # Each experiment group creates a single plot
    "experiment_groups": [
        # Each dict here becomes a plot
        {
            # Subtitle for the specific plot, will be <group_category> = <group_title> 
            "group_title": "$8$",
            # If rllib experience replay is used, the proportion of replayed to original samples
            # this simply scales the x axis by 1 + replay (e.g. 1:1 or 50% replay would be replay=1)
            "replay": 0,
            # Prefix path to csv files. Not required, but allows for shorter trial_paths
            "data_prefix": "/Users/smorad/data/corl_2021_exp/cartpole/h8/",
            # The data files for each trial
            "data": [
                # Each dict here becomes a line in the plot
                {
                    # Name of this line in the legend
                    "title": "GCM",
                    # Path to trials. Trials will be meaned together to produce a single line
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
            "group_title": "$16$", 
            "replay": 0,
            "data_prefix": "/Users/smorad/data/corl_2021_exp/cartpole/h16/",
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
            "group_title": "$32$", 
            "replay": 0,
            "data_prefix": "/Users/smorad/data/corl_2021_exp/cartpole/h32/",
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


def main():
    parser = argparse.ArgumentParser(description="Plot experiments using seaborn")
    parser.add_argument("config", type=str, help="Path to config, interpreted as literal python dict")
    args = parser.parse_args()
    # Load config as dict
    with open(args.config, "r") as f:
        contents = f.read()
    cfg = ast.literal_eval(contents)

    if cfg.get("use_latex", False):
        rc('text', usetex=True)
    exps = []
    for exp_group in cfg["experiment_groups"]:
        group_exps = []
        for data in exp_group["data"]:
            try:
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
                    # Can't reindex if there are duplicate values in index
                    duped = run_data[i].index.duplicated()
                    if np.any(duped):
                        print(f'Detected duplicated timesteps in {trial_paths[i]}, removing: {run_data[i].index[duped]}')
                        run_data[i] = run_data[i][~duped]
                    run_data[i] = run_data[i].reindex(run_data[i].index.union(new_idx))
                    run_data[i] = run_data[i].interpolate('index').loc[new_idx]
                    # Smooth
                    run_data[i][cfg['y']] = run_data[i][cfg['y']].rolling(cfg['smooth'], min_periods=1).mean()
                    
                runs = pd.concat(run_data)
                runs['trial_name'] = data['title']
                group_exps.append(runs)
            except Exception:
                print(f"Failed to parse data, group: {data}\n{exp_group}")
                raise

        group = pd.concat(group_exps)
        group['group_title'] = exp_group['group_title']
        group.index = group.index * (exp_group['replay'] + 1)
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
    sb.set_palette("muted")
    plot = sb.relplot(
        data=df,
        x=df.index,
        y=cfg['y_label'],
        hue=cfg['trial_category'],
        col=cfg['group_category'],
        kind='line',
        linewidth=2,
        ci=cfg.get("ci", 95),
    )
    for line in plot.legend.get_lines():
        line.set_linewidth(6.0)

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
