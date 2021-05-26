import numpy as np
import argparse
import pandas as pd
import seaborn as sb
import itertools
import functools
import matplotlib.pyplot as plt
from matplotlib import rc
rc('text', usetex=True)



def pargs():
    p = argparse.ArgumentParser()
    p.add_argument(
        "-i", "--item", action="append", nargs='+', #metavar=("data_title", "run_path", "hue_group <optional>")
    )
    p.add_argument("--x-label", type=str, help="x axis label")
    p.add_argument("--y-label", type=str, help="y axis label")
    p.add_argument("--domain", type=float, nargs=2, help="plot domain")
    p.add_argument("--range", type=float, nargs=2, help="plot range")
    p.add_argument(
        "--replay",
        type=float,
        help="Ratio of replayed episodes to new episodes",
        default=0,
    )
    p.add_argument("--smooth", type=int, help="Smoothing window", default=1)
    p.add_argument(
        "-x", type=str, help="Column name in csv", default="agent_timesteps_total"
    )
    p.add_argument(
        "-y", type=str, help="Column name in csv", default="episode_reward_mean"
    )
    p.add_argument(
        '--title', type=str, help="Plot title"
    )
    p.add_argument(
        '--output', type=str, help="Output path"
    )
    p.add_argument(
        '--vertical', action='store_true', help='Hide ticks and x label except for the last plot'
    )

    return p.parse_args()


def main():
    args = pargs()

    dfs = {}
    for item in args.item:
        trial = item[0]
        run_path = item[1]

        df_path = f"{run_path}/progress.csv"
        dfs[trial] = pd.read_csv(
                df_path,
                usecols=[args.x, args.y],
            ).set_index(args.x).rename(
            columns={args.y: args.y_label},
            errors="raise"
        )
        dfs[trial].index.name = args.x_label

        if len(item) == 3:
            hue = item[2]
            dfs[trial]["hue_group"] = float(hue)

    df = pd.concat(dfs.values(), keys=dfs.keys(), axis=1)
    df = df.interpolate(method="index")

    # Scale x axis
    df.index *= args.replay + 1
    # Smooth
    df = df.rolling(args.smooth, min_periods=1).mean()
    sb.set_theme()
    sb.set_context('talk')
    #import pdb; pdb.set_trace()
    for i, trial in enumerate(df.columns.levels[0]):
        plot_kwargs = {
            "x": args.x_label,
            "y": args.y_label,
            "data": df[trial],
            "legend": "brief",
            "label": trial,
        }

        pt = sb.lineplot(**plot_kwargs)

        if args.title:
            pt.set(title=args.title)

        if args.vertical and i < len(df.columns.levels[0]) - 1:
            pt.set(xticklabels=[])
            pt.set(xlabel=None)

        if args.domain:
            plt.xlim(*args.domain)
        if args.range:
            plt.ylim(*args.range)

    plt.tight_layout()
    if args.output:
        plt.savefig(args.output)

    plt.show()



if __name__ == "__main__":
    main()
