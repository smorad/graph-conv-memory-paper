import numpy as np
import argparse
import pandas as pd
import seaborn as sb
import itertools
import functools


def pargs():
    p = argparse.ArgumentParser()
    p.add_argument(
        "-i", "--item", action="append", nargs=2, metavar=("data_title", "run_path")
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
    p.add_argument("--smooth", type=float, help="Smoothing window", default=1)
    p.add_argument(
        "-x", type=str, help="Column name in csv", default="agent_timesteps_total"
    )
    p.add_argument(
        "-y", type=str, help="Column name in csv", default="episode_reward_mean"
    )

    return p.parse_args()


def load_df(run_path):
    df_path = f"{run_path}/progress.csv"
    pd.read_csv(df_path)


def main():
    args = pargs()

    prev_df = None
    for title, run_path in args.item:
        df_path = f"{run_path}/progress.csv"
        df = pd.read_csv(df_path).set_index(args.x)
        if prev_df:
            df = pd.merge(prev_df, df, on=args.x)

    df = df.rename(
        columns={args.y: args.y_label},
        errors="raise",
    )
    df.index.name = args.x_label
    df = df.interpolate(method="index")

    # Scale x axis
    df.index *= args.replay + 1
    # Smooth
    df = df.rolling(args.smooth).mean()
    sb.lineplot(x=args.x_label, y=args.y_label, data=df)

    import pdb

    pdb.set_trace()


if __name__ == "__main__":
    main()
