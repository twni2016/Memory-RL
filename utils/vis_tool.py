import pandas as pd

pd.options.mode.chained_assignment = (
    None  # ignore all warnings like SettingWithCopyWarning
)

import numpy as np
from typing import Callable
import os, glob
import shutil
import pickle
from ml_collections.config_dict import ConfigDict


def _flatten_dict(input_dict, parent_key="", sep="."):
    """
    Based on https://github.com/google/flax/blob/main/flax/metrics/tensorboard.py
    Flattens and simplifies dict such that it can be used by hparams.
    Args:
        input_dict: Input dict, e.g., from ConfigDict.
        parent_key: String used in recursion.
        sep: String used to separate parent and child keys.
    Returns:
        Flattened dict.
    """
    items = []
    for k, v in input_dict.items():
        new_key = parent_key + sep + k if parent_key else k

        # Valid types according to https://github.com/tensorflow/tensorboard/blob/1204566da5437af55109f7a4af18f9f8b7c4f864/tensorboard/plugins/hparams/summary_v2.py
        valid_types = (
            bool,
            int,
            float,
            str,
            np.bool_,
            np.integer,
            np.floating,
            np.character,
        )

        if isinstance(v, dict) or isinstance(v, ConfigDict):
            # Recursively flatten the dict.
            if isinstance(v, ConfigDict):
                v = v.to_dict()
            items.extend(_flatten_dict(v, new_key, sep=sep).items())
            continue
        elif not isinstance(v, valid_types):
            # Cast any incompatible values as strings such that they can be handled by hparams
            v = str(v)
        items.append((new_key, v))
    return dict(items)


def walk_through(
    path: str,
    metric: str,
    query_fn: Callable,
    start: int,
    end: int,
    steps: int,
    window: int,
    cutoff: float = 0.9,
    delete: bool = False,
):
    def isnan(number):
        return np.isnan(float(number))

    def smooth(df):
        try:
            df = df.dropna(subset=[metric])  # remove NaN rows
        except KeyError:
            print("!!key error csv", run)
            if delete:
                shutil.rmtree(run)
                print("deleted")
            return None

        if isnan(df["env_steps"].iloc[-1]) or df["env_steps"].iloc[-1] < cutoff * end:
            # an incomplete run
            print("!!incomplete csv", run, "num steps", df["env_steps"].iloc[-1])
            if delete:
                shutil.rmtree(run)
                print("deleted")
            else:
                print("\n")
            return None

        # smooth by moving average
        df[metric] = df[metric].rolling(window=window, min_periods=1).mean()

        # update the columns with interpolated values and aligned steps
        aligned_step = np.linspace(start, end, steps).astype(np.int32)
        ## we only do interpolation, not extrapolation
        aligned_step = aligned_step[aligned_step <= df["env_steps"].iloc[-1]]
        aligned_value = np.interp(aligned_step, df["env_steps"], df[metric])

        # enlarge or reduce to same number of rows
        print(run, df.shape[0], df["env_steps"].iloc[-1])

        extracted_df = pd.DataFrame(
            data={
                "env_steps": aligned_step,
                metric: aligned_value,
            }
        )

        return extracted_df

    dfs = []
    i = 0

    runs = sorted(glob.glob(os.path.join(path, "*")))

    for run in runs:
        flags = pickle.load(open(os.path.join(run, "flags.pkl"), "rb"))

        if not query_fn(flags):
            continue

        csv_path = os.path.join(run, "progress.csv")
        try:
            df = pd.read_csv(open(csv_path))
        except pd.errors.EmptyDataError:
            print("!!empty csv", run)
            if delete:
                shutil.rmtree(run)
                print("deleted")
            continue

        df = smooth(df)
        if df is None:
            continue
        i += 1

        # concat flags (dot)
        pd_flags = pd.json_normalize(_flatten_dict(flags))
        df_flag = pd.concat([pd_flags] * df.shape[0], axis=0)  # repeat rows
        df_flag.index = df.index  # copy index
        df = pd.concat([df, df_flag], axis=1)
        dfs.append(df)
        # print(flags)

    print("\n in total:", i)
    dfs = pd.concat(dfs, ignore_index=True)
    return dfs
