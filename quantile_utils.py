import pandas as pd
import numpy as np
import pickle

def fit_quantiles(labevents):

    quantile_map = {}

    for itemid, group in labevents.groupby("itemid"):

        vals = group["valuenum"].dropna()

        if len(vals) < 20:
            continue

        try:
            boundaries = np.quantile(
                vals,
                np.linspace(0, 1, 11)
            )

            boundaries = np.unique(boundaries)

            quantile_map[itemid] = boundaries

        except Exception:
            pass

    return quantile_map


def apply_quantile(value, boundaries):

    if pd.isna(value):
        return None

    bucket = np.digitize(
        value,
        boundaries[1:-1],
        right=True
    )

    return int(bucket)


def save_quantiles(quantile_map,
                   filename="quantiles.pkl"):

    with open(filename, "wb") as f:
        pickle.dump(quantile_map, f)


def load_quantiles(
        filename="quantiles.pkl"):

    with open(filename, "rb") as f:
        return pickle.load(f)