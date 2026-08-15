"""
Quantile computation and bucket mapping for continuous lab measurement discretization.
"""

import pickle
from typing import Dict, Any, Optional
import numpy as np
import pandas as pd


def fit_quantiles(labevents: pd.DataFrame, num_quantiles: int = 11, min_samples: int = 20) -> Dict[Any, np.ndarray]:
    """Fits quantile boundaries for each lab item ID in labevents DataFrame."""
    quantile_map = {}

    for itemid, group in labevents.groupby("itemid"):
        vals = group["valuenum"].dropna()

        if len(vals) < min_samples:
            continue

        try:
            boundaries = np.quantile(vals, np.linspace(0, 1, num_quantiles))
            boundaries = np.unique(boundaries)
            quantile_map[itemid] = boundaries
        except Exception:
            pass

    return quantile_map


def apply_quantile(value: float, boundaries: np.ndarray) -> Optional[int]:
    """Maps a scalar continuous numerical value into its corresponding bucket index based on boundaries."""
    if pd.isna(value):
        return None

    bucket = np.digitize(value, boundaries[1:-1], right=True)
    return int(bucket)


def save_quantiles(quantile_map: Dict[Any, np.ndarray], filename: str = "quantiles.pkl") -> None:
    """Serializes quantile map to disk as a pickle file."""
    with open(filename, "wb") as f:
        pickle.dump(quantile_map, f)


def load_quantiles(filename: str = "quantiles.pkl") -> Dict[Any, np.ndarray]:
    """Loads serialized quantile map from disk."""
    with open(filename, "rb") as f:
        return pickle.load(f)
