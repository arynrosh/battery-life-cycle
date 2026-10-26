"""Resampling and feature engineering.

Drive cycles are logged at 10 Hz with some variable-rate stretches. We bin to a
fixed rate (1 Hz by default) and add *causal* rolling means of voltage and
current. The rolling means give a memory-less model (FNN, trees) a sense of
recent load history, which is what separates "low voltage because SOC is low"
from "low voltage because the cell is under a big pulse right now".
"""
from __future__ import annotations

import numpy as np
import pandas as pd

BASE_FEATURES = ["voltage", "current", "temp_c"]


def resample(df: pd.DataFrame, hz: float = 1.0) -> pd.DataFrame:
    """Bin onto a uniform grid: mean of V/I/T per bin, last value of Ah/SOC."""
    bins = np.floor(df["time_s"].to_numpy() * hz).astype(np.int64)
    g = df.assign(_bin=bins).groupby("_bin", sort=True)
    out = g[["voltage", "current", "temp_c"]].mean()
    for col in ("ah", "soc"):
        if col in df:
            out[col] = g[col].last()
    out = out.reset_index()
    out["time_s"] = out.pop("_bin") / hz
    return out


def add_rolling_features(df: pd.DataFrame, windows_s: list[int], hz: float = 1.0) -> pd.DataFrame:
    """Causal (trailing) rolling means. Computed per file so nothing leaks
    across files or from the future."""
    df = df.copy()
    for w in windows_s:
        n = max(1, int(round(w * hz)))
        df[f"v_avg_{w}s"] = df["voltage"].rolling(n, min_periods=1).mean()
        df[f"i_avg_{w}s"] = df["current"].rolling(n, min_periods=1).mean()
    return df


def feature_columns(windows_s: list[int]) -> list[str]:
    cols = list(BASE_FEATURES)
    for w in windows_s:
        cols += [f"v_avg_{w}s", f"i_avg_{w}s"]
    return cols
