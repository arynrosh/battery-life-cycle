"""Splits, normalization and PyTorch datasets."""
from __future__ import annotations

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset


def split_frames(data: pd.DataFrame, split_cfg: dict) -> dict[str, pd.DataFrame]:
    data = data[data.ambient_c.isin(split_cfg["temps"])]
    return {k: data[data.cycle.isin(split_cfg[k])].reset_index(drop=True)
            for k in ("train", "val", "test")}


class Standardizer:
    """Fit on the training set only, then apply everywhere."""

    def fit(self, x: np.ndarray):
        self.mean = x.mean(0)
        self.std = x.std(0) + 1e-8
        return self

    def __call__(self, x: np.ndarray) -> np.ndarray:
        return ((x - self.mean) / self.std).astype(np.float32)

    def state(self) -> dict:
        return {"mean": self.mean.tolist(), "std": self.std.tolist()}


class TabularDataset(Dataset):
    def __init__(self, x: np.ndarray, y: np.ndarray):
        self.x = torch.from_numpy(x)
        self.y = torch.from_numpy(y.astype(np.float32))

    def __len__(self):
        return len(self.y)

    def __getitem__(self, i):
        return self.x[i], self.y[i]


class WindowDataset(Dataset):
    """Sliding windows of length `seq_len`, built per run so a window never
    spans two runs. The first `seq_len-1` samples of each run are left-padded
    with the run's first sample, so every timestep gets a prediction."""

    def __init__(self, df: pd.DataFrame, x: np.ndarray, seq_len: int, stride: int = 1):
        self.seq_len = seq_len
        self.x_runs, self.y_runs, self.index = [], [], []
        for r, (_, idx) in enumerate(df.groupby("run_id", sort=False).indices.items()):
            xr = x[idx]
            pad = np.repeat(xr[:1], seq_len - 1, axis=0)
            self.x_runs.append(torch.from_numpy(np.concatenate([pad, xr])))
            self.y_runs.append(torch.from_numpy(df.soc.to_numpy()[idx].astype(np.float32)))
            self.index += [(r, t) for t in range(0, len(idx), stride)]

    def __len__(self):
        return len(self.index)

    def __getitem__(self, i):
        r, t = self.index[i]
        return self.x_runs[r][t:t + self.seq_len], self.y_runs[r][t]
