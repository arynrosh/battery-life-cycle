"""Model definitions."""
from __future__ import annotations

import torch
from torch import nn


class FNN(nn.Module):
    """Feed-forward net: one time step (+ rolling-average features) -> SOC."""

    def __init__(self, n_in: int, hidden: list[int], dropout: float = 0.0):
        super().__init__()
        layers, d = [], n_in
        for h in hidden:
            layers += [nn.Linear(d, h), nn.LeakyReLU(0.1)]
            if dropout:
                layers.append(nn.Dropout(dropout))
            d = h
        layers += [nn.Linear(d, 1), nn.Sigmoid()]  # SOC lives in [0, 1]
        self.net = nn.Sequential(*layers)

    def forward(self, x):            # x: (B, n_in)
        return self.net(x).squeeze(-1)


class LSTMRegressor(nn.Module):
    """LSTM over the last `seq_len` samples of (V, I, T) -> SOC at the last step."""

    def __init__(self, n_in: int, hidden: int = 64, layers: int = 2):
        super().__init__()
        self.lstm = nn.LSTM(n_in, hidden, num_layers=layers, batch_first=True)
        self.head = nn.Sequential(nn.Linear(hidden, 1), nn.Sigmoid())

    def forward(self, x):            # x: (B, T, n_in)
        out, _ = self.lstm(x)
        return self.head(out[:, -1]).squeeze(-1)


def count_params(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def pick_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")
