from __future__ import annotations

import numpy as np
import pandas as pd


def soc_metrics(y_true, y_pred) -> dict:
    """Errors in percentage points of SOC."""
    e = (np.asarray(y_pred) - np.asarray(y_true)) * 100
    return {
        "rmse_pct": float(np.sqrt(np.mean(e ** 2))),
        "mae_pct": float(np.mean(np.abs(e))),
        "max_abs_pct": float(np.max(np.abs(e))),
    }


def per_run_table(df: pd.DataFrame, pred_col: str = "soc_pred") -> pd.DataFrame:
    rows = []
    for run_id, g in df.groupby("run_id", sort=False):
        rows.append({"run_id": run_id, "ambient_c": g.ambient_c.iloc[0],
                     "cycle": g.cycle.iloc[0], **soc_metrics(g.soc, g[pred_col])})
    return pd.DataFrame(rows).sort_values(["ambient_c", "cycle"])
