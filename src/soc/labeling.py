"""SOC ground truth via coulomb counting.

Every drive-cycle file in this dataset starts right after a full 1C/CV charge
(4.2 V, 50 mA cut-off), so SOC(t=0) = 100%. The cycler's Ah counter is reset at
the start of each file, so

    SOC(t) = 1 + Ah(t) / C_ref

where Ah is negative while discharging and C_ref is the cell's 1C discharge
capacity at that ambient temperature (from the Cap_1C test). The drive cycles
stop once 95% of C_ref is removed, so each file ends at ~5% SOC.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from .io import TestFile, read_digatron_csv


def capacity_table(files: list[TestFile]) -> pd.DataFrame:
    """Measured 1C capacity (Ah) for every Cap_1C test."""
    rows = []
    for f in files:
        if f.name == "Cap_1C":
            df = read_digatron_csv(f.path)
            rows.append({"temp_dir": f.temp_dir, "test_id": f.test_id,
                         "capacity_ah": float(-df["ah"].min())})
    return pd.DataFrame(rows).sort_values(["temp_dir", "test_id"])


def nearest_cap_1c(caps: pd.DataFrame, temp_dir: str, test_id: int) -> float:
    """Capacity from the most recent Cap_1C test at or before this test."""
    c = caps[caps.temp_dir == temp_dir]
    prior = c[c.test_id <= test_id]
    row = prior.iloc[-1] if len(prior) else c.iloc[0]
    return float(row.capacity_ah)


def reference_capacity(discharged_ah: float, cap_1c: float,
                       cutoff_frac: float = 0.95, min_complete: float = 0.85) -> float | None:
    """Reference capacity for one drive-cycle run, or None if the run is incomplete.

    The cycler stops each drive cycle once `cutoff_frac` (95%) of the 1C capacity
    is removed, so a complete run defines its own reference:
    C_ref = discharged_ah / 0.95. Where the matching Cap_1C test is in the CSV
    export this equals it to 3 decimals; it also covers runs whose Cap_1C test
    is missing (40degC test 556, 10degC test 567), where borrowing a later Cap_1C
    would push SOC below 0.

    Runs that removed < `min_complete` of the nearest Cap_1C were cut short
    (25degC HWFET stub, 40degC Mixed4-8) and are flagged for exclusion.
    """
    if discharged_ah < min_complete * cap_1c:
        return None
    return discharged_ah / cutoff_frac


def add_soc(df: pd.DataFrame, capacity_ah: float) -> pd.DataFrame:
    df = df.copy()
    df["soc"] = np.clip(1.0 + df["ah"] / capacity_ah, 0.0, 1.0)
    return df
