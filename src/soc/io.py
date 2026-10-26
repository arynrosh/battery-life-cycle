"""Readers for the McMaster LG HG2 dataset (Digatron cycler CSV exports)."""
from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

TEMP_DIRS = {
    "n20degC": -20, "n10degC": -10, "0degC": 0,
    "10degC": 10, "25degC": 25, "40degC": 40,
}
DRIVE_CYCLES = ["UDDS", "HWFET", "LA92", "US06"]
MIXED_CYCLES = [f"Mixed{i}" for i in range(1, 9)]

_FNAME = re.compile(r"^(?P<test_id>\d+)_(?P<name>.+)\.csv$")


@dataclass(frozen=True)
class TestFile:
    path: Path
    temp_dir: str
    ambient_c: int
    test_id: int
    name: str  # e.g. "UDDS", "Mixed3", "Cap_1C"


def list_test_files(raw_root: str | Path) -> list[TestFile]:
    """Index every CSV under <raw_root>/<temp_dir>/."""
    raw_root = Path(raw_root)
    files = []
    for temp_dir, ambient in TEMP_DIRS.items():
        for p in sorted((raw_root / temp_dir).glob("*.csv")):
            m = _FNAME.match(p.name)
            if m:
                files.append(TestFile(p, temp_dir, ambient, int(m["test_id"]), m["name"]))
    return files


def _prog_time_to_seconds(s: pd.Series) -> np.ndarray:
    """'HHH:MM:SS.fff' (hours can exceed 24) -> seconds as float."""
    parts = s.astype(str).str.split(":", expand=True).astype(float)
    return (parts[0] * 3600 + parts[1] * 60 + parts[2]).to_numpy()


def read_digatron_csv(path: str | Path) -> pd.DataFrame:
    """Read one cycler CSV and return tidy columns:
    time_s, voltage, current, temp_c, ah, status.

    The file has ~28 metadata lines, a header row starting with 'Time Stamp',
    then a units row which we skip. Current is negative on discharge.
    """
    path = Path(path)
    with open(path, encoding="latin-1") as f:
        for header_idx, line in enumerate(f):
            if line.startswith("Time Stamp"):
                break
        else:
            raise ValueError(f"No data header found in {path}")

    df = pd.read_csv(
        path,
        skiprows=list(range(header_idx)) + [header_idx + 1],
        encoding="latin-1",
        usecols=["Prog Time", "Status", "Voltage", "Current", "Temperature", "Capacity"],
    )
    out = pd.DataFrame({
        "time_s": _prog_time_to_seconds(df["Prog Time"]),
        "voltage": df["Voltage"].astype(float),
        "current": df["Current"].astype(float),
        "temp_c": df["Temperature"].astype(float),
        "ah": df["Capacity"].astype(float),
        "status": df["Status"].astype(str),
    })
    out["time_s"] -= out["time_s"].iloc[0]
    out["ah"] -= out["ah"].iloc[0]
    return out.dropna(subset=["voltage", "current"]).reset_index(drop=True)
