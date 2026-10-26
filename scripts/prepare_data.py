"""Build one tidy parquet of all drive-cycle runs with SOC labels.

    python scripts/prepare_data.py --config configs/default.yaml
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from soc.features import add_rolling_features, resample  # noqa: E402
from soc.io import DRIVE_CYCLES, MIXED_CYCLES, list_test_files, read_digatron_csv  # noqa: E402
from soc.labeling import add_soc, capacity_table, nearest_cap_1c, reference_capacity  # noqa: E402


def main(cfg_path: str) -> None:
    cfg = yaml.safe_load(open(cfg_path))
    files = list_test_files(cfg["raw_root"])
    if not files:
        sys.exit(f"No CSVs found under {cfg['raw_root']} - see README 'Get the data'.")

    caps = capacity_table(files)
    print("1C reference capacities (Ah):")
    print(caps.to_string(index=False))

    wanted = set(DRIVE_CYCLES + MIXED_CYCLES)
    frames, skipped = [], []
    for f in files:
        if f.name not in wanted:
            continue
        raw = read_digatron_csv(f.path)
        cap_1c = nearest_cap_1c(caps, f.temp_dir, f.test_id)
        cap = reference_capacity(float(-raw["ah"].min()), cap_1c)
        if cap is None:
            skipped.append(f"{f.temp_dir}_{f.test_id}_{f.name} (removed {-raw['ah'].min():.2f} Ah)")
            continue
        df = add_soc(raw, cap)
        df = resample(df, cfg["resample_hz"])
        df = add_rolling_features(df, cfg["rolling_windows_s"], cfg["resample_hz"])
        df["ambient_c"] = f.ambient_c
        df["cycle"] = f.name
        df["run_id"] = f"{f.temp_dir}_{f.test_id}_{f.name}"
        df["capacity_ah"] = cap
        frames.append(df)
        print(f"  {df.run_id[0]:<28} {len(df):>6} rows  SOC {df.soc.iloc[0]:.2f} -> {df.soc.iloc[-1]:.2f}")

    if skipped:
        print("\nSkipped incomplete runs:\n  " + "\n  ".join(skipped))
    data = pd.concat(frames, ignore_index=True)
    out = Path(cfg["processed_path"])
    out.parent.mkdir(parents=True, exist_ok=True)
    data.to_parquet(out, index=False)
    print(f"\nWrote {len(data):,} rows from {data.run_id.nunique()} runs -> {out}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/default.yaml")
    main(ap.parse_args().config)
