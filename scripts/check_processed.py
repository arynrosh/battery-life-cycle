"""read-only sanity checks on the processed Parquet.

    python scripts/check_processed.py
    python scripts/check_processed.py --config configs/default.yaml

Checks: missing values, coverage of every temperature x cycle, split sizes,
1 Hz time-grid gaps, SOC start/end, possible charge steps at the start of a
run, per-run capacity, and early cell-temperature rise (self-heating).
Nothing is written to disk.
"""
from __future__ import annotations

import argparse

import pandas as pd
import yaml


def main(cfg_path: str) -> None:
    cfg = yaml.safe_load(open(cfg_path))
    split = cfg["split"]
    step = 1.0 / cfg["resample_hz"]
    df = pd.read_parquet(cfg["processed_path"])
    pd.set_option("display.width", 200)
    pd.set_option("display.max_columns", 40)

    print(f"Rows: {len(df):,}   Runs: {df.run_id.nunique()}")
    print(f"Columns: {list(df.columns)}")

    # 1. Missing values
    print("\n== 1. Missing values per column ==")
    na = df.isna().sum()
    print(na[na > 0].to_string() if na.any() else "none")

    # 2. Coverage: rows per ambient temperature x cycle
    print("\n== 2. Rows per ambient temp x cycle (0 = run missing) ==")
    order = split["train"] + split["val"] + split["test"]
    piv = df.pivot_table(index="ambient_c", columns="cycle", values="soc",
                         aggfunc="size", fill_value=0)
    print(piv.reindex(columns=order, fill_value=0).to_string())

    # 3. Split sizes per temperature
    print("\n== 3. Rows per split x ambient temp ==")
    role = {c: k for k in ("train", "val", "test") for c in split[k]}
    sizes = (df.assign(split=df.cycle.map(role))
               .groupby(["ambient_c", "split"]).size()
               .unstack(fill_value=0)
               .reindex(columns=["train", "val", "test"], fill_value=0))
    sizes["train_%_of_25C"] = (100 * sizes["train"] / sizes.loc[25, "train"]).round(0) \
        if 25 in sizes.index else float("nan")
    print(sizes.to_string())

    # 4. Time grid: gaps and non-increasing time
    print(f"\n== 4. Time grid (expected step = {step:g} s) ==")
    dt = df.groupby("run_id", sort=False)["time_s"].diff()
    bad_order = int((dt <= 0).sum())
    gaps = df.assign(dt=dt)[dt > step + 1e-9]
    print(f"Non-increasing time steps: {bad_order}")
    print(f"Gaps (> 1 step): {len(gaps)} gaps in {gaps.run_id.nunique()} runs")
    if len(gaps):
        g = gaps.groupby("run_id")["dt"].agg(n_gaps="count", max_gap_s="max",
                                             total_missing_s="sum")
        g["total_missing_s"] -= g["n_gaps"] * step
        print(g.sort_values("total_missing_s", ascending=False).head(15).to_string())

    # 5. Per-run summary
    rows = []
    for run_id, r in df.groupby("run_id", sort=False):
        below = r.index[r.soc < 0.999]
        first_drop_t = r.loc[below[0], "time_s"] if len(below) else float("nan")
        early = r[r.time_s <= 600]
        rows.append({
            "run_id": run_id,
            "ambient": int(r.ambient_c.iloc[0]),
            "cycle": r.cycle.iloc[0],
            "minutes": r.time_s.iloc[-1] / 60,
            "soc_start": r.soc.iloc[0],
            "soc_end": r.soc.iloc[-1],
            "min_at_100%": first_drop_t / 60,      # long = rest or charge before discharge
            "ah_max": r.ah.max(),                   # > 0 = net charge above start point
            "cap_ah": r.capacity_ah.iloc[0],
            "T_start": r.temp_c.iloc[0],
            "T_rise_10min": early.temp_c.iloc[-1] - r.temp_c.iloc[0],
            "T_max": r.temp_c.max(),
        })
    s = pd.DataFrame(rows).set_index("run_id")

    print("\n== 5. Per-run summary: -20, -10 and 25 degC (for comparison) ==")
    focus = s[s.ambient.isin([-20, -10, 25])].sort_values(["ambient", "cycle"])
    print(focus.round(3).to_string())

    print("\n== 6. Runs that may start with a rest/charge step "
          "(> 2 min at 100% SOC or ah_max > 0.01 Ah) ==")
    flag = s[(s["min_at_100%"] > 2) | (s["ah_max"] > 0.01)]
    print(flag.round(3).to_string() if len(flag) else "none")

    print("\n== 7. Per-temperature averages (all runs) ==")
    agg = s.groupby("ambient").agg(
        runs=("cycle", "size"),
        cap_ah_mean=("cap_ah", "mean"),
        minutes_mean=("minutes", "mean"),
        T_start_mean=("T_start", "mean"),
        T_rise_10min_mean=("T_rise_10min", "mean"),
        T_max_mean=("T_max", "mean"),
    )
    print(agg.round(3).to_string())


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/default.yaml")
    main(ap.parse_args().config)