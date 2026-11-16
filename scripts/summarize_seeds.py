"""Combine multi-seed results into mean ± std tables. Read-only.

    python scripts/summarize_seeds.py outputs/fnn
    python scripts/summarize_seeds.py outputs/fnn --save outputs/fnn/seed_summary.csv

Reads every <dir>/seed*/metrics.json and per_run.csv and prints:
  1. overall val/test RMSE, MAE, max per seed, and mean ± std
  2. test metrics per temperature: mean ± std across seeds
  3. per-run test RMSE: mean ± std across seeds
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd


def main(run_dir: str, save: str | None) -> None:
    seed_dirs = sorted(p for p in Path(run_dir).glob("seed*") if (p / "metrics.json").exists())
    if not seed_dirs:
        raise SystemExit(f"No seed*/metrics.json found under {run_dir}")
    pd.set_option("display.width", 200)

    rows, temp_rows, run_frames = [], [], []
    for d in seed_dirs:
        m = json.load(open(d / "metrics.json"))
        seed = m.get("seed", d.name.removeprefix("seed"))
        rows.append({"seed": seed,
                     "val_rmse": m["val"]["rmse_pct"],
                     "test_rmse": m["test"]["rmse_pct"],
                     "test_mae": m["test"]["mae_pct"],
                     "test_max": m["test"]["max_abs_pct"]})
        for t, v in m["test_by_temp"].items():
            temp_rows.append({"seed": seed, "ambient_c": int(t), **v})
        if (d / "per_run.csv").exists():
            run_frames.append(pd.read_csv(d / "per_run.csv").assign(seed=seed))

    overall = pd.DataFrame(rows).set_index("seed")
    print(f"== 1. Overall, {len(overall)} seeds ({run_dir}) ==")
    print(overall.round(2).to_string())
    print("mean ± std: " + "   ".join(
        f"{c} {overall[c].mean():.2f} ± {overall[c].std(ddof=1):.2f}" for c in overall.columns))

    by_temp = pd.DataFrame(temp_rows)
    agg = by_temp.groupby("ambient_c")[["rmse_pct", "mae_pct", "max_abs_pct"]].agg(["mean", "std"])
    print("\n== 2. Test metrics per temperature (mean ± std across seeds) ==")
    table = pd.DataFrame({
        c: agg[c]["mean"].map("{:.2f}".format) + " ± " + agg[c]["std"].map("{:.2f}".format)
        for c in ("rmse_pct", "mae_pct", "max_abs_pct")
    })
    print(table.to_string())

    if run_frames:
        runs = pd.concat(run_frames)
        r = runs.groupby(["ambient_c", "cycle"])["rmse_pct"].agg(["mean", "std"]).round(2)
        print("\n== 3. Per-run test RMSE (mean ± std across seeds) ==")
        print(r.to_string())

    if save:
        agg.columns = [f"{a}_{b}" for a, b in agg.columns]
        agg.round(4).to_csv(save)
        print(f"\nSaved per-temperature summary -> {save}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("run_dir", help="folder containing seed*/ subfolders, e.g. outputs/fnn")
    ap.add_argument("--save", help="optional CSV path for the per-temperature summary")
    a = ap.parse_args()
    main(a.run_dir, a.save)