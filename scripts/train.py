"""Train and evaluate an SOC estimator.

    python scripts/train.py --model linear   # sanity-check baseline
    python scripts/train.py --model fnn      # feed-forward net on V/I/T + rolling means
    python scripts/train.py --model lstm     # LSTM on raw V/I/T sequences

Writes outputs/<model>/{metrics.json, per_run.csv, model.pt, plots/*.png}.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import torch  # noqa: E402
import yaml  # noqa: E402
from sklearn.linear_model import Ridge  # noqa: E402
from torch.utils.data import DataLoader  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from soc.data import Standardizer, TabularDataset, WindowDataset, split_frames  # noqa: E402
from soc.features import BASE_FEATURES, feature_columns  # noqa: E402
from soc.metrics import per_run_table, soc_metrics  # noqa: E402
from soc.models import FNN, LSTMRegressor, count_params, pick_device  # noqa: E402


def fit_torch(model, train_dl, val_dl, mcfg, device, log_every=1):
    opt = torch.optim.Adam(model.parameters(), lr=mcfg["lr"], weight_decay=mcfg.get("weight_decay", 0))
    sched = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, factor=0.5, patience=3)
    loss_fn = torch.nn.MSELoss()
    best, best_state, bad = float("inf"), None, 0
    for epoch in range(1, mcfg["epochs"] + 1):
        model.train()
        t0, tr = time.time(), 0.0
        for xb, yb in train_dl:
            xb, yb = xb.to(device), yb.to(device)
            opt.zero_grad()
            loss = loss_fn(model(xb), yb)
            loss.backward()
            opt.step()
            tr += loss.item() * len(yb)
        tr /= len(train_dl.dataset)
        val_rmse = np.sqrt(np.mean((predict(model, val_dl, device) - val_dl.dataset_y) ** 2)) * 100
        sched.step(val_rmse)
        if epoch % log_every == 0:
            print(f"  epoch {epoch:3d}  train_rmse {np.sqrt(tr)*100:5.2f}%  val_rmse {val_rmse:5.2f}%  ({time.time()-t0:.0f}s)", flush=True)
        if val_rmse < best - 1e-3:
            best, bad = val_rmse, 0
            best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
        else:
            bad += 1
            if bad >= mcfg["patience"]:
                print(f"  early stop at epoch {epoch}; best val_rmse {best:.2f}%")
                break
    model.load_state_dict(best_state)
    return model


@torch.no_grad()
def predict(model, dl, device) -> np.ndarray:
    model.eval()
    return np.concatenate([model(xb.to(device)).cpu().numpy() for xb, _ in dl])


def make_loader(ds, bs, shuffle):
    dl = DataLoader(ds, batch_size=bs, shuffle=shuffle, num_workers=0)
    return dl


def plot_runs(test: pd.DataFrame, out_dir: Path, model_name: str):
    out_dir.mkdir(parents=True, exist_ok=True)
    for temp, gt in test.groupby("ambient_c"):
        cycles = list(gt.cycle.unique())
        fig, axes = plt.subplots(len(cycles), 1, figsize=(9, 2.4 * len(cycles)), squeeze=False)
        for ax, cyc in zip(axes[:, 0], cycles):
            g = gt[gt.cycle == cyc]
            t = g.time_s / 60
            ax.plot(t, g.soc * 100, label="true", lw=1.5, color="#333")
            ax.plot(t, g.soc_pred * 100, label="pred", lw=1, color="#d1495b")
            m = soc_metrics(g.soc, g.soc_pred)
            ax.set_title(f"{cyc} @ {temp}°C — RMSE {m['rmse_pct']:.2f}%, max {m['max_abs_pct']:.1f}%", fontsize=9)
            ax.set_ylabel("SOC %")
            ax.grid(alpha=0.3)
        axes[0, 0].legend(loc="upper right", fontsize=8)
        axes[-1, 0].set_xlabel("time (min)")
        fig.suptitle(f"{model_name} — test drive cycles at {temp}°C")
        fig.tight_layout()
        fig.savefig(out_dir / f"test_{temp}C.png", dpi=110)
        plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", choices=["linear", "fnn", "lstm"], default="fnn")
    ap.add_argument("--config", default="configs/default.yaml")
    ap.add_argument("--epochs", type=int, help="override epochs (quick runs)")
    args = ap.parse_args()

    cfg = yaml.safe_load(open(args.config))
    torch.manual_seed(cfg["seed"])
    np.random.seed(cfg["seed"])
    out_dir = Path(cfg["outputs_dir"]) / args.model
    out_dir.mkdir(parents=True, exist_ok=True)

    data = pd.read_parquet(cfg["processed_path"])
    splits = split_frames(data, cfg["split"])
    for k, v in splits.items():
        print(f"{k:5s}: {v.run_id.nunique():2d} runs, {len(v):,} samples")

    # LSTM sees raw V/I/T and learns its own memory; the others get rolling means.
    cols = BASE_FEATURES if args.model == "lstm" else feature_columns(cfg["rolling_windows_s"])
    scaler = Standardizer().fit(splits["train"][cols].to_numpy())
    X = {k: scaler(v[cols].to_numpy()) for k, v in splits.items()}
    Y = {k: v.soc.to_numpy() for k, v in splits.items()}
    device = pick_device()
    t0 = time.time()

    if args.model == "linear":
        model = Ridge(alpha=1.0).fit(X["train"], Y["train"])
        preds = {k: np.clip(model.predict(X[k]), 0, 1) for k in ("val", "test")}
        n_params = len(cols) + 1
    else:
        mcfg = dict(cfg[args.model])
        if args.epochs:
            mcfg["epochs"] = args.epochs
        if args.model == "fnn":
            model = FNN(len(cols), mcfg["hidden"], mcfg.get("dropout", 0)).to(device)
            ds = {k: TabularDataset(X[k], Y[k]) for k in X}
        else:
            model = LSTMRegressor(len(cols), mcfg["hidden"], mcfg["layers"]).to(device)
            ds = {k: WindowDataset(splits[k], X[k], mcfg["seq_len"],
                                   stride=mcfg["stride"] if k == "train" else 1) for k in X}
        n_params = count_params(model)
        print(f"{args.model}: {n_params:,} params on {device}")
        train_dl = make_loader(ds["train"], mcfg["batch_size"], True)
        dls = {k: make_loader(ds[k], 4096, False) for k in ("val", "test")}
        dls["val"].dataset_y = Y["val"]
        model = fit_torch(model, train_dl, dls["val"], mcfg, device)
        preds = {k: predict(model, dls[k], device) for k in ("val", "test")}
        torch.save({"state_dict": model.state_dict(), "features": cols,
                    "scaler": scaler.state(), "config": mcfg}, out_dir / "model.pt")

    train_time = time.time() - t0
    test = splits["test"].assign(soc_pred=preds["test"])
    per_run = per_run_table(test)
    per_run.to_csv(out_dir / "per_run.csv", index=False)
    summary = {
        "model": args.model, "params": n_params, "train_time_s": round(train_time, 1),
        "features": cols,
        "val": soc_metrics(Y["val"], preds["val"]),
        "test": soc_metrics(Y["test"], preds["test"]),
        "test_by_temp": {int(t): soc_metrics(g.soc, g.soc_pred) for t, g in test.groupby("ambient_c")},
    }
    json.dump(summary, open(out_dir / "metrics.json", "w"), indent=2)
    plot_runs(test, out_dir / "plots", args.model)

    print(f"\nVAL  RMSE {summary['val']['rmse_pct']:.2f}%   TEST RMSE {summary['test']['rmse_pct']:.2f}%  "
          f"MAE {summary['test']['mae_pct']:.2f}%  max {summary['test']['max_abs_pct']:.1f}%")
    print(per_run.round(2).to_string(index=False))


if __name__ == "__main__":
    main()
