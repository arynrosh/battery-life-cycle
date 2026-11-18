# SOC estimation for the LG 18650HG2 cell

Data-driven state-of-charge (SOC) estimation for a 3 Ah LG HG2 18650 cell, using the
McMaster drive-cycle dataset (Kollmeyer et al.). Given measured **voltage, current and
temperature**, the models predict SOC (0–100%) across six ambient temperatures, from −20 °C to 40 °C.

## Quick start

```bash
git clone <this repo> && cd battery-soc
python -m venv .venv && source .venv/bin/activate      # Windows: .venv\Scripts\activate
pip install -r requirements.txt

# 1. Put the dataset at data/raw/LG_HG2_data/  (folders n20degC … 40degC)
# 2. Build the processed parquet (~20 s)
python scripts/prepare_data.py
# 3. Train + evaluate
python scripts/train.py --model linear   # sanity baseline, seconds
python scripts/train.py --model fnn      # ~1–2 min on a laptop CPU
python scripts/train.py --model lstm     # ~30+ min on CPU, a few min on a GPU
```

Each run writes to `outputs/<model>/`: `metrics.json`, `per_run.csv`, `model.pt`, and
`plots/test_<temp>C.png` (true vs predicted SOC for every test drive cycle).

### Get the data
The dataset is ~600 MB, so it is **not committed**; `data/` is gitignored. Download it from
Mendeley Data (*LG 18650HG2 Li-ion Battery Data*, doi:10.17632/b5mj79w5w9.2) or use the
team's shared copy, then unzip it so the layout is:

```
data/raw/LG_HG2_data/{n20degC,n10degC,0degC,10degC,25degC,40degC}/*.csv
```

## Repo layout

```
configs/default.yaml     all knobs: split, features, model hyper-params
src/soc/io.py            parse Digatron CSVs -> time_s, voltage, current, temp_c, ah
src/soc/labeling.py      SOC ground truth (coulomb counting)
src/soc/features.py      1 Hz resampling + causal rolling-mean features
src/soc/data.py          file-level splits, standardizer, PyTorch datasets
src/soc/models.py        FNN and LSTM
src/soc/metrics.py       RMSE / MAE / max error in % SOC
scripts/prepare_data.py  raw CSV -> data/processed/drive_cycles_1hz.parquet
scripts/train.py         train + evaluate + plot
notebooks/               exploration (keep notebooks thin; import from src/)
```

## How it works

### 1. Ground-truth SOC
Every drive-cycle file starts right after a full CC-CV charge (4.2 V, 50 mA cut-off), and the
cycler resets its Ah counter at the start of each file. So:

```
SOC(t) = 1 + Ah(t) / C_ref          (Ah < 0 while discharging)
```

The drive cycles are stopped once **95% of the cell's 1C capacity** at that temperature has
been removed, so every complete run ends at 5% SOC. We use that to get `C_ref` from each
run itself (`discharged Ah / 0.95`). Where the matching `Cap_1C` test is in the export, this
reproduces its measured capacity to 3 decimals. It also covers runs whose capacity test is
missing from the CSVs (40 °C test 556, 10 °C test 567). Without it, those runs get labels
below 0%.

Capacity changes a lot with temperature (≈2.7 Ah at 25 °C vs ≈1.6 Ah at −20 °C), so SOC is
relative to the capacity available *at that temperature*.

### 2. Data quirks we handle
* **Variable sample rate** (10 Hz during drive cycles, slower during pauses): everything is
  binned to 1 Hz (mean V/I/T, last Ah).
* **Incomplete files are dropped**: `25degC/551_HWFET` (a 10-minute stub with no discharge)
  and `40degC/562_Mixed4–8` (each stops after ~0.19 Ah). As a result there's **no HWFET test
  at 25 °C** and no Mixed validation run at 40 °C.
* Some temperatures lack `Mixed3` (0 °C, 10 °C).

### 3. Train / validation / test split
We split by **whole files, never by random rows**. Neighbouring 1 Hz samples are nearly
identical, so a random row split leaks and gives misleadingly good scores.

| split | cycles | runs |
|---|---|---|
| train | Mixed1–7 (random mixes of the four standard cycles) | 36 |
| val   | Mixed8 (used for early stopping) | 5 |
| test  | UDDS, HWFET, LA92, US06 | 23 |

All six temperatures are in every split; temperature is an input feature. This follows the
protocol of the papers that published this dataset, so our numbers are comparable to theirs.

### 4. Features & models
* **linear / fnn**: V, I, T plus causal (trailing) rolling means of V and I over 30 s, 120 s
  and 300 s. The rolling means give a memoryless model a sense of recent load. Without them it
  can't tell "low voltage because SOC is low" from "low voltage because of a current pulse
  right now".
* **lstm**: raw V, I, T over the last 120 s; it learns its own memory.
* Inputs are standardized with training-set statistics only. Output goes through a sigmoid, so
  predictions stay in [0, 1].

## Baseline results (test set, 23 standard drive cycles)

| model | params | RMSE | MAE | max error | −20 °C RMSE | 25 °C RMSE |
|---|---|---|---|---|---|---|
| linear (ridge) | 10 | 6.19% | 4.30% | 39.7% | 13.8% | 3.7% |
| FNN 64-64-32 | 6.9k | **1.90 ± 0.04%** | 1.39 ± 0.04% | 18.5 ± 0.5% | 3.31 ± 0.06% | 1.55 ± 0.13% |
| LSTM 2×64 | 51k | not yet run (~90 s/epoch on CPU, so use a GPU or Colab) | | | | |

Linear: single run (seed 42). FNN: mean ± std over 5 seeds (1, 2, 3, 4, 42); per-seed results in
`outputs/fnn/seed*/` and `outputs/fnn/seed_summary.csv` (`python scripts/summarize_seeds.py outputs/fnn`).

**What the diagnosis notebooks show** (`notebooks/neg_temp_analysis.ipynb` for −20/−10/0 °C,
`notebooks/pos_temp_analysis.ipynb` for 10/25/40 °C):
* **Cold (−20 °C)**: most of the error is in the first ~10 minutes. The cell warms ~2 °C under
  load and its voltage response changes faster than the features capture; the error recovers
  once it has warmed.
* **Warm (10–40 °C)**: error is 1.1–2.5% RMSE per run, mostly a small under-prediction.
* **Near full charge (all temps)**: the FNN under-predicts by ~2–3 pts at 95–100% SOC, on
  training runs too, so it is a fitting limit rather than poor generalization. This is the
  "start-up" error; it lasts until the run drops below ~90% SOC, not a fixed time.
* **Capacity mismatch (all temps)**: every test session's cell capacity is 0.6–2.4% higher than
  its training sessions' (largest: 10 °C US06, +3.2%), so SOC is under-predicted mid-run.
  Relabelling with the training capacity removes most of the bias and ~0.25 pt of RMSE at
  10–40 °C.
* **Final rest period**: the prediction drifts **up** at −20/−10 °C (voltage relaxes 170–480 mV)
  but **down** by 0.6–2.8 pts at 10–40 °C, settling at ~3.5% vs a true 5%.

## Roadmap / ideas for the team

Suggested order. Each item is a good, self-contained PR.

1. **Reproduce the baselines** on each teammate's machine (checks the setup).
2. **Exploratory notebook**: plot V/I/T/SOC for one run per temperature, plus the capacity vs
   temperature table. These make good report figures.
3. **Robustness to sensor noise and offsets**: add Gaussian noise (e.g. ±10 mV, ±50 mA) and a
   constant current offset to training inputs. Real BMS sensors are much worse than this
   lab cycler. Report test error with and without the same noise added at test time.
4. **Harder generalization test**: leave one temperature out (train on 5, test on the 6th),
   or train on 25 °C only and test everywhere. This is a stronger claim than the default split.
5. **Model sweep**: GRU, 1D-CNN, and LSTM sequence length (60/120/300 s); FNN rolling windows
   (try adding 600 s). Keep `configs/` per experiment so results are reproducible.
6. **Physics baseline for comparison**: coulomb counting with a wrong initial SOC, plus an OCV
   lookup built from the `C20DisCh` (C/20 discharge) files. Shows *why* ML is useful.
7. **Smoothing**: a simple low-pass / Kalman filter on the FNN output usually removes most of
   the jitter and cuts max error substantially.
8. **Deployability**: parameter count, inference time per sample, export to ONNX.

## Credit
Data: P. Kollmeyer, M. Naguib, M. Skells, *LG 18650HG2 Li-ion Battery Data*, Mendeley Data,
2020, doi:10.17632/b5mj79w5w9.2 (McMaster University). Related papers:
* C. Vidal et al., "Robust xEV Battery State-of-Charge Estimator Design using Deep Neural
  Networks," SAE 2020-01-1181.
* C. Vidal, P. Kollmeyer, E. Chemali, A. Emadi, "Li-ion Battery State of Charge Estimation
  Using LSTM RNN with Transfer Learning," IEEE ITEC 2019.
