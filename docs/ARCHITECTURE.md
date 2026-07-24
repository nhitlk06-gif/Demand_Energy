# Architecture

```
demanddata_2020..2025.csv (raw, half-hourly, NESO)
            │
            ▼
   demandforecast.cleaning
   - merge 6 years, parse dates
   - resync DST days to 48 periods/day
   - drop SCOTTISH_TRANSFER, interpolate gaps/anomalies
   - safety assertions
            │
            ▼
  data/processed/electricity_cleaned.csv  (105,216 rows × 22 cols)
            │
            ▼
   demandforecast.features
   - lag features (H = 12,24,36,48,336) × {ND, TSD, ENGLAND_WALES_DEMAND}
     NOTE: H=1,2 are intentionally excluded — see "Why lag ≥ 12?" below
   - rolling mean/std (windows 8/24/48), leakage-safe (shift(1) first)
   - cyclical sin/cos (period-of-day, day-of-week) + IS_WEEKEND
   - trim first 336 rows, drop admin columns
   - mutual-information feature selection (train+valid only, threshold 0.01)
            │
            ▼
  data/processed/electricity_features_fixed.csv (104,880 rows × 25 cols)
            │
            ▼
   demandforecast.splits          (chronological 2020-23 / 2024 / 2025)
            │
            ▼
   demandforecast.train
   - SNaive Daily / Weekly (no fitting — read a lag column)
   - Linear Regression
   - Random Forest
   - XGBoost (early stopping on 2024 valid)
   - LightGBM (early stopping on 2024 valid)
   - XGBoost (Tuned) / LightGBM (Tuned) — random-search hyperparameters
            │
            ▼
  models/*.pkl + models/metrics_summary.{csv,json} + models/feature_columns.json
            │
            ├─────────────────────────────┐
            ▼                             ▼
   app/api.py (FastAPI)          app/streamlit_app.py (dashboard)
   - /forecast, /models,          - Tab 1: talks directly to models/ on disk,
     /history, /backtest,           or to the API via DEMANDFORECAST_API_URL
     /evaluate                    - Tab 2: actual-vs-predicted comparison
            │                       chart against the real demand curve
            ▼
   demandforecast.forecast
   - forecast_horizon(): recursive multi-step forecast beyond the last
     observed period — each prediction is fed back in to build the next
     step's lag/rolling features; exogenous raw columns fall back to
     same-period-yesterday persistence.
   - backtest_predictions(): ordinary single-step prediction directly on
     real historical rows (train/valid/test split) — actual vs. predicted,
     side by side, for a fair "does the model track the real curve?" check.
```

## Why minimum lag = 12 (this project's key difference from `energy_england`)

The sibling project (`energy_england`) uses lags `{1, 2, 12, 24, 36, 48, 336}`.
Because `ND_LAG_1` is simply the value from 30 minutes ago and electricity
demand barely moves period-to-period, a model with access to `ND_LAG_1` can
score extremely well (R² ≈ 0.997) largely by *copying the last observation*
— which is a fine "1-step-ahead nowcast" but not a genuine test of a model's
ability to anticipate demand.

This project drops `LAG_1` and `LAG_2` so the **minimum available lag is 12
periods (6 hours)**. This is a materially harder and more realistic task:
the best model (LightGBM, tuned) reaches R² ≈ 0.960, RMSE ≈ 1,241 MW,
MAPE ≈ 3.59% on the fully held-out 2025 test year — still very strong, but
now the accuracy genuinely reflects the model's ability to use daily/weekly
seasonality and exogenous columns (embedded wind/solar generation,
interconnector flows) to anticipate demand 6 hours out, rather than reading
the answer off the previous settlement period.

An interesting side effect (see `paper/main.tex` and notebook 04 for the
full analysis): plain **Linear Regression actually underperforms the
SNaive-Daily baseline** on MAE/MAPE once the near-lags are removed — direct
evidence that the demand ⇄ feature relationship is meaningfully non-linear,
which is masked when `LAG_1`/`LAG_2` are available.

## Why two kinds of evaluation (forecast vs. backtest)?

- **Forward forecast** (`forecast_horizon`, dashboard Tab 1): recursive
  multi-step forecasting *beyond* the last observed settlement period. This
  is what you'd use operationally, but there's no ground truth yet to
  compare against, and errors can compound over long horizons.
- **Backtest** (`backtest_predictions`, dashboard Tab 2): ordinary
  single-step prediction on a *real, already-observed* chronological split
  (train / valid / held-out test). This is what directly answers "how
  closely does the model's predicted curve track the real demand curve?" —
  the comparison chart plots `actual_mw` against `predicted_mw` for the
  same timestamps, with no recursive feedback loop involved.
