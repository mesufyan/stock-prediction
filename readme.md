STOCK-PREDICTION  -  Project README

PURPOSE
-------
A learning project (not a trading tool) built to practise the TOOLING side of
ML/DL work: WSL, virtual environments, running .py scripts from the terminal,
the full fetch -> features -> split -> train -> deploy pipeline, and a Streamlit
web app.

Task: predict NVDA's 5-day FORWARD return (regression) from price/volume-derived
features.

KEY FINDING: No model beat the naive "predict a constant" baseline. This is a
CORRECT result about market efficiency, not a failure - NVDA's 5-day returns are
essentially unpredictable from these features. The more flexible the model
(GradientBoosting), the WORSE it did, which is the classic signature of
overfitting noise when no signal exists.


ENVIRONMENT
-----------
- Runs inside WSL 2 (Ubuntu) on Windows, CPU-only (no NVIDIA GPU).
- Python virtual environment in ./venv  (rebuild with requirements.txt; never
  copy the venv folder itself - it is disposable and path-specific).
- Activate:  source venv/bin/activate
- Rebuild :  python3 -m venv venv && source venv/bin/activate
             && pip install -r requirements.txt


FOLDER LAYOUT
-------------
stock-prediction/
├── data/            raw + processed CSVs
├── src/             all pipeline scripts
├── models/          saved model artifacts (.pkl)
├── outputs/         saved plots + results
├── notebooks/       (empty - kept for exploration)
├── venv/            virtual environment (do NOT copy/commit)
├── app.py           Streamlit web app
└── requirements.txt pinned dependency list (the reproducibility recipe)


DATA FILES (data/)
------------------
raw_prices.csv    Raw NVDA OHLCV, 2014-2024, from yfinance (auto_adjust=True so
                  prices are split/dividend adjusted). Cleaned at source to fix
                  yfinance's multi-level header quirk. ~2518 rows.

features.csv      Model-ready table: 8 features + the target, warm-up and
                  tail NaN rows dropped. ~2491 rows.
                  Columns:
                    ret_1d      - 1-day return (today vs yesterday)
                    ret_5d      - 5-day trailing return
                    ret_lag1    - return 2 days ago (ret_1d shifted 1)
                    ma10_ratio  - Close / 10-day moving average
                    ma20_ratio  - Close / 20-day moving average
                    vol_10d     - std of daily returns over 10 days
                    vol_20d     - std of daily returns over 20 days
                    vol_ratio   - Volume / 20-day average volume
                    target_5d   - 5-day FORWARD return (the label)


SCRIPTS (src/)  -  run in pipeline order
-----------------------------------------
1. fetch_data.py
   Downloads NVDA via yfinance, flattens yfinance's multi-level column header
   at the source, saves clean data/raw_prices.csv.
   Run: python src/fetch_data.py

2. build_features.py
   Reads raw_prices.csv, engineers the 8 backward-looking features, builds the
   5-day forward-return target (shift -5), drops NaN warm-up/tail rows, saves
   data/features.csv. All features use only past/present data (no leakage);
   only the target looks forward.
   Run: python src/build_features.py

3. split_baseline.py
   Chronological train/test split (80/20, NO shuffling) with a 5-day EMBARGO gap
   at the boundary so overlapping 5-day windows don't leak train->test. Prints
   the naive baselines ("predict 0" and "predict train-mean") - the bar every
   model must beat. Train-mean is computed on TRAIN ONLY (no peeking).
   Run: python src/split_baseline.py

4. train_model.py   [EXPERIMENT phase]
   Trains 3 models (LinearRegression, RandomForest, GradientBoosting) on the
   same split, scales features (fit on train only), compares each to the
   baseline, saves a results table to outputs/results.csv, prints the verdict.
   RESULT: no model beat the baseline.
   Run: python src/train_model.py

5. train_and_save.py   [DEPLOYMENT phase]
   Trains the chosen model (LinearRegression - the closest to baseline, least
   overfit) and SAVES the deployable artifacts with joblib:
     models/model.pkl        - the trained model
     models/scaler.pkl       - the FITTED scaler (reused, never refit)
     models/feature_cols.pkl - exact feature names IN ORDER
   The model + preprocessing + feature schema travel together as one artifact.
   Run: python src/train_and_save.py

6. backtest.py   [DEPLOYMENT - honest evaluation]
   LOADS the saved artifacts (no retraining), predicts on the test period,
   compares predicted vs actual 5-day returns. Saves:
     outputs/backtest_timeseries.png  - predicted (near-flat) vs actual (jagged)
     outputs/backtest_scatter.png     - no diagonal structure = no skill
   Showed predicted std ~0.005 vs actual std ~0.077: the model collapses to
   predicting near the mean because it found no signal.
   Run: python src/backtest.py

7. plot_results.py
   Reads outputs/results.csv, saves bar charts of MAE and RMSE for all methods
   with the baseline drawn as a reference line. Uses matplotlib "Agg" backend
   (saves to file, no GUI window - works headless).
   Saves: outputs/mae_comparison.png, outputs/rmse_comparison.png
   Run: python src/plot_results.py

8. predict_live.py   [DEPLOYMENT - Level 2]
   Fetches RECENT NVDA data (past the 2024 training cutoff), rebuilds features
   with the same logic, loads the saved model, outputs a single live
   5-day-ahead prediction on genuinely unseen data.
   Run: python src/predict_live.py


WEB APP (app.py)
----------------
Streamlit interactive app with two modes:
  Mode 1 "Latest data" - fetches most recent NVDA data, computes features,
                         predicts the next 5 days.
  Mode 2 "Past date"   - pick any historical date; the app computes features as
                         of that date, predicts, and reveals what ACTUALLY
                         happened 5 days later (interactive backtest).
Both modes DISPLAY the 8 computed features going into the model, so you can see
and sanity-check the inputs. Uses @st.cache_data / @st.cache_resource so data
and model aren't reloaded on every interaction.
Run: streamlit run app.py   (opens http://localhost:8501 in the browser)


MODELS (models/)
----------------
model.pkl         trained LinearRegression
scaler.pkl        fitted StandardScaler (from training data)
feature_cols.pkl  ordered list of the 8 feature names


OUTPUTS (outputs/)
------------------
results.csv                 MAE/RMSE for baselines + 3 models
mae_comparison.png          bar chart, MAE vs baseline line
rmse_comparison.png         bar chart, RMSE vs baseline line
backtest_timeseries.png     predicted vs actual over the test period
backtest_scatter.png        predicted vs actual scatter


KEY LESSONS BAKED INTO THIS PROJECT
-----------------------------------
- Predict RETURNS, not price (price prediction just echoes "today"); price and
  returns are a lossless pair, other features are lossy summaries of price.
- Always build a naive BASELINE first; a raw error number is meaningless without
  it. The baseline is both the bar to beat and a measure of target volatility.
- Time-series hygiene: chronological split (never shuffle), embargo gap for
  overlapping windows, scale/mean fit on TRAIN ONLY.
- More model power != more predictive power when there is no signal.
- Deployment = save model + preprocessing + feature schema together; reload and
  predict without retraining; live features must be built exactly as in training.
- venv is disposable and path-specific; requirements.txt is the portable recipe.


NEXT STEPS (planned, not yet done)
----------------------------------
- Move development into VS Code (WSL extension) + use the debugger.
- argparse: make ticker/dates command-line arguments instead of editing files.
- Git version control (with venv/ in .gitignore).
- Switch target to VOLATILITY (which has real, learnable signal) to get an
  honest model-beats-baseline result - optionally in PyTorch for DL practice.
===============================================================================
