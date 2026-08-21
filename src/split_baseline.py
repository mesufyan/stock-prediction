"""
split_and_baseline.py
Splits features chronologically (with an embargo gap) and computes
naive baselines the real model must beat.
Run:  python src/split_and_baseline.py
"""

import pandas as pd
import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error

INPUT_PATH = "data/features.csv"
HORIZON = 5           # target horizon (days)
TEST_FRACTION = 0.2   # last 20% of time = test set

def main():
    df = pd.read_csv(INPUT_PATH, index_col="Date", parse_dates=True)
    target_col = "target_5d"
    feature_cols = [c for c in df.columns if c != target_col]

    n = len(df)
    split_idx = int(n * (1 - TEST_FRACTION))

    # --- Embargo: drop HORIZON rows at the boundary so the last train
    #     target does not overlap the first test target ---
    train = df.iloc[:split_idx - HORIZON]
    test  = df.iloc[split_idx:]

    print(f"Total rows: {n}")
    print(f"Train: {len(train)} rows ({train.index.min().date()} -> {train.index.max().date()})")
    print(f"Test:  {len(test)} rows ({test.index.min().date()} -> {test.index.max().date()})")
    print(f"Embargo gap: {HORIZON} rows removed at the boundary\n")

    y_train = train[target_col].values
    y_test  = test[target_col].values

    # --- Baseline 1: always predict 0 (no movement) ---
    pred_zero = np.zeros_like(y_test)

    # --- Baseline 2: predict the MEAN 5-day return from the TRAIN set ---
    #     (train mean only -> no peeking at test = no leakage) ---
    train_mean = y_train.mean()
    pred_mean = np.full_like(y_test, train_mean)

    def report(name, y_true, y_pred):
        mae  = mean_absolute_error(y_true, y_pred)
        rmse = np.sqrt(mean_squared_error(y_true, y_pred))
        print(f"{name:22s}  MAE={mae:.5f}   RMSE={rmse:.5f}")

    print("--- Baselines on the TEST set ---")
    report("Predict 0", y_test, pred_zero)
    report("Predict train-mean", y_test, pred_mean)
    print(f"\n(train-mean 5-day return = {train_mean:.5f})")
    print("\nThese are the numbers your real model must beat.")

if __name__ == "__main__":
    main()
