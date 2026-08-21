"""
backtest.py
Loads the SAVED model (no retraining), predicts on the test period,
and plots predicted vs actual 5-day returns to see what really happens.
Run:  python src/backtest.py
"""

import joblib
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error

INPUT_PATH = "data/features.csv"
MODEL_DIR = "models"
HORIZON = 5
TEST_FRACTION = 0.2

def main():
    # --- LOAD the deployed artifacts (no training happens here!) ---
    model        = joblib.load(f"{MODEL_DIR}/model.pkl")
    scaler       = joblib.load(f"{MODEL_DIR}/scaler.pkl")
    feature_cols = joblib.load(f"{MODEL_DIR}/feature_cols.pkl")
    print("Loaded model, scaler, feature list from disk (no retraining).")

    df = pd.read_csv(INPUT_PATH, index_col="Date", parse_dates=True)
    target_col = "target_5d"

    # rebuild the SAME test split
    n = len(df)
    split_idx = int(n * (1 - TEST_FRACTION))
    test = df.iloc[split_idx:]

    # use saved feature_cols IN ORDER, then scale with the SAVED scaler
    X_test = test[feature_cols].values
    X_test_s = scaler.transform(X_test)        # reuse fitted scaler, do NOT refit
    y_true = test[target_col].values

    # predict with the loaded model
    y_pred = model.predict(X_test_s)

    mae = mean_absolute_error(y_true, y_pred)
    print(f"Test MAE = {mae:.5f}")
    print(f"Actual returns  -> mean {y_true.mean():+.4f}, std {y_true.std():.4f}, "
          f"range [{y_true.min():+.3f}, {y_true.max():+.3f}]")
    print(f"Predicted returns -> mean {y_pred.mean():+.4f}, std {y_pred.std():.4f}, "
          f"range [{y_pred.min():+.3f}, {y_pred.max():+.3f}]")

    # --- Plot 1: predicted vs actual over time ---
    plt.figure(figsize=(12, 5))
    plt.plot(test.index, y_true, label="Actual 5-day return", linewidth=1, alpha=0.8)
    plt.plot(test.index, y_pred, label="Predicted", linewidth=1.5)
    plt.axhline(0, color="gray", linestyle=":", linewidth=1)
    plt.title("Predicted vs Actual 5-day return (NVDA, test period)")
    plt.ylabel("5-day return")
    plt.legend()
    plt.tight_layout()
    plt.savefig("outputs/backtest_timeseries.png", dpi=120)
    plt.close()
    print("Saved outputs/backtest_timeseries.png")

    # --- Plot 2: scatter predicted vs actual (perfect model = diagonal line) ---
    plt.figure(figsize=(6, 6))
    plt.scatter(y_true, y_pred, alpha=0.4, s=15)
    lims = [min(y_true.min(), y_pred.min()), max(y_true.max(), y_pred.max())]
    plt.plot(lims, lims, "r--", label="perfect prediction")
    plt.xlabel("Actual 5-day return")
    plt.ylabel("Predicted 5-day return")
    plt.title("Predicted vs Actual (scatter)")
    plt.legend()
    plt.tight_layout()
    plt.savefig("outputs/backtest_scatter.png", dpi=120)
    plt.close()
    print("Saved outputs/backtest_scatter.png")

if __name__ == "__main__":
    main()
