"""
train_model.py
Trains real models on the 5-day return target and compares them
against the naive baselines. Uses the SAME chronological split + embargo.
Run:  python src/train_model.py
"""
import os
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error

INPUT_PATH = "data/features.csv"
HORIZON = 5
TEST_FRACTION = 0.2

def evaluate(name, y_true, y_pred):
    mae  = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    print(f"{name:24s}  MAE={mae:.5f}   RMSE={rmse:.5f}")
    return mae

def main():
    df = pd.read_csv(INPUT_PATH, index_col="Date", parse_dates=True)
    target_col = "target_5d"
    feature_cols = [c for c in df.columns if c != target_col]

    n = len(df)
    split_idx = int(n * (1 - TEST_FRACTION))
    train = df.iloc[:split_idx - HORIZON]   # embargo
    test  = df.iloc[split_idx:]

    X_train, y_train = train[feature_cols].values, train[target_col].values
    X_test,  y_test  = test[feature_cols].values,  test[target_col].values

    # --- Scale: fit on TRAIN ONLY, then apply to test (no leakage) ---
    scaler = StandardScaler().fit(X_train)
    X_train_s = scaler.transform(X_train)
    X_test_s  = scaler.transform(X_test)

        # collect every result into a list of dicts
    rows = []

    def record(name, y_true, y_pred):
        mae  = mean_absolute_error(y_true, y_pred)
        rmse = np.sqrt(mean_squared_error(y_true, y_pred))
        print(f"{name:24s}  MAE={mae:.5f}   RMSE={rmse:.5f}")
        rows.append({"method": name, "MAE": mae, "RMSE": rmse})
        return mae

    print("--- Naive baselines (the bar) ---")
    record("Predict 0", y_test, np.zeros_like(y_test))
    base = record("Predict train-mean", y_test, np.full_like(y_test, y_train.mean()))

    print("\n--- Models ---")
    models = {
        "LinearRegression":  LinearRegression(),
        "RandomForest":      RandomForestRegressor(n_estimators=200, max_depth=5, random_state=42, n_jobs=-1),
        "GradientBoosting":  GradientBoostingRegressor(n_estimators=200, max_depth=3, random_state=42),
    }
    for name, model in models.items():
        model.fit(X_train_s, y_train)
        preds = model.predict(X_test_s)
        record(name, y_test, preds)

    # --- SAVE results ---
    os.makedirs("outputs", exist_ok=True)
    results_df = pd.DataFrame(rows)
    results_df.to_csv("outputs/results.csv", index=False)
    print(f"\nSaved results to outputs/results.csv")

    print("\n--- Verdict ---")
    # only compare the MODEL rows (skip the two baseline rows)
    model_rows = [r for r in rows if "Predict" not in r["method"]]
    best = min(model_rows, key=lambda r: r["MAE"])
    if best["MAE"] < base:
        print(f"{best['method']} beat the baseline by {base - best['MAE']:.5f} MAE.")
    else:
        print(f"No model beat the baseline ({base:.5f}). The signal is weak/absent.")
        print("This is a REAL result about market efficiency, not a bug.")
if __name__ == "__main__":
    main()
