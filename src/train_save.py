"""
train_and_save.py
Trains models, saves the best one + the scaler to disk so they can be
loaded later for prediction WITHOUT retraining.
Run:  python src/train_and_save.py
"""

import os
import joblib
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error

INPUT_PATH = "data/features.csv"
MODEL_DIR = "models"
HORIZON = 5
TEST_FRACTION = 0.2

def main():
    df = pd.read_csv(INPUT_PATH, index_col="Date", parse_dates=True)
    target_col = "target_5d"
    feature_cols = [c for c in df.columns if c != target_col]

    n = len(df)
    split_idx = int(n * (1 - TEST_FRACTION))
    train = df.iloc[:split_idx - HORIZON]
    test  = df.iloc[split_idx:]

    X_train, y_train = train[feature_cols].values, train[target_col].values
    X_test,  y_test  = test[feature_cols].values,  test[target_col].values

    # scale (fit on train only)
    scaler = StandardScaler().fit(X_train)
    X_train_s = scaler.transform(X_train)
    X_test_s  = scaler.transform(X_test)

    # train the simplest model (it was closest to baseline)
    model = LinearRegression().fit(X_train_s, y_train)
    mae = mean_absolute_error(y_test, model.predict(X_test_s))
    print(f"Trained LinearRegression, test MAE = {mae:.5f}")

    # --- SAVE everything needed to predict later ---
    os.makedirs(MODEL_DIR, exist_ok=True)
    joblib.dump(model,  f"{MODEL_DIR}/model.pkl")
    joblib.dump(scaler, f"{MODEL_DIR}/scaler.pkl")
    joblib.dump(feature_cols, f"{MODEL_DIR}/feature_cols.pkl")
    print(f"Saved model, scaler, and feature list to {MODEL_DIR}/")

if __name__ == "__main__":
    main()
