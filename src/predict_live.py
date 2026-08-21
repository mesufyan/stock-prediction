"""
predict_live.py
Fetches RECENT NVDA data (past the training cutoff), rebuilds features,
loads the saved model, and outputs a live 5-day-ahead prediction.
Run:  python src/predict_live.py
"""

import joblib
import yfinance as yf
import pandas as pd
import numpy as np

MODEL_DIR = "models"
TICKER = "NVDA"
START = "2023-06-01"   # enough history to compute rolling features
# no END -> fetch up to today

def build_features(df):
    """Same feature logic as build_features.py (must match exactly)."""
    price = df["Close"]
    feat = pd.DataFrame(index=df.index)
    feat["ret_1d"]     = price.pct_change(1)
    feat["ret_5d"]     = price.pct_change(5)
    feat["ret_lag1"]   = feat["ret_1d"].shift(1)
    feat["ma10_ratio"] = price / price.rolling(10).mean()
    feat["ma20_ratio"] = price / price.rolling(20).mean()
    feat["vol_10d"]    = feat["ret_1d"].rolling(10).std()
    feat["vol_20d"]    = feat["ret_1d"].rolling(20).std()
    feat["vol_ratio"]  = df["Volume"] / df["Volume"].rolling(20).mean()
    return feat

def main():
    # --- load deployed artifacts ---
    model        = joblib.load(f"{MODEL_DIR}/model.pkl")
    scaler       = joblib.load(f"{MODEL_DIR}/scaler.pkl")
    feature_cols = joblib.load(f"{MODEL_DIR}/feature_cols.pkl")
    print("Loaded model, scaler, feature list.\n")

    # --- fetch recent data ---
    print(f"Fetching recent {TICKER} data from {START} to today ...")
    df = yf.download(TICKER, start=START, auto_adjust=True)
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)

    # --- build features, take the MOST RECENT complete row ---
    feat = build_features(df).dropna()
    latest_date = feat.index[-1]
    latest_row = feat[feature_cols].iloc[[-1]]   # keep as DataFrame, correct column order

    # --- scale with the SAVED scaler, predict ---
    X = scaler.transform(latest_row.values)
    pred = model.predict(X)[0]

    latest_price = df["Close"].loc[latest_date]
    print(f"\nMost recent data date: {latest_date.date()}")
    print(f"Latest close price:    {latest_price:.4f}")
    print(f"\nPredicted 5-day forward return: {pred:+.4f}  ({pred*100:+.2f}%)")
    print(f"Implied price in 5 trading days: {latest_price * (1+pred):.4f}")
    print("\n(Reminder: your backtest showed this model has no real predictive")
    print(" power - predictions huddle near the mean. This is a learning exercise.)")

if __name__ == "__main__":
    main()
