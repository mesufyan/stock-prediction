
import pandas as pd
import numpy as np

INPUT_PATH = "data/raw_prices.csv"
OUTPUT_PATH = "data/features.csv"
HORIZON = 5   # predict return over the next 5 trading days

def main():
    # Read CSV; treat the Date column as the index and parse it as dates
    df = pd.read_csv(INPUT_PATH, index_col="Date", parse_dates=True)
    print(f"Loaded {len(df)} rows from {INPUT_PATH}")

    # 'Close' is already split/dividend-adjusted (auto_adjust=True), so use it directly.
    price = df["Close"]

    feat = pd.DataFrame(index=df.index)

    # --- Return features (backward-looking: use only past/present) ---
    feat["ret_1d"]  = price.pct_change(1)              # yesterday->today return
    feat["ret_5d"]  = price.pct_change(5)              # 5-day trailing return
    feat["ret_lag1"] = feat["ret_1d"].shift(1)         # return one more day back

    # --- Moving averages (as ratios to price, so they're scale-free) ---
    feat["ma10_ratio"] = price / price.rolling(10).mean()
    feat["ma20_ratio"] = price / price.rolling(20).mean()

    # --- Volatility: std of daily returns over trailing windows ---
    feat["vol_10d"] = feat["ret_1d"].rolling(10).std()
    feat["vol_20d"] = feat["ret_1d"].rolling(20).std()

    # --- Volume feature: today's volume vs its 20-day average ---
    feat["vol_ratio"] = df["Volume"] / df["Volume"].rolling(20).mean()

    # --- TARGET: 5-day FORWARD return (looks ahead -> shift negative) ---
    feat["target_5d"] = price.pct_change(HORIZON).shift(-HORIZON)

    # --- Drop rows with NaN (warm-up at start, missing target at end) ---
    before = len(feat)
    feat = feat.dropna()
    after = len(feat)
    print(f"Dropped {before - after} rows with NaN (warm-up + last {HORIZON} days)")

    feat.to_csv(OUTPUT_PATH)
    print(f"Saved {after} rows, {feat.shape[1]} columns to {OUTPUT_PATH}")
    print("\nColumns:", list(feat.columns))
    print("\nFirst rows:")
    print(feat.head())

if __name__ == "__main__":
    main()
