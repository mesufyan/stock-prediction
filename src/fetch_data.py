"""
fetch_data.py
Downloads historical stock data via yfinance and saves it as a clean CSV.
Run:  python src/fetch_data.py
"""

import yfinance as yf
import pandas as pd
import os

TICKER = "NVDA"
START = "2014-01-01"
END = "2024-01-01"
OUTPUT_PATH = "data/raw_prices.csv"

def main():
    print(f"Fetching {TICKER} from {START} to {END} ...")
    df = yf.download(TICKER, start=START, end=END, auto_adjust=True)

    if df.empty:
        print("ERROR: No data returned. Check the ticker or your internet.")
        return

    # --- CLEAN: flatten yfinance's multi-level column header ---
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)   # keep 'Close','High',... drop the ticker level
    df.index.name = "Date"

    os.makedirs(os.path.dirname(OUTPUT_PATH), exist_ok=True)
    df.to_csv(OUTPUT_PATH)

    print(f"Saved {len(df)} rows to {OUTPUT_PATH}")
    print(df.head())

if __name__ == "__main__":
    main()
