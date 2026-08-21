"""
app.py - Streamlit NVDA 5-day return predictor (learning demo).
Two modes:
  1. Latest data  -> fetch most recent data, predict next 5 days.
  2. Past date    -> pick a date, predict, and compare to what ACTUALLY happened.
Shows the 8 computed features going into the model.
Run:  streamlit run app.py
"""

import streamlit as st
import joblib
import numpy as np
import pandas as pd
import yfinance as yf

MODEL_DIR = "models"
TICKER = "NVDA"

@st.cache_resource
def load_artifacts():
    model        = joblib.load(f"{MODEL_DIR}/model.pkl")
    scaler       = joblib.load(f"{MODEL_DIR}/scaler.pkl")
    feature_cols = joblib.load(f"{MODEL_DIR}/feature_cols.pkl")
    return model, scaler, feature_cols

@st.cache_data
def fetch_prices(start, end=None):
    df = yf.download(TICKER, start=start, end=end, auto_adjust=True)
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    return df

def build_features(df):
    """Must match build_features.py EXACTLY."""
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

model, scaler, feature_cols = load_artifacts()

st.title("NVDA 5-Day Return Predictor")
st.caption("Learning demo. This model has ~no real predictive power - it collapses "
           "to predicting near the mean. Do not trade on it.")

mode = st.radio("Choose mode:",
                ["Latest data (predict next 5 days)",
                 "Past date (predict + compare to reality)"])

# ---------- MODE 1: LATEST ----------
if mode.startswith("Latest"):
    df = fetch_prices(start="2023-06-01")
    feat = build_features(df).dropna()
    latest_date = feat.index[-1]
    row = feat[feature_cols].iloc[[-1]]

    X = scaler.transform(row.values)
    pred = model.predict(X)[0]
    latest_price = df["Close"].loc[latest_date]

    st.subheader("Features computed from the latest real data")
    st.dataframe(row.T.rename(columns={row.index[-1]: "value"}))

    st.subheader("Prediction")
    st.write(f"Most recent date: **{latest_date.date()}**, close **{latest_price:.2f}**")
    st.metric("Predicted 5-day return", f"{pred*100:+.2f}%")
    st.write(f"Implied price in 5 trading days: **{latest_price*(1+pred):.2f}**")

# ---------- MODE 2: PAST DATE ----------
else:
    df = fetch_prices(start="2014-01-01")
    feat_all = build_features(df).dropna()

    min_d, max_d = feat_all.index.min().date(), feat_all.index.max().date()
    picked = st.date_input("Pick a date", value=max_d, min_value=min_d, max_value=max_d)
    picked = pd.Timestamp(picked)

    # nearest available trading day on/before the picked date
    valid = feat_all.index[feat_all.index <= picked]
    if len(valid) == 0:
        st.error("No data on/before that date. Pick a later one.")
        st.stop()
    d = valid[-1]

    row = feat_all[feature_cols].loc[[d]]
    X = scaler.transform(row.values)
    pred = model.predict(X)[0]

    st.subheader(f"Features going into the model (as of {d.date()})")
    st.dataframe(row.T.rename(columns={d: "value"}))

    st.subheader("Prediction vs reality")
    st.metric("Predicted 5-day return", f"{pred*100:+.2f}%")

    # what actually happened: price 5 trading days later
    pos = df.index.get_loc(d)
    if pos + 5 < len(df):
        actual = (df["Close"].iloc[pos+5] / df["Close"].iloc[pos]) - 1
        st.metric("ACTUAL 5-day return", f"{actual*100:+.2f}%",
                  delta=f"{(pred-actual)*100:+.2f}% off")
    else:
        st.info("Not enough future data yet to know the actual outcome.")
