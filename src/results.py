"""
plot_results.py
Reads outputs/results.csv and saves bar charts comparing MAE and RMSE
across baselines and models, with the baseline drawn as a reference line.
Run:  python src/plot_results.py
"""

import pandas as pd
import matplotlib
matplotlib.use("Agg")          # <-- no GUI needed; safe on any/headless machine
import matplotlib.pyplot as plt

RESULTS_PATH = "outputs/results.csv"

def main():
    df = pd.read_csv(RESULTS_PATH)

    # the bar to beat = the better (lower) of the two baselines
    baseline_val = df[df["method"].str.contains("Predict")]["MAE"].min()

    # color baselines differently from models
    colors = ["#888888" if "Predict" in m else "#2c7fb8" for m in df["method"]]

    for metric in ["MAE", "RMSE"]:
        plt.figure(figsize=(8, 5))
        bars = plt.bar(df["method"], df[metric], color=colors)

        # reference line at the baseline value for this metric
        base = df[df["method"].str.contains("Predict")][metric].min()
        plt.axhline(base, color="red", linestyle="--", linewidth=1,
                    label=f"Baseline {metric} = {base:.4f}")

        plt.title(f"{metric}: models vs baselines (5-day return, NVDA)")
        plt.ylabel(metric)
        plt.xticks(rotation=30, ha="right")
        plt.legend()
        plt.tight_layout()

        out = f"outputs/{metric.lower()}_comparison.png"
        plt.savefig(out, dpi=120)
        plt.close()
        print(f"Saved {out}")

if __name__ == "__main__":
    main()
