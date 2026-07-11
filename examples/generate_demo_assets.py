"""
Generates the demo plots and result numbers used in the project README.
Uses synthetic random-walk price data

Run: python examples/generate_demo_assets.py
"""

import os
import sys

import matplotlib.pyplot as plt

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from nifty_prediction.pipeline import run_pipeline, make_synthetic_csv

ASSETS_DIR = os.path.join(os.path.dirname(__file__), "..", "assets")
os.makedirs(ASSETS_DIR, exist_ok=True)

csv_path = make_synthetic_csv("/tmp/synthetic_nifty_demo.csv", n_rows=20000, seed=0)

result = run_pipeline(
    csv_path,
    window_size=50,
    train_fraction=0.8,
    epochs=200,
    lr=1e-3,
    verbose=False,
)

# --- Loss curve ---
plt.figure(figsize=(7, 4))
plt.plot(result["loss_history"])
plt.xlabel("Epoch")
plt.ylabel("MSE loss (normalized scale)")
plt.title("Training loss over time")
plt.tight_layout()
plt.savefig(os.path.join(ASSETS_DIR, "loss_curve.png"), dpi=120)
plt.close()

# --- Predicted vs actual, first test window ---
plt.figure(figsize=(9, 4))
plt.plot(result["Y_test"][0], label="Actual", linewidth=2)
plt.plot(result["Y_pred"][0], label="Predicted", linestyle="--")
plt.xlabel("Minutes into the window")
plt.ylabel("Price")
plt.title("Predicted vs actual price — first test window")
plt.legend()
plt.tight_layout()
plt.savefig(os.path.join(ASSETS_DIR, "predictions_vs_actual.png"), dpi=120)
plt.close()

print("Metrics:")
for name, value in result["metrics"].items():
    print(f"  {name}: {value:.4f}")

print(f"\nSaved plots to {ASSETS_DIR}/")
