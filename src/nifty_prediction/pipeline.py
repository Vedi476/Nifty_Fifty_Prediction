"""
Run directly, e.g.:
    python -m nifty_prediction.pipeline --synthetic
    python -m nifty_prediction.pipeline --csv path/to/NIFTY_INTRADAY_2026.csv --window-size 1875
"""

import argparse

import numpy as np
import pandas as pd

from .data import load_prices
from .windowing import make_windows
from .split import train_test_split_by_time
from .normalize import compute_norm_stats, apply_norm, undo_norm
from .model import LinearWindowModel
from .evaluate import evaluate


def run_pipeline(csv_path, window_size=1875, train_fraction=0.8,
                  epochs=100, lr=1e-5, price_column="close", date_column="date",
                  verbose=False):
    prices = load_prices(csv_path, price_column, date_column)
    X, Y = make_windows(prices, window_size)
    X_train, X_test, Y_train, Y_test = train_test_split_by_time(X, Y, train_fraction)

    x_mean, x_std = compute_norm_stats(X_train)
    y_mean, y_std = compute_norm_stats(Y_train)
    X_train_n = apply_norm(X_train, x_mean, x_std)
    X_test_n = apply_norm(X_test, x_mean, x_std)
    Y_train_n = apply_norm(Y_train, y_mean, y_std)

    model = LinearWindowModel(window_size, window_size)
    loss_history = model.train(X_train_n, Y_train_n, epochs=epochs, lr=lr, verbose=verbose)

    Y_pred_n = model.forward(X_test_n)
    Y_pred = undo_norm(Y_pred_n, y_mean, y_std)

    metrics = evaluate(Y_test, Y_pred)

    return {
        "model": model,
        "loss_history": loss_history,
        "Y_test": Y_test,
        "Y_pred": Y_pred,
        "metrics": metrics,
    }


def make_synthetic_csv(path, n_rows=20000, seed=0):
    rng = np.random.default_rng(seed)
    dates = pd.date_range("2020-01-01", periods=n_rows, freq="min")
    steps = rng.normal(loc=0.0, scale=1.0, size=n_rows)
    prices = 20000 + np.cumsum(steps)
    pd.DataFrame({"date": dates, "close": prices}).to_csv(path, index=False)
    return path


def main():
    parser = argparse.ArgumentParser(description="NIFTY intraday price prediction pipeline.")
    parser.add_argument("--csv", type=str, default=None, help="Path to a price CSV file")
    parser.add_argument("--window-size", type=int, default=1875)
    parser.add_argument("--train-fraction", type=float, default=0.8)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--lr", type=float, default=1e-5)
    parser.add_argument("--synthetic", action="store_true", help="Generate and use fake demo data")
    args = parser.parse_args()

    csv_path = args.csv
    if args.synthetic:
        csv_path = make_synthetic_csv("synthetic_nifty.csv")

    if not csv_path:
        print("Provide --csv path/to/file.csv, or use --synthetic to try it on fake data.")
        return

    result = run_pipeline(
        csv_path,
        window_size=args.window_size,
        train_fraction=args.train_fraction,
        epochs=args.epochs,
        lr=args.lr,
    )

    print(f"Loss: {result['loss_history'][0]:.4f} -> {result['loss_history'][-1]:.4f}")
    print("Test metrics:")
    for name, value in result["metrics"].items():
        print(f"  {name}: {value:.4f}")


if __name__ == "__main__":
    main()
