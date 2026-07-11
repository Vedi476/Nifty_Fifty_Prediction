"""NIFTY intraday price prediction — a linear regression model built from scratch."""

from .data import load_prices
from .windowing import make_windows
from .split import train_test_split_by_time
from .normalize import compute_norm_stats, apply_norm, undo_norm
from .model import LinearWindowModel
from .evaluate import evaluate

__all__ = [
    "load_prices",
    "make_windows",
    "train_test_split_by_time",
    "compute_norm_stats",
    "apply_norm",
    "undo_norm",
    "LinearWindowModel",
    "evaluate",
]
