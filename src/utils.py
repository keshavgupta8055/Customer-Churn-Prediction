"""
utils.py
--------
Shared helper functions used across the project.
"""

import time
import matplotlib.pyplot as plt
from pathlib import Path

from src.config import FIGURES_DIR


def print_section(title: str, width: int = 60) -> None:
    """
    Print a formatted section header to the console.

    Parameters
    ----------
    title : str
        Section title.
    width : int
        Total line width.
    """
    border = "=" * width
    print(f"\n{border}")
    print(f"  {title}")
    print(f"{border}")


def save_figure(filename: str, dpi: int = 150) -> None:
    """
    Save the current matplotlib figure to the outputs/figures/ directory
    and close it to free memory.

    Parameters
    ----------
    filename : str
        Output filename (e.g. 'confusion_matrix_RF.png').
    dpi : int
        Resolution.
    """
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    path = FIGURES_DIR / filename
    plt.savefig(path, dpi=dpi, bbox_inches="tight")
    plt.close()


def timer(func):
    """
    Decorator that prints the wall-clock runtime of a function.

    Usage
    -----
    @timer
    def my_function():
        ...
    """
    def wrapper(*args, **kwargs):
        start = time.time()
        result = func(*args, **kwargs)
        elapsed = time.time() - start
        mins, secs = divmod(elapsed, 60)
        print(f"  [time] {func.__name__} completed in {int(mins)}m {secs:.1f}s")
        return result
    return wrapper


def ensure_dirs(paths: list) -> None:
    """
    Create a list of directories if they do not already exist.

    Parameters
    ----------
    paths : list of pathlib.Path
    """
    for p in paths:
        Path(p).mkdir(parents=True, exist_ok=True)


def get_best_model_name(metrics: dict, metric: str = "roc_auc") -> str:
    """
    Return the name of the model with the highest score on a given metric.

    Parameters
    ----------
    metrics : dict
        {model_name: metrics_dict} as returned by evaluate_all_models.
    metric : str
        Metric key to rank by.

    Returns
    -------
    str
        Name of the best model.
    """
    return max(metrics, key=lambda name: metrics[name][metric])
