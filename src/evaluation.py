"""
evaluation.py — Classification metrics and decision-threshold analysis.

Provides tools to:
  - Generate classification reports at a given threshold
  - Plot precision-recall curves and identify the recall >= 0.80 operating point
  - Compute confusion matrices
  - Return threshold-keyed metrics for threshold sweep analysis
"""
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    precision_recall_curve,
)

TARGET_RECALL = 0.80  # minimum acceptable recall for the "will fail" class


def evaluate_at_threshold(
    y_true: np.ndarray | pd.Series,
    y_proba: np.ndarray,
    threshold: float = 0.5,
) -> dict:
    """Return a metrics dict for predictions at a given probability threshold.

    Args:
        y_true: Ground-truth binary labels.
        y_proba: Predicted probabilities for the positive class.
        threshold: Decision threshold (default 0.5).

    Returns:
        Dict with keys: threshold, precision, recall, f1, confusion_matrix.
    """
    # TODO: implement
    raise NotImplementedError


def find_recall_threshold(
    y_true: np.ndarray | pd.Series,
    y_proba: np.ndarray,
    min_recall: float = TARGET_RECALL,
) -> float:
    """Find the lowest threshold that achieves at least min_recall.

    Args:
        y_true: Ground-truth binary labels.
        y_proba: Predicted probabilities for the positive class.
        min_recall: Minimum acceptable recall value.

    Returns:
        Threshold float value.
    """
    # TODO: implement
    raise NotImplementedError


def plot_precision_recall(
    y_true: np.ndarray | pd.Series,
    y_proba: np.ndarray,
    save_path: str | None = None,
) -> plt.Figure:
    """Plot a precision-recall curve with the operating point annotated.

    Args:
        y_true: Ground-truth binary labels.
        y_proba: Predicted probabilities for the positive class.
        save_path: If provided, save the figure to this path.

    Returns:
        matplotlib Figure object.
    """
    # TODO: implement
    raise NotImplementedError
