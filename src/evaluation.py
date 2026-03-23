"""
evaluation.py — Regression metrics and cost-driven alert threshold optimisation.

Two responsibilities:
  1. calculate_metrics     — standard regression error metrics for RUL prediction
  2. optimize_alert_threshold — sweep prediction thresholds and find the one that
                                minimises total fleet maintenance cost
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.metrics import (
    confusion_matrix,
    f1_score,
    mean_absolute_error,
    mean_squared_error,
    precision_score,
    r2_score,
    recall_score,
)

# Business alert window: vehicles with true RUL <= this are "at risk".
ALERT_WINDOW: int = 7   # cycles — matches the 7-cycle business definition


def calculate_metrics(
    y_true: np.ndarray | pd.Series,
    y_pred: np.ndarray | pd.Series,
) -> dict[str, float]:
    """Compute RMSE, MAE, and R² for a set of RUL predictions.

    Args:
        y_true: Ground-truth RUL values (continuous).
        y_pred: Model-predicted RUL values (continuous).

    Returns:
        Dictionary with keys:

        * ``rmse`` — Root Mean Squared Error in cycles. Penalises large
          errors more heavily than MAE; useful for catching catastrophic
          mispredictions close to failure.
        * ``mae``  — Mean Absolute Error in cycles. Average per-prediction
          error; easy to explain to non-technical stakeholders.
        * ``r2``   — Coefficient of determination. 1.0 = perfect; 0.0 =
          no better than predicting the mean; negative = worse than mean.

        All values are rounded to 4 decimal places.

    Note:
        Business context — RMSE is the primary metric for this project
        because errors near the failure point are disproportionately
        costly: a 10-cycle prediction error when the true RUL is 5
        means a vehicle either breaks down on-route (false sense of
        safety) or gets pulled prematurely (wasted capacity). RMSE's
        squared penalty makes it sensitive to exactly these high-stakes
        near-failure mispredictions.
    """
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)

    rmse = float(np.sqrt(mean_squared_error(y_true, y_pred)))
    mae  = float(mean_absolute_error(y_true, y_pred))
    r2   = float(r2_score(y_true, y_pred))

    return {
        "rmse": round(rmse, 4),
        "mae":  round(mae,  4),
        "r2":   round(r2,   4),
    }


def optimize_alert_threshold(
    y_true_rul: np.ndarray | pd.Series,
    y_pred_rul: np.ndarray | pd.Series,
    cost_fn: float = 2400.0,
    cost_fp: float = 400.0,
    threshold_range: range = range(5, 16),
    alert_window: int = ALERT_WINDOW,
) -> dict:
    """Find the alert threshold that minimises total fleet maintenance cost.

    Converts continuous RUL predictions into binary maintenance alerts
    ("flag this vehicle") by comparing predicted RUL to a threshold.
    The true label is fixed: a vehicle is genuinely at risk if its true
    RUL is within ``alert_window`` cycles. The threshold only affects
    which predictions we act on.

    For each candidate threshold *t* in ``threshold_range``:

    * ``y_pred_binary = (y_pred_rul <= t)``
    * ``y_true_binary = (y_true_rul <= alert_window)``
    * ``total_cost = FN × cost_fn + FP × cost_fp``

    Args:
        y_true_rul: Ground-truth RUL values (continuous).
        y_pred_rul: Model-predicted RUL values (continuous).
        cost_fn: Cost per false negative (missed failure). Defaults to
            $2,400 — lost revenue plus emergency repair premium.
        cost_fp: Cost per false positive (unnecessary maintenance).
            Defaults to $400 — scheduled labour and parts.
        threshold_range: Iterable of integer thresholds to evaluate.
            Defaults to ``range(5, 16)``.
        alert_window: Cycles within which a vehicle is considered
            genuinely at risk. Defaults to 7.

    Returns:
        Dictionary with keys:

        * ``optimal_threshold`` (int) — threshold with lowest total cost.
        * ``optimal_cost`` (float) — fleet cost at the optimal threshold.
        * ``results_df`` (DataFrame) — one row per threshold with columns
          ``threshold``, ``precision``, ``recall``, ``f1``,
          ``tp``, ``fp``, ``fn``, ``tn``, ``total_cost``.

    Note:
        Business context — the naive threshold of 7 cycles (matching the
        alert window exactly) ignores the asymmetric cost structure.
        A missed failure costs 6× more than a false alarm, so a more
        aggressive threshold — flagging vehicles slightly earlier than
        strictly necessary — may produce fewer false negatives and a
        lower total fleet bill even though it triggers more unnecessary
        maintenance events. This function makes that trade-off explicit
        and quantifiable.
    """
    y_true_rul = np.asarray(y_true_rul, dtype=float)
    y_pred_rul = np.asarray(y_pred_rul, dtype=float)
    y_true_bin = (y_true_rul <= alert_window).astype(int)

    rows: list[dict] = []
    for thresh in threshold_range:
        y_pred_bin = (y_pred_rul <= thresh).astype(int)

        prec = float(precision_score(y_true_bin, y_pred_bin, zero_division=0))
        rec  = float(recall_score(y_true_bin,    y_pred_bin, zero_division=0))
        f1   = float(f1_score(y_true_bin,         y_pred_bin, zero_division=0))

        tn, fp, fn, tp = confusion_matrix(
            y_true_bin, y_pred_bin, labels=[0, 1]
        ).ravel()

        total_cost = int(fn) * cost_fn + int(fp) * cost_fp

        rows.append({
            "threshold" : int(thresh),
            "precision" : round(prec, 4),
            "recall"    : round(rec,  4),
            "f1"        : round(f1,   4),
            "tp"        : int(tp),
            "fp"        : int(fp),
            "fn"        : int(fn),
            "tn"        : int(tn),
            "total_cost": total_cost,
        })

    results_df = pd.DataFrame(rows)
    best_row   = results_df.loc[results_df["total_cost"].idxmin()]

    return {
        "optimal_threshold": int(best_row["threshold"]),
        "optimal_cost"     : float(best_row["total_cost"]),
        "results_df"       : results_df,
    }
