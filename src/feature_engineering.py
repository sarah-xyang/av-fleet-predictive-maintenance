"""
feature_engineering.py — Rolling window statistics and RUL labeling.

Transforms raw CMAPS telemetry into model-ready features:
  - Computes Remaining Useful Life (RUL) for training data
  - Derives binary label: will vehicle fail within FAILURE_WINDOW cycles?
  - Adds rolling mean/std/min/max for selected sensor channels
"""
import pandas as pd

FAILURE_WINDOW = 7  # operational cycles; matches business prediction horizon


def add_rul(df: pd.DataFrame) -> pd.DataFrame:
    """Add a 'rul' column (Remaining Useful Life) to a training DataFrame.

    RUL is computed per unit as: max_cycle_for_unit - current_cycle.

    Args:
        df: DataFrame with 'unit_number' and 'time_in_cycles' columns.

    Returns:
        DataFrame with new 'rul' column (int).
    """
    # TODO: implement
    raise NotImplementedError


def add_failure_label(df: pd.DataFrame, window: int = FAILURE_WINDOW) -> pd.DataFrame:
    """Add binary label 'will_fail_in_{window}' derived from RUL.

    Label is 1 if rul <= window, else 0.

    Args:
        df: DataFrame with a 'rul' column (from add_rul).
        window: Number of cycles defining the at-risk window.

    Returns:
        DataFrame with new binary label column.
    """
    # TODO: implement
    raise NotImplementedError


def add_rolling_features(
    df: pd.DataFrame,
    sensor_cols: list[str],
    window_size: int = 10,
) -> pd.DataFrame:
    """Add rolling window statistics for specified sensor columns.

    Computes mean, std, min, max over a rolling window per unit.
    NaN values at the start of each unit's history are forward-filled.

    Args:
        df: DataFrame sorted by unit_number, time_in_cycles.
        sensor_cols: List of sensor column names to compute stats for.
        window_size: Number of cycles in the rolling window.

    Returns:
        DataFrame with added columns: {sensor}_rolling_{stat}.
    """
    # TODO: implement
    raise NotImplementedError
