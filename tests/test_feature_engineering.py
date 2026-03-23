"""
tests/test_feature_engineering.py

Unit tests for src/feature_engineering.py and src/evaluation.py.

All tests use pytest-style plain functions (no classes).
Toy DataFrames are used throughout so tests run instantly with no
dependency on the raw data files.
"""
import numpy as np
import pandas as pd
import pytest

from src.evaluation import calculate_metrics
from src.feature_engineering import (
    CONSTANT_SENSORS,
    OP_SETTING_COLS,
    add_rolling_features,
    calculate_rul,
    cap_rul,
    drop_constant_sensors,
)


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def _make_rul_df() -> pd.DataFrame:
    """Two-vehicle DataFrame used by RUL tests.

    Vehicle 1: cycles 1, 2, 3  (max = 3)
    Vehicle 2: cycles 1, 2     (max = 2)

    Expected RUL:
      vehicle 1, cycle 1 → 3 - 1 = 2
      vehicle 1, cycle 2 → 3 - 2 = 1
      vehicle 1, cycle 3 → 3 - 3 = 0  (failure point)
      vehicle 2, cycle 1 → 2 - 1 = 1
      vehicle 2, cycle 2 → 2 - 2 = 0  (failure point)
    """
    return pd.DataFrame({
        "unit_id"       : [1, 1, 1, 2, 2],
        "time_in_cycles": [1, 2, 3, 1, 2],
        "sensor_2"      : [1.0, 2.0, 3.0, 4.0, 5.0],
    })


def _make_sensor_df() -> pd.DataFrame:
    """DataFrame that contains both a constant sensor (sensor_1) and an
    informative sensor (sensor_2), plus all columns that drop_constant_sensors
    expects to be present so it can attempt to drop them."""
    cols = {
        "unit_id"       : [1, 1, 2],
        "time_in_cycles": [1, 2, 1],
        "sensor_2"      : [1.0, 2.0, 3.0],   # informative — must be kept
        "sensor_1"      : [0.0, 0.0, 0.0],   # constant    — must be dropped
        "op_setting_1"  : [100.0, 100.0, 100.0],
    }
    return pd.DataFrame(cols)


# ─────────────────────────────────────────────────────────────────────────────
# Test 1 — calculate_rul: exact RUL values
# ─────────────────────────────────────────────────────────────────────────────

def test_calculate_rul_values() -> None:
    """RUL is computed correctly for each vehicle and each cycle."""
    df     = _make_rul_df()
    result = calculate_rul(df)

    # Vehicle 1, cycle 1: max=3, current=1 → RUL should be 2
    rul_v1_c1 = result.loc[
        (result["unit_id"] == 1) & (result["time_in_cycles"] == 1), "rul"
    ].values[0]
    assert rul_v1_c1 == 2, f"Expected RUL=2, got {rul_v1_c1}"

    # Vehicle 1, cycle 3: max=3, current=3 → RUL should be 0 (failure point)
    rul_v1_c3 = result.loc[
        (result["unit_id"] == 1) & (result["time_in_cycles"] == 3), "rul"
    ].values[0]
    assert rul_v1_c3 == 0, f"Expected RUL=0 at failure point, got {rul_v1_c3}"

    # Vehicle 2, cycle 1: max=2, current=1 → RUL should be 1
    rul_v2_c1 = result.loc[
        (result["unit_id"] == 2) & (result["time_in_cycles"] == 1), "rul"
    ].values[0]
    assert rul_v2_c1 == 1, f"Expected RUL=1, got {rul_v2_c1}"


def test_calculate_rul_does_not_mutate_input() -> None:
    """calculate_rul must not modify the caller's DataFrame."""
    df     = _make_rul_df()
    before = set(df.columns)
    _      = calculate_rul(df)
    assert set(df.columns) == before, "calculate_rul mutated the input DataFrame"


def test_calculate_rul_column_exists() -> None:
    """The 'rul' column is present in the result."""
    result = calculate_rul(_make_rul_df())
    assert "rul" in result.columns


def test_calculate_rul_failure_rows_are_zero() -> None:
    """Every vehicle's last cycle must have RUL == 0."""
    result   = calculate_rul(_make_rul_df())
    last_idx = result.groupby("unit_id")["time_in_cycles"].idxmax()
    assert (result.loc[last_idx, "rul"] == 0).all()


# ─────────────────────────────────────────────────────────────────────────────
# Test 2 — cap_rul: ceiling is enforced, low values are unchanged
# ─────────────────────────────────────────────────────────────────────────────

def test_cap_rul_never_exceeds_cap() -> None:
    """No value in rul_capped should exceed the cap."""
    df = pd.DataFrame({"rul": [50, 100, 125, 150, 200, 300]})
    result = cap_rul(df, cap=125)
    assert result["rul_capped"].max() <= 125


def test_cap_rul_values_below_cap_are_unchanged() -> None:
    """Values that were already at or below the cap must not change."""
    raw = [50, 100, 125, 150, 200, 300]
    df  = pd.DataFrame({"rul": raw})
    result = cap_rul(df, cap=125)
    for orig, capped in zip(raw, result["rul_capped"]):
        if orig <= 125:
            assert capped == orig, (
                f"Value {orig} was already ≤ 125 but was changed to {capped}"
            )


def test_cap_rul_does_not_mutate_input() -> None:
    """cap_rul must not modify the caller's DataFrame."""
    df     = pd.DataFrame({"rul": [100, 200, 300]})
    before = list(df["rul"])
    _      = cap_rul(df, cap=125)
    assert list(df["rul"]) == before, "cap_rul mutated the input DataFrame"


def test_cap_rul_column_added() -> None:
    """The 'rul_capped' column is present in the result."""
    df     = pd.DataFrame({"rul": [10, 200]})
    result = cap_rul(df, cap=125)
    assert "rul_capped" in result.columns


# ─────────────────────────────────────────────────────────────────────────────
# Test 3 — drop_constant_sensors: correct columns removed / retained
# ─────────────────────────────────────────────────────────────────────────────

def test_drop_constant_sensors_removes_correct_columns() -> None:
    """sensor_1 (constant) must be dropped; sensor_2 (informative) must stay."""
    df     = _make_sensor_df()
    result = drop_constant_sensors(df)
    assert "sensor_1"  not in result.columns, "sensor_1 should have been dropped"
    assert "sensor_2"  in result.columns,     "sensor_2 should have been kept"


def test_drop_constant_sensors_removes_op_settings() -> None:
    """Operating-condition columns must also be removed."""
    df     = _make_sensor_df()
    result = drop_constant_sensors(df)
    assert "op_setting_1" not in result.columns


def test_drop_constant_sensors_does_not_mutate_input() -> None:
    """drop_constant_sensors must not modify the caller's DataFrame."""
    df     = _make_sensor_df()
    before = set(df.columns)
    _      = drop_constant_sensors(df)
    assert set(df.columns) == before, "drop_constant_sensors mutated the input"


def test_drop_constant_sensors_safe_on_missing_columns() -> None:
    """Function must not raise if some constant sensors are absent from df."""
    # DataFrame that only has sensor_2 — none of the constant sensors present.
    df = pd.DataFrame({"unit_id": [1], "time_in_cycles": [1], "sensor_2": [1.0]})
    result = drop_constant_sensors(df)  # should not raise
    assert "sensor_2" in result.columns


# ─────────────────────────────────────────────────────────────────────────────
# Test 4 — calculate_metrics: perfect prediction corner case
# ─────────────────────────────────────────────────────────────────────────────

def test_calculate_metrics_perfect_prediction() -> None:
    """When predictions match truth exactly, RMSE=0 and R²=1."""
    y = [10.0, 20.0, 30.0, 40.0, 50.0]
    metrics = calculate_metrics(y_true=y, y_pred=y)
    assert metrics["rmse"] == 0.0,  f"Expected RMSE=0.0, got {metrics['rmse']}"
    assert metrics["r2"]   == 1.0,  f"Expected R²=1.0, got {metrics['r2']}"


def test_calculate_metrics_returns_required_keys() -> None:
    """Return dict must contain rmse, mae, and r2."""
    metrics = calculate_metrics(y_true=[1, 2, 3], y_pred=[1, 2, 4])
    for key in ("rmse", "mae", "r2"):
        assert key in metrics, f"Missing key '{key}' in metrics dict"


def test_calculate_metrics_rmse_nonnegative() -> None:
    """RMSE must always be >= 0."""
    metrics = calculate_metrics(y_true=[10, 20, 30], y_pred=[12, 18, 33])
    assert metrics["rmse"] >= 0.0


def test_calculate_metrics_accepts_numpy_arrays() -> None:
    """calculate_metrics must accept NumPy arrays, not only lists."""
    y_true = np.array([5.0, 10.0, 15.0])
    y_pred = np.array([5.0, 10.0, 15.0])
    metrics = calculate_metrics(y_true, y_pred)
    assert metrics["rmse"] == 0.0


# ─────────────────────────────────────────────────────────────────────────────
# Test 5 — add_rolling_features: shape, column names, no NaNs
# ─────────────────────────────────────────────────────────────────────────────

def test_add_rolling_features_row_count_unchanged() -> None:
    """Row count must be identical before and after rolling features."""
    df     = _make_rul_df()
    result = add_rolling_features(df, sensors=["sensor_2"], window=3)
    assert len(result) == len(df)


def test_add_rolling_features_column_names() -> None:
    """Output must contain rolling_mean_{sensor} and rolling_std_{sensor} columns."""
    df     = _make_rul_df()
    result = add_rolling_features(df, sensors=["sensor_2"], window=3)
    assert "rolling_mean_sensor_2" in result.columns
    assert "rolling_std_sensor_2"  in result.columns


def test_add_rolling_features_drops_raw_sensor() -> None:
    """The original raw sensor column must be absent from the result."""
    df     = _make_rul_df()
    result = add_rolling_features(df, sensors=["sensor_2"], window=3)
    assert "sensor_2" not in result.columns


def test_add_rolling_features_no_nans() -> None:
    """No NaN values should remain in the rolling feature columns."""
    df     = _make_rul_df()
    result = add_rolling_features(df, sensors=["sensor_2"], window=3)
    rolling_cols = [c for c in result.columns if c.startswith("rolling_")]
    assert result[rolling_cols].isna().sum().sum() == 0
