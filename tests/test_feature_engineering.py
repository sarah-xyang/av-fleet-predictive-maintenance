"""
Tests for src/feature_engineering.py

Uses a synthetic 3-unit DataFrame to verify:
  - RUL is computed correctly per unit
  - Binary label is derived correctly from RUL
  - Rolling window features have the expected shape and column names
"""
import pandas as pd
import pytest

from src.feature_engineering import (
    FAILURE_WINDOW,
    add_failure_label,
    add_rolling_features,
    add_rul,
)


@pytest.fixture
def synthetic_df() -> pd.DataFrame:
    """Three units with 10, 15, and 8 cycles respectively."""
    rows = []
    for unit_id, n_cycles in [(1, 10), (2, 15), (3, 8)]:
        for cycle in range(1, n_cycles + 1):
            rows.append(
                {
                    "unit_number": unit_id,
                    "time_in_cycles": cycle,
                    "sensor_1": float(cycle),
                    "sensor_2": float(cycle * 2),
                }
            )
    return pd.DataFrame(rows)


class TestAddRul:
    def test_rul_at_last_cycle_is_zero(self, synthetic_df):
        df = add_rul(synthetic_df)
        last_cycles = df.groupby("unit_number")["time_in_cycles"].max()
        for unit_id, max_cycle in last_cycles.items():
            last_rul = df.loc[
                (df["unit_number"] == unit_id) & (df["time_in_cycles"] == max_cycle),
                "rul",
            ].values[0]
            assert last_rul == 0, f"Unit {unit_id}: expected RUL=0 at last cycle"

    def test_rul_at_first_cycle(self, synthetic_df):
        df = add_rul(synthetic_df)
        # Unit 1 has 10 cycles; first cycle RUL should be 9
        first_rul = df.loc[
            (df["unit_number"] == 1) & (df["time_in_cycles"] == 1), "rul"
        ].values[0]
        assert first_rul == 9

    def test_rul_column_exists(self, synthetic_df):
        df = add_rul(synthetic_df)
        assert "rul" in df.columns


class TestAddFailureLabel:
    def test_label_column_created(self, synthetic_df):
        df = add_rul(synthetic_df)
        df = add_failure_label(df)
        expected_col = f"will_fail_in_{FAILURE_WINDOW}"
        assert expected_col in df.columns

    def test_label_is_binary(self, synthetic_df):
        df = add_rul(synthetic_df)
        df = add_failure_label(df)
        col = f"will_fail_in_{FAILURE_WINDOW}"
        assert set(df[col].unique()).issubset({0, 1})

    def test_label_positive_when_rul_lte_window(self, synthetic_df):
        df = add_rul(synthetic_df)
        df = add_failure_label(df)
        col = f"will_fail_in_{FAILURE_WINDOW}"
        should_be_positive = df["rul"] <= FAILURE_WINDOW
        assert (df.loc[should_be_positive, col] == 1).all()
        assert (df.loc[~should_be_positive, col] == 0).all()


class TestAddRollingFeatures:
    def test_output_shape_unchanged(self, synthetic_df):
        df = add_rolling_features(synthetic_df, sensor_cols=["sensor_1", "sensor_2"])
        assert len(df) == len(synthetic_df)

    def test_rolling_columns_created(self, synthetic_df):
        df = add_rolling_features(
            synthetic_df, sensor_cols=["sensor_1"], window_size=3
        )
        expected = [
            "sensor_1_rolling_mean",
            "sensor_1_rolling_std",
            "sensor_1_rolling_min",
            "sensor_1_rolling_max",
        ]
        for col in expected:
            assert col in df.columns, f"Missing expected column: {col}"

    def test_no_nans_after_rolling(self, synthetic_df):
        df = add_rolling_features(
            synthetic_df, sensor_cols=["sensor_1", "sensor_2"], window_size=3
        )
        rolling_cols = [c for c in df.columns if "rolling" in c]
        assert df[rolling_cols].isna().sum().sum() == 0
