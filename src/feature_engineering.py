"""
feature_engineering.py — Transform raw CMAPS telemetry into model-ready features.

Pipeline stages exposed here:
  1. drop_constant_sensors  — remove uninformative columns identified in EDA
  2. calculate_rul           — label each training row with Remaining Useful Life
  3. cap_rul                 — clip RUL to focus learning on the degradation window
  4. add_rolling_features    — replace raw readings with per-vehicle trend statistics
"""
import pandas as pd

# ── Project constants ──────────────────────────────────────────────────────────
# Sensors confirmed as near-constant across the full FD001 training set (EDA,
# notebook 01). Including them adds noise without adding signal.
CONSTANT_SENSORS: list[str] = [
    "sensor_1", "sensor_5",  "sensor_6",  "sensor_8",
    "sensor_10", "sensor_13", "sensor_15", "sensor_16",
    "sensor_18", "sensor_19",
]

# Operating-condition columns are also near-constant in the FD001
# single-condition subset and carry no degradation information.
OP_SETTING_COLS: list[str] = [
    "op_setting_1", "op_setting_2", "op_setting_3",
]

# Sensors retained after EDA-driven filtering (11 informative channels).
INFORMATIVE_SENSORS: list[str] = [
    "sensor_2",  "sensor_3",  "sensor_4",  "sensor_7",  "sensor_9",
    "sensor_11", "sensor_12", "sensor_14", "sensor_17", "sensor_20",
    "sensor_21",
]

DEFAULT_RUL_CAP:    int = 125   # cycles — degradation signal only meaningful within this window
DEFAULT_WINDOW:     int = 5     # cycles — rolling statistic lookback period


def drop_constant_sensors(df: pd.DataFrame) -> pd.DataFrame:
    """Remove uninformative columns from a CMAPS DataFrame.

    Drops the 10 near-constant sensor channels and the 3 operating-
    condition columns identified during EDA (notebook 01). Only columns
    that are actually present in ``df`` are dropped, so this function is
    safe to call on subsets that may already be missing some columns.

    Args:
        df: Raw CMAPS DataFrame produced by ``load_cmaps_data``.

    Returns:
        Copy of ``df`` with uninformative columns removed. The original
        DataFrame is not modified.

    Note:
        Business context — constant sensors behave like a broken
        dashboard gauge: they always display a number but never change
        regardless of vehicle health. Passing them to a model wastes
        capacity and can introduce spurious correlations. Removing them
        upfront makes the model focus exclusively on channels that
        actually move as a vehicle degrades.
    """
    cols_to_drop = [
        c for c in (CONSTANT_SENSORS + OP_SETTING_COLS)
        if c in df.columns
    ]
    return df.drop(columns=cols_to_drop).copy()


def calculate_rul(df: pd.DataFrame) -> pd.DataFrame:
    """Add a Remaining Useful Life (RUL) column to a training DataFrame.

    For each vehicle (``unit_id``), RUL is computed as::

        RUL = max_cycle_for_this_vehicle - current_time_in_cycles

    The last observed row for each vehicle (its failure cycle) receives
    RUL = 0. Every earlier row receives a positive integer counting down
    to zero.

    Args:
        df: DataFrame containing ``unit_id`` and ``time_in_cycles``
            columns. Intended for the training set, where every vehicle
            is observed to failure.

    Returns:
        Copy of ``df`` with a new integer column ``rul`` appended. The
        original DataFrame is not modified.

    Note:
        Business context — RUL is like an odometer running backwards:
        it starts high when a vehicle is young and counts down to zero
        at the failure point. The model learns to estimate where each
        vehicle sits on that countdown from its current sensor readings,
        enabling the fleet to schedule maintenance before the counter
        reaches zero.
    """
    df = df.copy()
    max_cycles   = df.groupby("unit_id")["time_in_cycles"].transform("max")
    df["rul"]    = max_cycles - df["time_in_cycles"]
    return df


def cap_rul(df: pd.DataFrame, cap: int = DEFAULT_RUL_CAP) -> pd.DataFrame:
    """Clip RUL values at a maximum cap and store the result as ``rul_capped``.

    Early in a vehicle's life the sensor readings are stable and look
    essentially identical regardless of how long the vehicle will
    ultimately operate. The degradation signal only becomes visible
    within the final ``cap`` cycles. Capping prevents the model from
    trying to differentiate between, say, RUL = 300 and RUL = 250 —
    rows that look the same and should be treated the same.

    Args:
        df: DataFrame containing a ``rul`` column (output of
            ``calculate_rul``).
        cap: Maximum RUL value to retain. Defaults to 125 cycles, the
            standard CMAPS FD001 benchmark value.

    Returns:
        Copy of ``df`` with a new column ``rul_capped`` added. The
        original ``rul`` column and the original DataFrame are not
        modified.

    Note:
        Business context — capping is a standard industry technique for
        CMAPS data. It embodies the insight that a vehicle with 300
        cycles left and one with 200 cycles left look identical to
        sensors and should generate the same model output: "healthy,
        no action needed." The model's discrimination budget is spent
        on the last 125 cycles where sensors actually start to drift.
    """
    df             = df.copy()
    df["rul_capped"] = df["rul"].clip(upper=cap)
    return df


def add_rolling_features(
    df: pd.DataFrame,
    sensors: list[str],
    window: int = DEFAULT_WINDOW,
) -> pd.DataFrame:
    """Replace raw sensor columns with per-vehicle rolling statistics.

    For each sensor in ``sensors``, computes a rolling mean and rolling
    standard deviation over the last ``window`` cycles within each
    vehicle's history. The raw sensor columns are then dropped; the
    rolling statistics replace them as the model's input features.

    ``min_periods=1`` ensures that the first few rows of each vehicle
    (where less than ``window`` cycles of history exist) still receive
    a value rather than NaN.

    Args:
        df: DataFrame containing ``unit_id``, ``time_in_cycles``, and
            all columns listed in ``sensors``.
        sensors: List of sensor column names to process (e.g.
            ``['sensor_2', 'sensor_3']``).
        window: Rolling window width in cycles. Defaults to 5.

    Returns:
        Copy of ``df`` sorted by ``(unit_id, time_in_cycles)`` with:

        * ``rolling_mean_{sensor}`` — rolling mean over ``window`` cycles
        * ``rolling_std_{sensor}``  — rolling std  over ``window`` cycles

        for each sensor in ``sensors``. The original raw sensor columns
        are removed. Row count is unchanged.

    Note:
        Business context — a single sensor snapshot tells you where a
        vehicle is right now; rolling statistics tell you where it is
        *heading*. A rising rolling mean in a temperature channel is a
        much stronger failure signal than any individual reading. By
        replacing raw values with trends, we give the model the
        derivative of degradation, not just the current state.
    """
    df = df.copy().sort_values(["unit_id", "time_in_cycles"]).reset_index(drop=True)

    for sensor in sensors:
        grouped = df.groupby("unit_id", group_keys=False)[sensor]
        df[f"rolling_mean_{sensor}"] = grouped.transform(
            lambda x: x.rolling(window, min_periods=1).mean()
        )
        df[f"rolling_std_{sensor}"] = grouped.transform(
            lambda x: x.rolling(window, min_periods=1).std().fillna(0.0)
        )

    df = df.drop(columns=[s for s in sensors if s in df.columns])
    return df
