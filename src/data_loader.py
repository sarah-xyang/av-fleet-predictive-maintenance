"""
data_loader.py — Load NASA CMAPS FD001 text files into clean DataFrames.

The raw files are plain-text, space-separated, with no column headers and
no index column. This module centralises the schema definition so every
notebook and script uses identical column names.
"""
from pathlib import Path

import pandas as pd

# ── Column schema ─────────────────────────────────────────────────────────────
# 26 columns in order: two identity columns, three operating-condition
# columns, and 21 sensor measurement channels.
COLUMN_NAMES: list[str] = (
    ["unit_id", "time_in_cycles"]
    + [f"op_setting_{i}" for i in range(1, 4)]
    + [f"sensor_{i}"     for i in range(1, 22)]
)


def load_cmaps_data(filepath: str | Path) -> pd.DataFrame:
    """Load a CMAPS FD001 sensor file into a named DataFrame.

    Reads a space-separated text file with no header row, assigns the
    canonical 26-column schema, and drops any all-NaN columns that can
    appear when the raw file has trailing whitespace on each line.

    Args:
        filepath: Path to a CMAPS text file (train_FD001.txt or
            test_FD001.txt). Accepts both ``str`` and ``pathlib.Path``.

    Returns:
        DataFrame with 26 named columns. ``unit_id`` and
        ``time_in_cycles`` are integers; all sensor and setting columns
        are floats.

    Note:
        Business context — this function exists because the raw NASA
        files ship with no headers and variable-width whitespace
        between columns. Every downstream step (EDA, preprocessing,
        modeling) needs identical column names; centralising the schema
        here means a single fix propagates everywhere if the source
        format ever changes.
    """
    df = pd.read_csv(
        filepath,
        sep=r"\s+",
        header=None,
        names=COLUMN_NAMES,
        engine="python",
    )
    # Drop any all-NaN columns produced by trailing whitespace in the file.
    df = df.dropna(axis=1, how="all")
    df["unit_id"]        = df["unit_id"].astype(int)
    df["time_in_cycles"] = df["time_in_cycles"].astype(int)
    return df


def load_rul_labels(filepath: str | Path) -> pd.DataFrame:
    """Load the ground-truth RUL file for the CMAPS test set.

    RUL_FD001.txt contains one integer per line: the true Remaining
    Useful Life at the last observed cycle for each test vehicle. The
    file has no header. Line *n* corresponds to ``unit_id`` *n*
    (1-indexed), matching the ``unit_id`` values in the test set.

    Args:
        filepath: Path to ``RUL_FD001.txt``. Accepts both ``str``
            and ``pathlib.Path``.

    Returns:
        DataFrame with columns ``['unit_id', 'true_rul']``.
        ``unit_id`` runs from 1 to the number of test vehicles.
        ``true_rul`` is an integer.

    Note:
        Business context — unlike the training set (where every vehicle
        is observed to failure so RUL can be computed per row), the test
        set is truncated at an unknown point. NASA provides the true RUL
        at that cutoff separately in this file. This mirrors real
        deployment: at inference time you see the vehicle's current
        sensor snapshot and must predict how many cycles remain.
    """
    df = pd.read_csv(
        filepath,
        sep=r"\s+",
        header=None,
        names=["true_rul"],
        engine="python",
    )
    df = df.dropna(axis=1, how="all")
    df.insert(0, "unit_id", range(1, len(df) + 1))
    df["unit_id"]  = df["unit_id"].astype(int)
    df["true_rul"] = df["true_rul"].astype(int)
    return df
