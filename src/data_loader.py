"""
data_loader.py — Load and validate NASA CMAPS FD001 text files.

The raw files are plain-text, space-separated, no column headers.
26 columns: unit_number, time_in_cycles, 3 operational settings, 21 sensors.
"""
from pathlib import Path

import pandas as pd

COLUMN_NAMES = (
    ["unit_number", "time_in_cycles"]
    + [f"operational_setting_{i}" for i in range(1, 4)]
    + [f"sensor_{i}" for i in range(1, 22)]
)


def load_cmaps(filepath: str | Path) -> pd.DataFrame:
    """Load a CMAPS FD001 text file into a DataFrame.

    Args:
        filepath: Path to a space-separated CMAPS file (train or test).

    Returns:
        DataFrame with named columns; unit_number and time_in_cycles as int.
    """
    # TODO: implement
    raise NotImplementedError


def load_rul_labels(filepath: str | Path) -> pd.Series:
    """Load the ground-truth RUL file for the test set.

    Args:
        filepath: Path to RUL_FD001.txt (one integer per line).

    Returns:
        Series of RUL values indexed from 1 (matching unit_number).
    """
    # TODO: implement
    raise NotImplementedError
