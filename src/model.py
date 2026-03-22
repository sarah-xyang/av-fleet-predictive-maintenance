"""
model.py — Training pipeline and time-series-safe cross-validation.

Trains an XGBoost classifier to predict whether a vehicle will fail
within FAILURE_WINDOW operational cycles.

Cross-validation splits by unit boundary to prevent data leakage
(a unit's future cycles must never appear in a training fold).
"""
import joblib
from pathlib import Path

import pandas as pd
from sklearn.pipeline import Pipeline

MODELS_DIR = Path(__file__).resolve().parents[1] / "models"


def build_pipeline() -> Pipeline:
    """Return an unfitted sklearn Pipeline with preprocessing and XGBoost.

    Returns:
        sklearn Pipeline ready for .fit(X, y).
    """
    # TODO: implement
    raise NotImplementedError


def train(X: pd.DataFrame, y: pd.Series) -> Pipeline:
    """Fit the pipeline on training data and return the fitted model.

    Args:
        X: Feature DataFrame (no label columns, no unit_number/time_in_cycles).
        y: Binary label Series (1 = will fail within window).

    Returns:
        Fitted sklearn Pipeline.
    """
    # TODO: implement
    raise NotImplementedError


def save_model(pipeline: Pipeline, filename: str = "xgb_pipeline.joblib") -> Path:
    """Persist a fitted pipeline to the models/ directory.

    Args:
        pipeline: Fitted sklearn Pipeline.
        filename: Output filename within models/.

    Returns:
        Path to the saved file.
    """
    # TODO: implement
    raise NotImplementedError


def load_model(filename: str = "xgb_pipeline.joblib") -> Pipeline:
    """Load a previously saved pipeline from the models/ directory.

    Args:
        filename: Filename within models/.

    Returns:
        Fitted sklearn Pipeline.
    """
    # TODO: implement
    raise NotImplementedError
