"""
business_impact.py — Cost-benefit calculation engine.

Translates model performance metrics into fleet operations dollar figures.

Cost assumptions (overridable):
  - False negative (missed failure): $2,400  (lost revenue + emergency repair)
  - False positive (unnecessary maintenance): $400  (scheduled labor + parts)
  - True positive (caught failure, proactive repair): $400
  - True negative: $0

These map directly to the confusion matrix quadrants.
"""
import pandas as pd

# Default cost constants — override by passing cost_matrix to functions
DEFAULT_COST_MATRIX = {
    "false_negative": 2400,   # unplanned breakdown cost
    "false_positive": 400,    # unnecessary preventive maintenance cost
    "true_positive": 400,     # proactive maintenance (same cost as FP but avoids FN)
    "true_negative": 0,
}


def compute_net_savings(
    tp: int,
    fp: int,
    fn: int,
    tn: int,
    cost_matrix: dict = DEFAULT_COST_MATRIX,
) -> dict:
    """Calculate net savings vs. a no-model baseline (fix everything on breakdown).

    Baseline cost = (tp + fn) * false_negative_cost (all failures are unplanned).
    Model cost = tp * true_positive_cost + fp * false_positive_cost
                 + fn * false_negative_cost.

    Args:
        tp, fp, fn, tn: Confusion matrix counts.
        cost_matrix: Dict of cost assumptions (see module defaults).

    Returns:
        Dict with keys: baseline_cost, model_cost, net_savings, savings_pct.
    """
    # TODO: implement
    raise NotImplementedError


def savings_scenario_table(
    tp_rate: float,
    fp_rate: float,
    fn_rate: float,
    fleet_size: int = 100,
    cycles_per_year: int = 365,
    failure_rate: float = 0.05,
    cost_matrix: dict = DEFAULT_COST_MATRIX,
) -> pd.DataFrame:
    """Build a conservative / expected / optimistic savings table.

    Models three scenarios by applying multipliers to tp/fp/fn rates.

    Args:
        tp_rate: True positive rate (recall) at chosen threshold.
        fp_rate: False positive rate at chosen threshold.
        fn_rate: False negative rate (1 - recall) at chosen threshold.
        fleet_size: Number of vehicles in the fleet.
        cycles_per_year: Average operational cycles per vehicle per year.
        failure_rate: Fraction of vehicle-cycles that end in failure.
        cost_matrix: Dict of cost assumptions.

    Returns:
        DataFrame with columns: scenario, net_savings_annual, savings_vs_baseline_pct.
    """
    # TODO: implement
    raise NotImplementedError
