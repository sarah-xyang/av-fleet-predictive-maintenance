"""
business_impact.py — Fleet ROI and cost-benefit calculation engine.

Translates model performance (precision, recall) into the dollar figures
that fleet operations managers and executives use to decide whether to
deploy a predictive maintenance system.

All cost assumptions are function parameters with sensible defaults —
call the functions with different values to run scenario analyses.
"""
from __future__ import annotations


def calculate_fleet_roi(
    fleet_size: int,
    precision: float,
    recall: float,
    failure_rate: float = 0.08,
    unplanned_cost: float = 2_400.0,
    scheduled_cost: float = 400.0,
    model_upkeep_annual: float = 20_000.0,
) -> dict[str, float]:
    """Calculate annual fleet maintenance cost and ROI with vs. without the model.

    Models a single year of fleet operations under two regimes:

    **Baseline (no model):** every failure is unplanned.

    **Model-assisted:** the model intercepts ``recall`` fraction of
    failures before they happen (converting them from unplanned to
    scheduled events), misses the remainder, and generates false alarms
    at a rate determined by ``precision``.

    All intermediate quantities are returned so callers can build
    reports or sensitivity tables without re-implementing the maths.

    Args:
        fleet_size: Number of vehicles in the fleet.
        precision: Model precision — fraction of maintenance alerts that
            correspond to genuine upcoming failures (TP / (TP + FP)).
        recall: Model recall — fraction of genuine upcoming failures that
            the model flags in time (TP / (TP + FN)).
        failure_rate: Fraction of the fleet that experiences an
            unplanned breakdown per year without the model. Defaults to
            0.08 (8%), based on typical mid-scale AV fleet benchmarks.
        unplanned_cost: Cost per unplanned breakdown in dollars.
            Defaults to $2,400 (lost revenue + emergency repair premium).
        scheduled_cost: Cost per proactive maintenance event in dollars.
            Defaults to $400 (scheduled labour + parts).
        model_upkeep_annual: Annual cost to maintain the model in
            production (monitoring, retraining, infrastructure).
            Defaults to $20,000.

    Returns:
        Dictionary with the following keys (all dollar values are floats):

        * ``baseline_cost``       — annual cost with no model (all failures unplanned)
        * ``expected_failures``   — expected number of failures per year
        * ``caught_failures``     — failures intercepted by the model (TP)
        * ``missed_failures``     — failures not caught by the model (FN)
        * ``false_alarms``        — unnecessary maintenance events (FP)
        * ``cost_proactive``      — cost of proactive maintenance (TP × scheduled_cost)
        * ``cost_missed``         — cost of missed failures (FN × unplanned_cost)
        * ``cost_false_alarms``   — cost of false alarms (FP × scheduled_cost)
        * ``model_upkeep_annual`` — annual model maintenance cost (pass-through)
        * ``model_cost``          — total annual cost with model (includes upkeep)
        * ``gross_savings``       — savings before subtracting model upkeep cost
        * ``net_savings``         — savings after subtracting model upkeep cost
        * ``roi_pct``             — net savings as a percentage of baseline cost

    Note:
        Business context — this function answers the single most
        important question in any AI deployment decision: is the system
        worth the cost of building and running it? By expressing model
        performance (precision, recall) directly in dollars, it gives
        fleet operations managers a number they can defend to a CFO
        without any knowledge of machine learning.
    """
    # ── Baseline: no model, all failures are unplanned ────────────────────────
    # Every vehicle that fails generates the full emergency breakdown cost.
    expected_failures = fleet_size * failure_rate
    baseline_cost     = expected_failures * unplanned_cost

    # ── Model-assisted failure decomposition ──────────────────────────────────
    # Recall tells us how many failures the model catches in time to schedule.
    caught_failures = expected_failures * recall
    missed_failures = expected_failures * (1.0 - recall)

    # Precision tells us the ratio of real alerts to total alerts.
    # Rearranging TP/(TP+FP) = precision → FP = TP × (1/precision − 1).
    false_alarms = caught_failures * (1.0 / precision - 1.0) if precision > 0 else 0.0

    # ── Cost components with model deployed ───────────────────────────────────
    cost_proactive   = caught_failures * scheduled_cost   # caught → cheap scheduled maint.
    cost_missed      = missed_failures * unplanned_cost   # missed → full breakdown cost
    cost_false_alarms = false_alarms   * scheduled_cost   # false alarm → wasted maint.

    # Total model cost includes the upkeep of the model itself.
    model_cost = cost_proactive + cost_missed + cost_false_alarms + model_upkeep_annual

    # ── Savings metrics ───────────────────────────────────────────────────────
    # Gross savings: improvement in direct maintenance cost before model upkeep.
    gross_savings = baseline_cost - (model_cost - model_upkeep_annual)

    # Net savings: what the fleet actually keeps after paying for the model.
    net_savings = baseline_cost - model_cost

    # ROI expressed as a percentage of what the fleet was spending before.
    roi_pct = (net_savings / baseline_cost) * 100.0 if baseline_cost > 0 else 0.0

    return {
        "baseline_cost"      : round(baseline_cost,      2),
        "expected_failures"  : round(expected_failures,   2),
        "caught_failures"    : round(caught_failures,     2),
        "missed_failures"    : round(missed_failures,     2),
        "false_alarms"       : round(false_alarms,        2),
        "cost_proactive"     : round(cost_proactive,      2),
        "cost_missed"        : round(cost_missed,         2),
        "cost_false_alarms"  : round(cost_false_alarms,   2),
        "model_upkeep_annual": round(model_upkeep_annual, 2),
        "model_cost"         : round(model_cost,          2),
        "gross_savings"      : round(gross_savings,       2),
        "net_savings"        : round(net_savings,         2),
        "roi_pct"            : round(roi_pct,             4),
    }
