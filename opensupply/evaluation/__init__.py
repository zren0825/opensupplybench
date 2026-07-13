"""Evaluation utilities."""

from opensupply.evaluation.metrics import (
    metrics_from_result,
    decision_stability,
    add_regret,
)
from opensupply.evaluation.oracle import (
    ClairvoyantPolicy,
    oracle_result,
    oracle_total_cost,
    add_oracle_regret,
)
from opensupply.evaluation.vbc import (
    virtual_best_classical,
    add_vbc_regret,
)
from opensupply.evaluation.weighting import (
    PREVALENCE_PRIOR,
    scenario_weights,
    cost_ratio,
    weighted_metric,
)
from opensupply.evaluation.tuning import (
    train_test_split,
    tune_safety_factor,
)

__all__ = [
    "metrics_from_result",
    "decision_stability",
    "add_regret",
    "ClairvoyantPolicy",
    "oracle_result",
    "oracle_total_cost",
    "add_oracle_regret",
    "virtual_best_classical",
    "add_vbc_regret",
    "PREVALENCE_PRIOR",
    "scenario_weights",
    "cost_ratio",
    "weighted_metric",
    "train_test_split",
    "tune_safety_factor",
]
