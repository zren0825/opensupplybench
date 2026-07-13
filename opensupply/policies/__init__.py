"""Replenishment policies.

Baselines (Phase 3, hardened in v0.2) — all matched to the fixed-order-cost
economics (they order on a trigger or cadence, not daily):
    ReorderPointPolicy        - continuous-review (s, S), simple safety buffer
    ForecastSafetyStockPolicy - (s, S) with forecast-driven safety stock
    PeriodicReviewPolicy      - (R, S) base-stock, orders every R days
    CostMinimizationPolicy    - simulation-based optimizer (also a tool)

LLM methods (Phase 4) live in opensupply.agents.
"""

from opensupply.policies.base import Policy
from opensupply.policies.rule_based import ReorderPointPolicy
from opensupply.policies.forecast_safety import ForecastSafetyStockPolicy
from opensupply.policies.periodic_review import PeriodicReviewPolicy
from opensupply.policies.cost_min import CostMinimizationPolicy
from opensupply.policies.tuned import TunedBaseStockPolicy

BASELINE_POLICIES = {
    "rule_based": ReorderPointPolicy,
    "forecast_safety": ForecastSafetyStockPolicy,
    "periodic_review": PeriodicReviewPolicy,
    "cost_min": CostMinimizationPolicy,
}

# The classical policies whose per-scenario minimum defines the Virtual Best
# Classical reference (the strong bar the hybrid must beat). `tuned_ss` is added
# by the analysis once its safety factor is fitted on the train split.
CLASSICAL_FOR_VBC = ("rule_based", "forecast_safety", "periodic_review",
                     "cost_min", "tuned_ss")

__all__ = [
    "Policy",
    "ReorderPointPolicy",
    "ForecastSafetyStockPolicy",
    "PeriodicReviewPolicy",
    "CostMinimizationPolicy",
    "TunedBaseStockPolicy",
    "BASELINE_POLICIES",
    "CLASSICAL_FOR_VBC",
]
