"""Tuned continuous-review (s, S) base-stock policy.

Same structure as ForecastSafetyStockPolicy, but the safety buffer is a single
tunable `safety_factor` k (multiplying forecast σ) rather than a fixed
service-level z. `k` is meant to be fitted on a *training* split of scenarios and
evaluated on held-out seeds (see evaluation/tuning.py), so the classical baseline
is genuinely well-tuned — not a raw textbook formula — without hindsight
overfitting. This is what makes "the hybrid beats a strong baseline" credible.
"""

from __future__ import annotations

import math

from opensupply.policies.base import Policy
from opensupply.simulator import Observation
from opensupply.forecast import forecast_tool


class TunedBaseStockPolicy(Policy):
    name = "tuned_ss"

    def __init__(self, safety_factor: float = 1.645, method: str = "exp_smoothing"):
        self.k = safety_factor
        self.method = method

    def decide(self, obs: Observation) -> float:
        s = obs.scenario
        L, R = obs.expected_lead, s.review_period
        fc = forecast_tool(obs.demand_history, int(round(L + R)), method=self.method)
        safety = self.k * fc.uncertainty * math.sqrt(max(1.0, L))
        reorder_point = fc.mean_demand * L + safety
        order_up_to = fc.mean_demand * (L + R) + safety
        if obs.inventory_position <= reorder_point:
            return max(0.0, order_up_to - obs.inventory_position)
        return 0.0
