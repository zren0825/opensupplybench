"""Baseline 3: Periodic-review (R, S) base-stock policy (v0.2).

The canonical OR benchmark: review only every R days and order up to a base-stock
level S covering the protection interval (lead time + review period) plus safety
stock. Because it orders on a fixed cadence rather than continuously, it is
naturally efficient under a fixed ordering cost — a strong classical baseline.
"""

from __future__ import annotations

import math

from opensupply.policies.base import Policy
from opensupply.policies.util import z_for
from opensupply.simulator import Observation
from opensupply.forecast import forecast_tool


class PeriodicReviewPolicy(Policy):
    name = "periodic_review"

    def __init__(self, method: str = "exp_smoothing"):
        self.method = method

    def decide(self, obs: Observation) -> float:
        s = obs.scenario
        R = s.review_period
        if obs.day % R != 0:  # only act on review days
            return 0.0
        L = obs.expected_lead
        protect = L + R  # must cover demand until the next order can arrive
        fc = forecast_tool(obs.demand_history, int(round(protect)), method=self.method)
        z = z_for(s.service_level)
        base_stock = fc.mean_demand * protect + z * fc.uncertainty * math.sqrt(max(1.0, protect))
        return max(0.0, base_stock - obs.inventory_position)
