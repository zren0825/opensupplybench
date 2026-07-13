"""Baseline 2: Forecast + safety stock, continuous-review (s, S) (Phase 3.2, v0.2).

Uses forecast_tool to estimate demand and its uncertainty over the protection
interval, sizes safety stock from a target service level (`safety = z·σ·√L`),
and orders to an order-up-to level only when inventory position crosses the
reorder point. The reorder trigger (added in v0.2) is what stops the v0.1 bug of
re-ordering almost every day and drowning in fixed order fees.
"""

from __future__ import annotations

import math

from opensupply.policies.base import Policy
from opensupply.policies.util import z_for
from opensupply.simulator import Observation
from opensupply.forecast import forecast_tool


class ForecastSafetyStockPolicy(Policy):
    name = "forecast_safety"

    def __init__(self, method: str = "exp_smoothing"):
        self.method = method

    def decide(self, obs: Observation) -> float:
        s = obs.scenario
        L, R = obs.expected_lead, s.review_period
        fc = forecast_tool(obs.demand_history, int(round(L + R)), method=self.method)
        z = z_for(s.service_level)
        safety = z * fc.uncertainty * math.sqrt(max(1.0, L))
        reorder_point = fc.mean_demand * L + safety
        order_up_to = fc.mean_demand * (L + R) + safety
        if obs.inventory_position <= reorder_point:
            return max(0.0, order_up_to - obs.inventory_position)
        return 0.0
