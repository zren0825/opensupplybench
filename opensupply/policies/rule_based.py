"""Baseline 1: Rule-based continuous-review (s, S) policy (Phase 3.1, v0.2).

When inventory position drops to the reorder point s, order up to S; otherwise
order nothing. Both levels use a running mean/σ of demand and a modest safety
buffer, so — unlike the v0.1 version — this baseline carries real safety stock
and reorders on a trigger (not daily). It is the simple, robust reference the
smarter methods must beat.
"""

from __future__ import annotations

import numpy as np

from opensupply.policies.base import Policy
from opensupply.simulator import Observation


class ReorderPointPolicy(Policy):
    name = "rule_based"

    def __init__(self, safety_factor: float = 1.0):
        # z ≈ 1.0 → ~84% cycle service: a plain, defensible buffer.
        self.safety_factor = safety_factor

    def _stats(self, obs: Observation):
        hist = obs.demand_history
        if len(hist) < 3:
            b = obs.scenario.base_demand
            return b, 0.5 * b
        w = hist[-28:]
        return float(np.mean(w)), float(np.std(w))

    def decide(self, obs: Observation) -> float:
        s = obs.scenario
        mean_d, sigma = self._stats(obs)
        L, R = obs.expected_lead, s.review_period
        safety = self.safety_factor * sigma * np.sqrt(max(1.0, L))
        reorder_point = mean_d * L + safety
        order_up_to = mean_d * (L + R) + safety
        if obs.inventory_position <= reorder_point:
            return max(0.0, order_up_to - obs.inventory_position)
        return 0.0
