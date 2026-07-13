"""Baseline 3: Cost-minimization policy / optimizer tool (Phase 3.3).

Enumerate candidate order quantities and, for each, Monte-Carlo simulate the
next `lead + review` days to estimate expected cost (holding + stockout).
Pick the cheapest. This doubles as the `optimizer` tool the Phase-4 hybrid
agent calls, so it is written to be reusable outside a Policy too.
"""

from __future__ import annotations

import numpy as np

from opensupply.policies.base import Policy
from opensupply.simulator import Observation


def expected_cost_of_order(
    order_qty: int,
    on_hand: int,
    on_order: int,
    demand_samples: np.ndarray,   # shape (n_sims, horizon)
    lead: int,
    holding_cost: float,
    stockout_cost: float,
    order_cost: float,
) -> float:
    """Estimate expected holding+stockout+ordering cost over the horizon if we
    place `order_qty` now (arriving after `lead` days). Existing pipeline
    (`on_order`) is assumed to arrive at the start for a conservative estimate.
    """
    n_sims, horizon = demand_samples.shape
    inv = np.full(n_sims, float(on_hand + on_order))
    cost = np.full(n_sims, order_cost if order_qty > 0 else 0.0)
    for t in range(horizon):
        if t == lead:
            inv += order_qty
        d = demand_samples[:, t]
        sold = np.minimum(d, inv)
        lost = d - sold
        inv -= sold
        cost += holding_cost * inv + stockout_cost * lost
    return float(cost.mean())


class CostMinimizationPolicy(Policy):
    name = "cost_min"

    def __init__(
        self,
        review_period: int = 7,
        n_sims: int = 50,
        n_candidates: int = 16,
        seed: int = 12345,
    ):
        self.review_period = review_period
        self.n_sims = n_sims
        self.n_candidates = n_candidates
        self._rng = np.random.default_rng(seed)

    def _demand_samples(self, obs: Observation, horizon: int) -> np.ndarray:
        hist = obs.demand_history
        if len(hist) < 3:
            mean = 20.0
            return self._rng.poisson(mean, size=(self.n_sims, horizon))
        recent = hist[-28:]
        # bootstrap resample recent demand to preserve its distribution shape
        idx = self._rng.integers(0, len(recent), size=(self.n_sims, horizon))
        return recent[idx]

    def best_order(self, obs: Observation) -> int:
        L = int(round(obs.expected_lead))
        horizon = L + obs.scenario.review_period
        samples = self._demand_samples(obs, horizon)

        # Candidate range must cover spikes: use the *recent peak* demand, not
        # just the mean, so the optimizer can order enough during a viral/promo
        # surge (the mean lags badly there).
        mean_d = float(np.mean(samples))
        peak_d = float(np.max(samples)) if samples.size else mean_d
        q_max = int(max(1.0, max(mean_d * 2.0, peak_d)) * horizon)
        candidates = np.unique(
            np.linspace(0, q_max, self.n_candidates).round().astype(int)
        )

        best_q, best_cost = 0, float("inf")
        for q in candidates:
            c = expected_cost_of_order(
                int(q),
                obs.on_hand,
                obs.on_order,
                samples,
                L,
                obs.scenario.holding_cost,
                obs.scenario.stockout_cost,
                obs.scenario.order_cost,
            )
            if c < best_cost:
                best_cost, best_q = c, int(q)
        return best_q

    def decide(self, obs: Observation) -> float:
        return float(self.best_order(obs))
