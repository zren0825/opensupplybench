"""Clairvoyant oracle for regret (v0.2).

Provides a strong *reference* cost per scenario: a policy that knows the realized
demand path and orders just-in-time to cover it, on the review cadence. Regret is
then `policy_cost − oracle_cost`, which is far more meaningful than comparing to
the best evaluated policy.

Honest caveat: this is a strong heuristic reference, NOT a provable lower bound.
It is clairvoyant about demand but still faces the same *stochastic* lead times
as every policy, and ordering on the review cadence is a heuristic, so the true
cost-minimizing policy under lead-time uncertainty (a stochastic DP) could do
slightly better. Reported as such in the paper.
"""

from __future__ import annotations

from typing import Dict, List
import numpy as np

from opensupply.scenario import Scenario
from opensupply.simulator import Simulator, Observation
from opensupply.policies.base import Policy


class ClairvoyantPolicy(Policy):
    """Orders on the review cadence up to exactly the *known* future demand over
    the protection interval — so holding and stockout are both driven near zero,
    leaving mostly unavoidable ordering cost."""

    name = "oracle"

    def __init__(self, demand, review: int | None = None):
        self.demand = np.asarray(demand)
        self.review = review

    def decide(self, obs: Observation) -> float:
        s = obs.scenario
        R = self.review if self.review is not None else s.review_period
        if obs.day % R != 0:
            return 0.0
        L = int(round(obs.expected_lead))
        t = obs.day
        protect = L + R
        need = float(np.sum(self.demand[t + 1 : t + 1 + protect]))
        return max(0.0, need - obs.inventory_position)


# Candidate ordering cadences the clairvoyant tries; the cheapest is the
# reference. Sweeping makes it a tighter (rarely-beaten) reference than a single
# fixed cadence, trading holding vs. fixed ordering cost with hindsight.
_ORACLE_REVIEWS = (1, 2, 3, 5, 7, 10, 14)


def oracle_result(scenario: Scenario):
    sim = Simulator(scenario)  # one build; demand + lead draws are fixed by seed
    best = None
    for r in _ORACLE_REVIEWS:
        res = Simulator(scenario).run(ClairvoyantPolicy(sim.demand, review=r))
        if best is None or res.total_cost < best.total_cost:
            best = res
    return best


def oracle_total_cost(scenario: Scenario) -> float:
    return oracle_result(scenario).total_cost


def add_oracle_regret(rows: List[Dict], scenarios: Dict[str, Scenario],
                      group_key: str = "scenario_id") -> None:
    """Mutate rows in place, adding a 'regret' column = total_cost − oracle_cost
    for that scenario. `scenarios` maps scenario_id -> Scenario."""
    cache: Dict[str, float] = {}
    for r in rows:
        sid = r[group_key]
        if sid not in cache:
            cache[sid] = oracle_total_cost(scenarios[sid])
        r["oracle_cost"] = round(cache[sid], 4)   # stored so normalization is exact
        r["regret"] = max(0.0, r["total_cost"] - cache[sid])
