"""Train/test tuning of the classical baseline (no hindsight overfitting).

Splits scenarios into train/test by seed index, grid-searches the base-stock
safety factor on train (minimizing mean cost-ratio-to-oracle), and returns the
value to evaluate on the held-out test scenarios. This gives a strong, fairly
tuned classical baseline rather than a raw formula — the fair opponent for the
hybrid agent.
"""

from __future__ import annotations

from typing import List, Tuple

from opensupply.scenario import Scenario
from opensupply.simulator import Simulator
from opensupply.policies.tuned import TunedBaseStockPolicy
from opensupply.evaluation.oracle import oracle_total_cost

DEFAULT_CANDIDATES = (0.0, 0.5, 1.0, 1.28, 1.645, 2.0, 2.5, 3.0)


def _seed_index(scenario_id: str) -> int:
    tail = scenario_id.rsplit("seed-", 1)[-1]
    try:
        return int(tail)
    except ValueError:
        return 0


def train_test_split(scenarios: List[Scenario], test_on_odd: bool = True):
    """Split by seed-index parity so train/test are balanced across every cell."""
    train, test = [], []
    for sc in scenarios:
        (test if (_seed_index(sc.scenario_id) % 2 == 1) == test_on_odd else train).append(sc)
    return train, test


def _mean_cost_ratio(scenarios, k) -> float:
    ratios = []
    for sc in scenarios:
        res = Simulator(sc).run(TunedBaseStockPolicy(safety_factor=k))
        oc = oracle_total_cost(sc)
        ratios.append(res.total_cost / oc if oc > 0 else 1.0)
    return sum(ratios) / len(ratios) if ratios else float("inf")


def tune_safety_factor(train: List[Scenario], candidates=DEFAULT_CANDIDATES) -> Tuple[float, dict]:
    """Return (best_k, {k: mean_cost_ratio}) fitted on the train scenarios."""
    scores = {k: _mean_cost_ratio(train, k) for k in candidates}
    best_k = min(scores, key=scores.get)
    return best_k, scores
