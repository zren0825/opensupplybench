"""Tests for the v0.2 research-hardening: count demand, strong baselines,
fill rate, clairvoyant oracle. All offline.
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np

from opensupply import Scenario, Simulator, DemandGenerator, DEMAND_TYPES
from opensupply.policies import BASELINE_POLICIES, PeriodicReviewPolicy
from opensupply.evaluation import metrics_from_result, add_oracle_regret
from opensupply.agents.schemas import SCENARIO_TYPES


def test_demand_taxonomy_matches_classifier():
    # every generated demand type is something the LLM classifier can name
    assert set(DEMAND_TYPES).issubset(set(SCENARIO_TYPES))
    assert len(DEMAND_TYPES) == 7


def test_dispersion_controls_variance():
    g = DemandGenerator()
    lo = g.generate("stable", 400, seed=1, dispersion="low")
    hi = g.generate("stable", 400, seed=1, dispersion="high")
    assert hi.var() > lo.var()  # more overdispersion -> larger variance
    assert lo.dtype.kind == "i"  # integer counts


def test_periodic_review_orders_only_on_cadence():
    sc = Scenario(seed=2, horizon_days=90, demand_type="stable", review_period=7)
    res = Simulator(sc).run(PeriodicReviewPolicy())
    assert all(o["day"] % 7 == 0 for o in res.order_history)


def test_fill_rate_present_and_bounded():
    sc = Scenario(seed=2, horizon_days=60, demand_type="seasonal")
    m = metrics_from_result(Simulator(sc).run(BASELINE_POLICIES["rule_based"]()))
    assert 0.0 <= m["fill_rate"] <= 1.0
    assert abs(m["fill_rate"] + m["stockout_rate"] - 1.0) < 1e-9


def test_oracle_regret_nonnegative():
    scenarios = {}
    rows = []
    for dem in ["stable", "viral_spike", "intermittent"]:
        sc = Scenario(scenario_id=dem, seed=5, horizon_days=90, demand_type=dem)
        scenarios[dem] = sc
        res = Simulator(sc).run(BASELINE_POLICIES["forecast_safety"]())
        rows.append({"scenario_id": dem, "total_cost": res.total_cost})
    add_oracle_regret(rows, scenarios)
    assert all(r["regret"] >= 0.0 for r in rows)  # clipped at 0


def test_implied_service_level_increases_with_penalty():
    lo = Scenario(stockout_cost=1.0).implied_service_level()
    hi = Scenario(stockout_cost=12.0).implied_service_level()
    assert 0.5 < lo < hi < 1.0


if __name__ == "__main__":
    for fn in list(globals().values()):
        if callable(fn) and getattr(fn, "__name__", "").startswith("test_"):
            fn()
    print("all v0.2 tests passed")
