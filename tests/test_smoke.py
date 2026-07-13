"""Smoke tests: the simulator + baselines run and produce sane numbers.

Run with: python -m pytest -q   (or: python tests/test_smoke.py)
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from opensupply import Scenario, Simulator, DemandGenerator, DEMAND_TYPES, forecast_tool
from opensupply.policies import (
    ReorderPointPolicy,
    ForecastSafetyStockPolicy,
    CostMinimizationPolicy,
)


def test_demand_reproducible():
    g = DemandGenerator()
    for dt in DEMAND_TYPES:
        a = g.generate(dt, 90, seed=3)
        b = g.generate(dt, 90, seed=3)
        assert (a == b).all(), f"{dt} not reproducible"
        assert (a >= 0).all() and len(a) == 90


def test_forecast_shape():
    fc = forecast_tool([10, 12, 9, 11, 13, 8, 10], horizon_days=7)
    assert fc.low_demand <= fc.mean_demand <= fc.high_demand
    assert fc.uncertainty >= 0


def test_policies_run():
    scenario = Scenario(seed=1, horizon_days=90, demand_type="stable")
    for policy in (ReorderPointPolicy(), ForecastSafetyStockPolicy(),
                   CostMinimizationPolicy()):
        res = Simulator(scenario).run(policy)
        assert res.total_cost >= 0
        assert res.units_sold >= 0
        assert len(res.daily) == 90
        assert res.lost_sales + res.units_sold > 0  # some demand occurred


def test_stockout_costs_more_when_penalty_high():
    """Sanity: a higher stockout penalty makes the same policy cost more or
    push it to hold more inventory — total stockout cost component should
    scale with the penalty for a fixed policy."""
    lo = Simulator(Scenario(seed=5, demand_type="viral_spike", stockout_cost=1.0)).run(
        ReorderPointPolicy()
    )
    hi = Simulator(Scenario(seed=5, demand_type="viral_spike", stockout_cost=12.0)).run(
        ReorderPointPolicy()
    )
    assert hi.stockout_cost >= lo.stockout_cost


if __name__ == "__main__":
    test_demand_reproducible()
    test_forecast_shape()
    test_policies_run()
    test_stockout_costs_more_when_penalty_high()
    print("all smoke tests passed")
