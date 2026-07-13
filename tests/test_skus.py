"""Tests for the v0.4 SKU archetypes + economic conventions. Offline."""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from opensupply import Scenario
from opensupply.skus import SKU_ARCHETYPES, SKU_BY_NAME


def test_archetypes_are_a_small_fixed_set():
    assert 4 <= len(SKU_ARCHETYPES) <= 8
    names = [s.name for s in SKU_ARCHETYPES]
    assert len(names) == len(set(names))  # unique names


def test_stockout_tethered_to_margin():
    for s in SKU_ARCHETYPES:
        e = s.economics()
        margin = e["selling_price"] - e["unit_cost"]
        assert abs(e["stockout_cost"] - margin * s.stockout_multiple) < 1e-6


def test_budget_is_days_of_supply():
    for s in SKU_ARCHETYPES:
        e = s.economics()
        assert abs(e["budget_per_order"] - s.base_demand * s.unit_cost * s.days_of_supply) < 1e-6


def test_holding_scales_with_unit_cost():
    for s in SKU_ARCHETYPES:
        e = s.economics()
        assert abs(e["holding_cost"] - s.unit_cost * s.holding_rate) < 1e-6


def test_archetypes_span_the_holding_stockout_tradeoff():
    sls = [Scenario(demand_type="stable", **s.economics()).implied_service_level()
           for s in SKU_ARCHETYPES]
    # a real spread of cost regimes: holding-dominated (low SL) to stockout-dominated
    assert min(sls) < 0.7 and max(sls) > 0.95
    assert max(sls) - min(sls) > 0.3


def test_economics_deterministic():
    assert SKU_BY_NAME["premium"].economics() == SKU_BY_NAME["premium"].economics()
    assert SKU_BY_NAME["bulky"].economics()["sku_type"] == "bulky"


if __name__ == "__main__":
    for fn in list(globals().values()):
        if callable(fn) and getattr(fn, "__name__", "").startswith("test_"):
            fn()
    print("all sku tests passed")
