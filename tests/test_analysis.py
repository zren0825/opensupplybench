"""Tests for VBC, weighting, and tuning (Phase 6 analysis). Offline."""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from opensupply import Scenario
from opensupply.evaluation import (
    virtual_best_classical, add_vbc_regret, scenario_weights, cost_ratio,
    weighted_metric, PREVALENCE_PRIOR, train_test_split, tune_safety_factor,
)


def _rows():
    # two scenarios, three classical policies + one 'hybrid'
    return [
        {"scenario_id": "a", "demand_type": "stable", "policy": "p1", "total_cost": 100, "oracle_cost": 80, "fill_rate": 0.9},
        {"scenario_id": "a", "demand_type": "stable", "policy": "p2", "total_cost": 90, "oracle_cost": 80, "fill_rate": 0.95},
        {"scenario_id": "a", "demand_type": "stable", "policy": "hybrid", "total_cost": 85, "oracle_cost": 80, "fill_rate": 0.96},
        {"scenario_id": "b", "demand_type": "viral_spike", "policy": "p1", "total_cost": 500, "oracle_cost": 300, "fill_rate": 0.7},
        {"scenario_id": "b", "demand_type": "viral_spike", "policy": "p2", "total_cost": 600, "oracle_cost": 300, "fill_rate": 0.6},
        {"scenario_id": "b", "demand_type": "viral_spike", "policy": "hybrid", "total_cost": 450, "oracle_cost": 300, "fill_rate": 0.8},
    ]


def test_vbc_picks_per_scenario_best_classical():
    rows = _rows()
    vbc = virtual_best_classical(rows, ["p1", "p2"])
    assert vbc["a"]["cost"] == 90 and vbc["a"]["policy"] == "p2"
    assert vbc["b"]["cost"] == 500 and vbc["b"]["policy"] == "p1"


def test_hybrid_beats_vbc_flagged():
    rows = _rows()
    add_vbc_regret(rows, ["p1", "p2"])
    hyb = [r for r in rows if r["policy"] == "hybrid"]
    # hybrid (85 vs VBC 90 on a; 450 vs 500 on b) beats VBC on both
    assert all(r["beats_vbc"] for r in hyb)


def test_prevalence_weighting_differs_from_uniform():
    rows = _rows()
    hyb = [r for r in rows if r["policy"] == "hybrid"]
    uni = weighted_metric(hyb, cost_ratio, "uniform")
    prev = weighted_metric(hyb, cost_ratio, "prevalence")
    # uniform averages the two ratios equally; prevalence down-weights the rare
    # viral scenario, so the two aggregates must differ.
    assert abs(uni - prev) > 1e-6
    assert abs(sum(PREVALENCE_PRIOR.values()) - 1.0) < 1e-9


def test_weights_sum_to_one():
    rows = _rows()
    for scheme in ("uniform", "prevalence"):
        w = scenario_weights(rows, scheme)
        assert abs(sum(w.values()) - 1.0) < 1e-9


def test_tuning_train_test_split_and_fit():
    scs = [Scenario(scenario_id=f"stable__fixed__b-low__s-low__seed-{i}",
                    seed=i, horizon_days=45, demand_type="stable") for i in range(6)]
    train, test = train_test_split(scs)
    assert len(train) == 3 and len(test) == 3  # parity split
    best_k, scores = tune_safety_factor(train, candidates=(0.5, 1.5, 3.0))
    assert best_k in (0.5, 1.5, 3.0)
    assert all(v > 0 for v in scores.values())


if __name__ == "__main__":
    for fn in list(globals().values()):
        if callable(fn) and getattr(fn, "__name__", "").startswith("test_"):
            fn()
    print("all analysis tests passed")
