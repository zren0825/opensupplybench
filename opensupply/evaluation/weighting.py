"""Scenario weighting + scale-normalized aggregation.

Two problems with a plain equal-weighted mean of raw cost:
  1. Scale domination — a high-volume/high-penalty scenario costs 20-50x a
     low-volume stable one, so the mean measures the expensive scenarios only.
     Fix: aggregate a *normalized* metric (cost ÷ oracle) so scenarios are
     comparable regardless of scale.
  2. Prevalence — a balanced grid puts equal mass on `viral_spike` and `stable`,
     but viral spikes are rare in practice. Equal weighting over-represents the
     hard cases and flatters an adaptive method. Fix: report both a uniform
     weighting and a documented realistic prevalence prior.

The prevalence prior below is a *documented assumption*, not measured — stated so
a reviewer can challenge the numbers, not the methodology. Common patterns carry
most of the mass; genuine spikes/launches are rare.
"""

from __future__ import annotations

from collections import defaultdict
from typing import Dict, List

# Documented heuristic prior over demand patterns (sums to 1.0). Stated as an
# assumption in the paper; the Phase-7 case study can later replace it with a
# data-grounded estimate.
PREVALENCE_PRIOR: Dict[str, float] = {
    "stable": 0.25,
    "seasonal": 0.20,
    "promotion_spike": 0.15,
    "intermittent": 0.15,
    "declining": 0.10,
    "new_product": 0.10,
    "viral_spike": 0.05,
}


def scenario_weights(rows: List[Dict], scheme: str = "uniform",
                     type_key: str = "demand_type") -> Dict[int, float]:
    """Return per-row weights (indexed by position) that sum to 1.

    'uniform'    — every scenario counts equally.
    'prevalence' — each demand type gets PREVALENCE_PRIOR mass, split equally
                   among the scenarios of that type present in `rows`.
    """
    n = len(rows)
    if scheme == "uniform":
        return {i: 1.0 / n for i in range(n)} if n else {}
    if scheme == "prevalence":
        by_type = defaultdict(list)
        for i, r in enumerate(rows):
            by_type[r[type_key]].append(i)
        present = {t: PREVALENCE_PRIOR.get(t, 0.0) for t in by_type}
        total = sum(present.values()) or 1.0
        weights = {}
        for t, idxs in by_type.items():
            share = (present[t] / total) / len(idxs)
            for i in idxs:
                weights[i] = share
        return weights
    raise ValueError(f"unknown weighting scheme {scheme!r}")


def cost_ratio(row: Dict) -> float:
    """Scale-free cost = total_cost / oracle_cost (>= ~1). Requires the
    'oracle_cost' column written by add_oracle_regret."""
    oc = row.get("oracle_cost")
    if not oc:
        return float("nan")
    return row["total_cost"] / oc


def weighted_metric(rows: List[Dict], value_fn, scheme: str = "uniform") -> float:
    """Weighted mean of value_fn(row) over rows under the chosen scheme."""
    w = scenario_weights(rows, scheme)
    return sum(w[i] * value_fn(r) for i, r in enumerate(rows))
