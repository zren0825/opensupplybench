"""Evaluation metrics (Phase 5.2).

A single, shared metric definition every policy is scored against, so the
Phase-6 results tables are apples-to-apples. Regret is measured against the
best cost achieved by any evaluated policy on the same scenario (a practical
stand-in for an oracle until a true oracle is added).
"""

from __future__ import annotations

from typing import Dict, List
import numpy as np

from opensupply.simulator import SimulationResult


def metrics_from_result(result: SimulationResult) -> Dict[str, float]:
    demand_total = result.units_sold + result.lost_sales
    stockout_rate = result.lost_sales / demand_total if demand_total else 0.0
    fill_rate = result.units_sold / demand_total if demand_total else 1.0
    return {
        "total_cost": result.total_cost,
        "holding_cost": result.holding_cost,
        "stockout_cost": result.stockout_cost,
        "ordering_cost": result.ordering_cost,
        "stockout_rate": stockout_rate,
        "fill_rate": fill_rate,           # achieved service level (units filled / demanded)
        "lost_sales": float(result.lost_sales),
        "overstock": float(result.ending_inventory),
        "avg_inventory": result.avg_inventory,
        "num_orders": float(result.num_orders),
    }


def decision_stability(order_quantities: List[float]) -> float:
    """Coefficient of variation of a policy's order quantities under repeated
    identical inputs. Lower = more stable (0 = deterministic). Used mainly to
    show LLM-only decisions are unstable (Phase 4.1)."""
    q = np.asarray(order_quantities, dtype=float)
    if len(q) < 2 or q.mean() == 0:
        return 0.0
    return float(q.std() / q.mean())


def add_regret(rows: List[Dict[str, float]], group_key: str = "scenario_id") -> None:
    """Mutate rows in place, adding a 'rel_regret' column = total_cost minus the
    best total_cost seen for the same scenario across the *evaluated* policies.
    This is a within-experiment relative gap; prefer `add_oracle_regret` (regret
    vs. a clairvoyant reference) for the headline number."""
    best: Dict[str, float] = {}
    for r in rows:
        k = r[group_key]
        best[k] = min(best.get(k, float("inf")), r["total_cost"])
    for r in rows:
        r["rel_regret"] = r["total_cost"] - best[r[group_key]]
