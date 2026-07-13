"""First experiment (plan: Day 6-7 / Phase 2-3 checkpoint).

Runs a single 90-day scenario through all three non-LLM baselines and prints
total cost / stockout / order history, then saves the inventory curve to
paper/figures/. This is the "prove the problem can be simulated" milestone.

Usage:
    python experiments/run_first_experiment.py
"""

from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from opensupply import Scenario, Simulator
from opensupply.policies import (
    ReorderPointPolicy,
    ForecastSafetyStockPolicy,
    CostMinimizationPolicy,
)
from opensupply.evaluation import metrics_from_result

FIG_DIR = os.path.join(os.path.dirname(__file__), "..", "paper", "figures")


def make_policies():
    return {
        "rule_based": ReorderPointPolicy(),
        "forecast_safety": ForecastSafetyStockPolicy(),
        "cost_min": CostMinimizationPolicy(),
    }


def main():
    scenario = Scenario(
        scenario_id="demo_viral_delayed",
        seed=7,
        horizon_days=90,
        demand_type="viral_spike",
        lead_time_type="delayed",
        holding_cost=0.05,
        stockout_cost=8.0,
        order_cost=5.0,
        initial_inventory=20,
        business_note="Small home-goods shop; one product went semi-viral on social.",
        supplier_note="Overseas supplier, occasionally 3-9 days late.",
    )

    print(f"Scenario: {scenario.scenario_id}  "
          f"(demand={scenario.demand_type}, lead={scenario.lead_time_type})\n")

    results = {}
    print(f"{'policy':<18}{'total':>10}{'hold':>9}{'stockout':>10}"
          f"{'order':>8}{'lost':>7}{'orders':>8}{'end_inv':>9}")
    print("-" * 79)
    for name, policy in make_policies().items():
        res = Simulator(scenario).run(policy)
        results[name] = res
        m = metrics_from_result(res)
        print(f"{name:<18}{m['total_cost']:>10.1f}{m['holding_cost']:>9.1f}"
              f"{m['stockout_cost']:>10.1f}{m['ordering_cost']:>8.1f}"
              f"{int(m['lost_sales']):>7}{int(m['num_orders']):>8}"
              f"{int(m['overstock']):>9}")

    _plot(scenario, results)


def _plot(scenario, results):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("\n(matplotlib not installed; skipping inventory-curve figure)")
        return

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 7), sharex=True)
    demand = [d["demand"] for d in next(iter(results.values())).daily]
    ax1.bar(range(len(demand)), demand, color="0.8", label="demand")
    for name, res in results.items():
        ax1.plot([d["on_hand"] for d in res.daily], label=f"{name} on-hand")
    ax1.set_ylabel("units")
    ax1.set_title(f"Inventory vs. demand — {scenario.scenario_id}")
    ax1.legend(fontsize=8)

    for name, res in results.items():
        days = [o["day"] for o in res.order_history]
        qtys = [o["qty"] for o in res.order_history]
        ax2.stem(days, qtys, label=name) if days else None
    ax2.set_ylabel("order qty")
    ax2.set_xlabel("day")
    ax2.set_title("Order history")
    ax2.legend(fontsize=8)

    os.makedirs(FIG_DIR, exist_ok=True)
    out = os.path.join(FIG_DIR, "first_experiment.png")
    fig.tight_layout()
    fig.savefig(out, dpi=120)
    print(f"\nSaved figure -> {os.path.relpath(out)}")


if __name__ == "__main__":
    main()
