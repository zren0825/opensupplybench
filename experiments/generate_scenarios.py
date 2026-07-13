"""Generate the reproducible benchmark scenario grid (Phase 5.1).

Grid = demand_types x lead_time_types x budget_levels x stockout_cost_levels
x seeds (v0.2: 7 demand types). Each scenario also gets a SKU scale
(`base_demand`) and demand `dispersion` drawn reproducibly per scenario, so the
benchmark spans low/high-volume and Poisson/overdispersed regimes without
exploding the grid. With 40 seeds this yields 7 x 3 x 3 x 3 x 40 = 7560
scenarios (>1000, per the plan).

Usage:
    python experiments/generate_scenarios.py --seeds 40 --out data/scenarios
    python experiments/generate_scenarios.py --seeds 3            # quick smoke set
"""

from __future__ import annotations

import argparse
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from opensupply.scenario import (
    Scenario,
    BUDGET_LEVELS,
    STOCKOUT_COST_LEVELS,
)
from opensupply.demand import DEMAND_TYPES
from opensupply.leadtime import LEAD_TIME_TYPES

# Per-scenario heterogeneity, drawn reproducibly (keeps the grid small).
BASE_LEVELS = [8, 20, 60]           # low / medium / high volume SKUs
DISPERSIONS = ["low", "medium", "high"]


def build_grid(seeds: int):
    scenarios = []
    idx = 0
    for demand in DEMAND_TYPES:
        for lead in LEAD_TIME_TYPES:
            for blevel, budget in BUDGET_LEVELS.items():
                for slevel, scost in STOCKOUT_COST_LEVELS.items():
                    for seed in range(seeds):
                        sid = f"{demand}__{lead}__b-{blevel}__s-{slevel}__seed-{seed}"
                        rng = np.random.default_rng(1000 + idx)
                        base = float(rng.choice(BASE_LEVELS))
                        disp = str(rng.choice(DISPERSIONS))
                        scenarios.append(
                            Scenario(
                                scenario_id=sid,
                                seed=1000 + idx,
                                horizon_days=90,
                                demand_type=demand,
                                lead_time_type=lead,
                                base_demand=base,
                                dispersion=disp,
                                mean_lead=5,
                                review_period=7,
                                service_level=0.95,
                                budget_per_order=budget,
                                budget_level=blevel,
                                stockout_cost=scost,
                                stockout_cost_level=slevel,
                                unit_cost=1.0,
                                holding_cost=0.05,
                                order_cost=5.0,
                                initial_inventory=int(base),
                            )
                        )
                        idx += 1
    return scenarios


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, default=40)
    ap.add_argument("--out", default="data/scenarios")
    args = ap.parse_args()

    scenarios = build_grid(args.seeds)
    os.makedirs(args.out, exist_ok=True)
    index = []
    for sc in scenarios:
        path = os.path.join(args.out, f"{sc.scenario_id}.json")
        with open(path, "w") as f:
            f.write(sc.to_json())
        index.append(sc.scenario_id)
    with open(os.path.join(args.out, "index.json"), "w") as f:
        json.dump(index, f, indent=2)

    print(f"Wrote {len(scenarios)} scenarios to {args.out}/")


if __name__ == "__main__":
    main()
