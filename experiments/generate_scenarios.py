"""Generate the reproducible benchmark scenario grid (Phase 5.1, v0.4).

Full factorial:

    SKU archetype  ×  demand pattern  ×  lead-time regime  ×  seed

A small fixed set of interpretable SKU types (`opensupply.skus.SKU_ARCHETYPES`)
is each run through **every** demand/lead condition ("season"), so the benchmark
covers all combinations of product-type × situation. Every SKU faces the same
conditions, which makes SKU type a controlled factor in the analysis (e.g. "does
the hybrid help more on high-stockout SKUs during a viral spike?"). For a given
(demand, lead, seed) cell, all SKUs share the same random seed, so they face the
same underlying "season" (a paired comparison across SKUs).

With 6 archetypes and 25 seeds this yields 6 x 7 x 3 x 25 = 3150 scenarios.

Usage:
    python experiments/generate_scenarios.py --seeds 25 --out data/scenarios
    python experiments/generate_scenarios.py --seeds 2             # quick smoke set
"""

from __future__ import annotations

import argparse
import glob
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from opensupply.scenario import Scenario
from opensupply.demand import DEMAND_TYPES
from opensupply.leadtime import LEAD_TIME_TYPES
from opensupply.skus import SKU_ARCHETYPES


def build_grid(seeds: int):
    scenarios = []
    n_lead = len(LEAD_TIME_TYPES)
    for di, demand in enumerate(DEMAND_TYPES):
        for li, lead in enumerate(LEAD_TIME_TYPES):
            for k in range(seeds):
                # One seed per (demand, lead, k) condition, shared across SKUs so
                # every SKU faces the same "season" (paired comparison).
                cond_seed = 1000 + ((di * n_lead + li) * seeds + k)
                for sku in SKU_ARCHETYPES:
                    sid = f"{sku.name}__{demand}__{lead}__seed-{k}"
                    scenarios.append(
                        Scenario(
                            scenario_id=sid,
                            seed=cond_seed,
                            horizon_days=90,
                            demand_type=demand,
                            lead_time_type=lead,
                            **sku.economics(),
                        )
                    )
    return scenarios


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, default=25)
    ap.add_argument("--out", default="data/scenarios")
    args = ap.parse_args()

    scenarios = build_grid(args.seeds)
    os.makedirs(args.out, exist_ok=True)
    # Clear any stale scenarios so different benchmark versions never mix in the
    # same directory (they would silently pollute run_benchmark's results).
    removed = 0
    for f in glob.glob(os.path.join(args.out, "*.json")):
        os.remove(f)
        removed += 1
    if removed:
        print(f"Cleared {removed} existing scenario files from {args.out}/")
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
