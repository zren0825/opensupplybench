"""Fit the tuned (s, S) safety factor on a train split (Phase 6, offline).

Grid-searches the base-stock safety factor on training scenarios (minimizing mean
cost-ratio-to-oracle) and reports the fitted value plus its held-out test
performance — so the classical baseline is genuinely tuned, not a raw formula,
without hindsight overfitting.

Usage:
    python experiments/tune_baselines.py --scenarios data/scenarios
"""

from __future__ import annotations

import argparse
import glob
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from opensupply import Scenario
from opensupply.evaluation.tuning import train_test_split, tune_safety_factor, _mean_cost_ratio


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scenarios", default="data/scenarios")
    args = ap.parse_args()

    files = sorted(f for f in glob.glob(os.path.join(args.scenarios, "*.json"))
                   if not f.endswith("index.json"))
    scenarios = [Scenario.from_json(open(f).read()) for f in files]
    if not scenarios:
        raise SystemExit("no scenarios found — run generate_scenarios.py first")

    train, test = train_test_split(scenarios)
    print(f"scenarios: {len(scenarios)}  (train={len(train)}, test={len(test)})\n")

    best_k, scores = tune_safety_factor(train)
    print("train mean cost/oracle by safety factor k:")
    for k, s in sorted(scores.items()):
        mark = "  <- best" if k == best_k else ""
        print(f"  k={k:<5} -> {s:.3f}{mark}")

    test_ratio = _mean_cost_ratio(test, best_k)
    print(f"\nFitted k = {best_k}; held-out TEST mean cost/oracle = {test_ratio:.3f}")
    print("Use TunedBaseStockPolicy(safety_factor=%s) as the tuned classical baseline." % best_k)


if __name__ == "__main__":
    main()
