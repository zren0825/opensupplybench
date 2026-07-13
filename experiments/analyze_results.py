"""Analyze a benchmark results CSV (Phase 6.1/6.2) — offline, no API.

Reports the honest headline the reviewers care about:
  * scale-normalized cost (cost ÷ clairvoyant oracle), so a $12k viral scenario
    and a $500 stable one count comparably;
  * under BOTH uniform and documented-prevalence weighting;
  * regret vs. the Virtual Best Classical (per-scenario best classical policy) —
    does the hybrid add value beyond picking the right classical policy?
  * a per-demand-type breakdown (where each method actually wins).

Usage:
    python experiments/analyze_results.py --results data/results.csv
"""

from __future__ import annotations

import argparse
import csv
import os
import sys
from collections import defaultdict

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from opensupply.policies import CLASSICAL_FOR_VBC
from opensupply.evaluation import add_vbc_regret
from opensupply.evaluation.weighting import cost_ratio, weighted_metric

_NUM = ("total_cost", "oracle_cost", "regret", "fill_rate")


def load_rows(path: str):
    with open(path) as f:
        rows = list(csv.DictReader(f))
    if not rows or "oracle_cost" not in rows[0]:
        raise SystemExit(
            "results CSV missing 'oracle_cost' — regenerate with the current "
            "run_benchmark.py (it stores oracle_cost via add_oracle_regret)."
        )
    for r in rows:
        for k in _NUM:
            if k in r and r[k] != "":
                r[k] = float(r[k])
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", default="data/results.csv")
    args = ap.parse_args()

    rows = load_rows(args.results)
    classical = [p for p in CLASSICAL_FOR_VBC if any(r["policy"] == p for r in rows)]
    add_vbc_regret(rows, classical)

    by_policy = defaultdict(list)
    for r in rows:
        by_policy[r["policy"]].append(r)

    print(f"Policies: {list(by_policy)}   |   classical (VBC pool): {classical}\n")
    print(f"{'policy':<16}{'cost/oracle':>13}{'cost/oracle':>13}{'fill':>7}"
          f"{'beats_VBC':>11}")
    print(f"{'':<16}{'(uniform)':>13}{'(prevalence)':>13}{'':>7}{'(share)':>11}")
    print("-" * 60)
    # order by uniform normalized cost (best first)
    order = sorted(by_policy, key=lambda p: weighted_metric(by_policy[p], cost_ratio, "uniform"))
    for p in order:
        pr = by_policy[p]
        uni = weighted_metric(pr, cost_ratio, "uniform")
        prev = weighted_metric(pr, cost_ratio, "prevalence")
        fill = weighted_metric(pr, lambda r: r["fill_rate"], "uniform")
        beat = sum(r.get("beats_vbc", False) for r in pr) / len(pr)
        beat_str = f"{beat:>10.1%}" if p not in classical else f"{'—':>10}"
        print(f"{p:<16}{uni:>13.3f}{prev:>13.3f}{fill:>7.1%}{beat_str:>11}")

    _breakdown(by_policy, rows, "demand_type")
    if any("sku_type" in r for r in rows):
        _breakdown(by_policy, rows, "sku_type")


def _breakdown(by_policy, rows, key):
    """Mean normalized cost (cost/oracle) per policy within each level of `key`."""
    levels = sorted({r[key] for r in rows if r.get(key)})
    print(f"\nNormalized cost (cost/oracle) by {key}:")
    header = f"{'policy':<16}" + "".join(f"{str(t)[:10]:>12}" for t in levels)
    print(header)
    print("-" * len(header))
    for p, pr in sorted(by_policy.items()):
        cells = []
        for t in levels:
            sub = [r for r in pr if r.get(key) == t]
            cells.append(weighted_metric(sub, cost_ratio, "uniform") if sub else float("nan"))
        print(f"{p:<16}" + "".join(f"{c:>12.2f}" for c in cells))


if __name__ == "__main__":
    main()
