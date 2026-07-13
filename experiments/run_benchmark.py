"""Run policies over the benchmark scenario set (Phase 6.1).

Loads scenarios from a directory (see generate_scenarios.py), runs each selected
policy on each scenario, and writes a tidy results CSV with per-scenario metrics,
regret, and (for LLM methods) token cost. This is the harness that produces the
main experiment table.

Methods:
    baselines  (free, default) - rule_based, forecast_safety, cost_min
    llm                        - llm_only, hybrid          (calls the Claude API)
    all                        - baselines + llm

The LLM methods cost real API tokens, so they are gated: selecting them prints a
cost estimate and refuses to run unless you pass --live, and a scenario cap is
enforced unless you pass --force-full. Baseline runs need neither the anthropic
SDK nor an API key.

Usage:
    # free baseline run over the full grid
    python experiments/run_benchmark.py --scenarios data/scenarios --out data/results.csv

    # estimate the cost of an LLM run without calling anything
    python experiments/run_benchmark.py --methods all --limit 50

    # actually run the LLM methods (spends tokens)
    python experiments/run_benchmark.py --methods all --limit 50 --live \
        --model claude-sonnet-5 --out data/results_llm.csv
"""

from __future__ import annotations

import argparse
import csv
import glob
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from opensupply import Scenario, Simulator
from opensupply.policies import BASELINE_POLICIES
from opensupply.evaluation import metrics_from_result, add_oracle_regret

LLM_METHOD_NAMES = ("llm_only", "hybrid")
# Usage columns present on every row (0 for the deterministic baselines).
USAGE_FIELDS = ("llm_calls", "input_tokens", "output_tokens", "est_cost_usd")

# Rough per-call token sizes for the cost *estimate* only (actuals are logged
# per row from real usage). Input grows with history; output is a short JSON.
_EST_INPUT_TOKENS = 1500
_EST_OUTPUT_TOKENS = 120
# Rough LLM calls per 90-day scenario: llm_only calls every day; hybrid classifies
# on a cadence and reviews only significant orders (see hybrid.py).
_EST_CALLS_PER_SCENARIO = {"llm_only": 90, "hybrid": 15}
# Mirror of the pricing table in agents/llm_client.py ($/1M input, $/1M output).
_PRICING = {
    "claude-sonnet-5": (3.0, 15.0),
    "claude-opus-4-8": (5.0, 25.0),
    "claude-haiku-4-5": (1.0, 5.0),
}


def load_scenarios(path: str, limit: int | None = None):
    if not os.path.isdir(path):
        raise SystemExit(f"scenario path not found: {path}")
    files = sorted(glob.glob(os.path.join(path, "*.json")))
    files = [f for f in files if not f.endswith("index.json")]
    if limit is not None:
        files = files[:limit]
    for f in files:
        with open(f) as fh:
            yield Scenario.from_json(fh.read())


def _count_scenarios(path: str, limit: int | None) -> int:
    files = [f for f in glob.glob(os.path.join(path, "*.json")) if not f.endswith("index.json")]
    n = len(files)
    return min(n, limit) if limit is not None else n


def resolve_methods(methods: str) -> list[str]:
    if methods == "baselines":
        return list(BASELINE_POLICIES)
    if methods == "llm":
        return list(LLM_METHOD_NAMES)
    if methods == "all":
        return list(BASELINE_POLICIES) + list(LLM_METHOD_NAMES)
    raise SystemExit(f"unknown --methods value: {methods}")


def _load_llm_policies():
    """Import LLM policies lazily so baseline runs need no SDK / pydantic."""
    try:
        from opensupply.agents import LLM_POLICIES
    except ImportError as e:  # pragma: no cover - depends on optional deps
        raise SystemExit(
            f"LLM methods require the Phase-4 dependencies (anthropic, pydantic): {e}\n"
            f"Install them with: pip install -r requirements.txt"
        )
    return LLM_POLICIES


def build_policies(method_names, model: str):
    """Fresh instance per run so RNG state and LLM usage counters reset."""
    llm_policies = _load_llm_policies() if any(m in LLM_METHOD_NAMES for m in method_names) else {}
    out = {}
    for name in method_names:
        if name in BASELINE_POLICIES:
            out[name] = BASELINE_POLICIES[name]()
        elif name in LLM_METHOD_NAMES:
            out[name] = llm_policies[name](model=model)
        else:
            raise SystemExit(f"unknown method: {name}")
    return out


def estimate_cost(method_names, n_scenarios: int, model: str) -> float:
    in_price, out_price = _PRICING.get(model, (3.0, 15.0))
    per_call = (_EST_INPUT_TOKENS * in_price + _EST_OUTPUT_TOKENS * out_price) / 1e6
    total = 0.0
    for name in method_names:
        if name in LLM_METHOD_NAMES:
            total += n_scenarios * _EST_CALLS_PER_SCENARIO[name] * per_call
    return total


def _usage_row(policy, model: str) -> dict:
    u = getattr(policy, "usage", None)
    if u is None:
        return {k: 0 for k in USAGE_FIELDS}
    return {
        "llm_calls": u.calls,
        "input_tokens": u.input_tokens,
        "output_tokens": u.output_tokens,
        "est_cost_usd": round(u.cost_usd(model), 6),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scenarios", default="data/scenarios")
    ap.add_argument("--out", default="data/results.csv")
    ap.add_argument(
        "--methods",
        choices=["baselines", "llm", "all"],
        default="baselines",
        help="which methods to run (default: baselines, which are free)",
    )
    ap.add_argument(
        "--model",
        default="claude-sonnet-5",
        help="Claude model for LLM methods (e.g. claude-sonnet-5, claude-opus-4-8)",
    )
    ap.add_argument("--limit", type=int, default=None, help="cap number of scenarios")
    ap.add_argument(
        "--live",
        action="store_true",
        help="actually call the Claude API for LLM methods (spends tokens). "
        "Without this, LLM methods only print a cost estimate and exit.",
    )
    ap.add_argument(
        "--force-full",
        action="store_true",
        help="allow a live LLM run over more than the safety cap of scenarios",
    )
    args = ap.parse_args()

    method_names = resolve_methods(args.methods)
    has_llm = any(m in LLM_METHOD_NAMES for m in method_names)
    n_scenarios = _count_scenarios(args.scenarios, args.limit)

    # --- Cost gate for LLM methods ------------------------------------------
    if has_llm:
        est = estimate_cost(method_names, n_scenarios, args.model)
        print(
            f"LLM methods selected: {[m for m in method_names if m in LLM_METHOD_NAMES]}\n"
            f"  model={args.model}  scenarios={n_scenarios}\n"
            f"  ROUGH estimated cost: ${est:,.2f} "
            f"(~{_EST_INPUT_TOKENS} in / {_EST_OUTPUT_TOKENS} out tokens per call; "
            f"caching/Batch API not applied)"
        )
        if not args.live:
            print(
                "\nNo API calls made (dry run). Re-run with --live to actually call "
                "the Claude API.\nTip: keep --limit small first, and consider "
                "--model claude-haiku-4-5 for a cheap smoke test."
            )
            return
        if not os.environ.get("ANTHROPIC_API_KEY"):
            raise SystemExit(
                "ANTHROPIC_API_KEY is not set — cannot run LLM methods live. "
                "Set it, or drop --live for a dry run."
            )
        SAFETY_CAP = 100
        if n_scenarios > SAFETY_CAP and not args.force_full:
            raise SystemExit(
                f"Refusing to run LLM methods live over {n_scenarios} scenarios "
                f"(> safety cap of {SAFETY_CAP}). Add --limit N to shrink the run, "
                f"or --force-full to override (estimated ${est:,.2f})."
            )
        print("\n--live set: calling the Claude API now.\n")

    # --- Run -----------------------------------------------------------------
    rows = []
    scenarios_by_id = {}
    n = 0
    for scenario in load_scenarios(args.scenarios, args.limit):
        n += 1
        scenarios_by_id[scenario.scenario_id] = scenario
        for pname, policy in build_policies(method_names, args.model).items():
            res = Simulator(scenario).run(policy)
            m = metrics_from_result(res)
            rows.append(
                {
                    "scenario_id": scenario.scenario_id,
                    "policy": pname,
                    "demand_type": scenario.demand_type,
                    "lead_time_type": scenario.lead_time_type,
                    "budget_level": scenario.budget_level,
                    "stockout_cost_level": scenario.stockout_cost_level,
                    **{k: round(v, 4) for k, v in m.items()},
                    **_usage_row(policy, args.model),
                }
            )
        if n % 50 == 0:
            print(f"  ...{n} scenarios")

    add_oracle_regret(rows, scenarios_by_id)  # regret vs. clairvoyant reference

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    fieldnames = list(rows[0].keys())
    with open(args.out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        w.writerows(rows)

    print(f"\nRan {n} scenarios x {len(method_names)} methods "
          f"= {len(rows)} rows -> {args.out}")
    _print_summary(rows)


def _print_summary(rows):
    """Mean total_cost, regret, and token cost per policy — a preview of the main table."""
    from collections import defaultdict

    agg = defaultdict(lambda: {"total": 0.0, "regret": 0.0, "cost": 0.0, "n": 0})
    for r in rows:
        a = agg[r["policy"]]
        a["total"] += r["total_cost"]
        a["regret"] += r["regret"]
        a["cost"] += r.get("est_cost_usd", 0.0)
        a["n"] += 1
    print(f"\n{'policy':<18}{'mean_total_cost':>18}{'mean_regret':>14}{'total_$':>12}")
    print("-" * 62)
    for policy, a in sorted(agg.items(), key=lambda kv: kv[1]["total"] / kv[1]["n"]):
        print(f"{policy:<18}{a['total']/a['n']:>18.2f}{a['regret']/a['n']:>14.2f}"
              f"{a['cost']:>12.4f}")


if __name__ == "__main__":
    main()
