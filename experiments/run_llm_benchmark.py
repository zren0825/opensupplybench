"""Cost-optimized LLM benchmark sweep (Phase 6.1).

Runs the LLM methods over a stratified subsample (K seeds per condition cell)
and uses the Batch API (50% off) for `llm_only` — the cost driver — by driving
all scenarios in lockstep and batching each simulated day's decisions into one
API request. `hybrid` runs synchronously (it's already cheap: ~6 calls each),
and the deterministic baselines are free.

Safety: defaults to a DRY RUN that prints the cost estimate and exits. Pass
--live to actually call the API (requires ANTHROPIC_API_KEY). Pass --stub for
an offline end-to-end smoke test (no network, canned responses).

Usage:
    # cost preview only, no API calls:
    python experiments/run_llm_benchmark.py --scenarios data/scenarios --seeds 3

    # offline smoke test of the whole pipeline:
    python experiments/run_llm_benchmark.py --scenarios data/scenarios --seeds 1 --stub

    # the real, batched, discounted run (spends tokens):
    python experiments/run_llm_benchmark.py --scenarios data/scenarios --seeds 3 \
        --live --out data/results_llm.csv
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
from opensupply.agents.cost import format_table


def _seed_index(scenario_id: str) -> int:
    tail = scenario_id.rsplit("seed-", 1)[-1]
    try:
        return int(tail)
    except ValueError:
        return 10 ** 9


def load_subsample(path: str, seeds: int):
    """Stratified: keep the first `seeds` seed-indices in every cell."""
    files = sorted(f for f in glob.glob(os.path.join(path, "*.json"))
                   if not f.endswith("index.json"))
    out = []
    for f in files:
        with open(f) as fh:
            sc = Scenario.from_json(fh.read())
        if _seed_index(sc.scenario_id) < seeds:
            out.append(sc)
    return out


# --- batched lockstep driver for llm_only --------------------------------------

def run_llm_only_batched(scenarios, llm):
    """Drive every scenario's stepper in lockstep; batch each day's decisions."""
    from opensupply.agents.prompts import LLM_ONLY_SYSTEM, build_decision_prompt
    from opensupply.agents.schemas import LLMDecision

    gens = [Simulator(sc).stepper() for sc in scenarios]
    results = [None] * len(scenarios)
    obs = [None] * len(scenarios)
    active = []
    for i, g in enumerate(gens):
        try:
            obs[i] = next(g)
            active.append(i)
        except StopIteration as e:
            results[i] = e.value

    while active:
        reqs = [(LLM_ONLY_SYSTEM, build_decision_prompt(obs[i])) for i in active]
        decisions = llm.parse_batch(reqs, LLMDecision)
        nxt = []
        for i, dec in zip(active, decisions):
            qty = float(max(0, dec.order_quantity)) if dec is not None else 0.0
            try:
                obs[i] = gens[i].send(qty)
                nxt.append(i)
            except StopIteration as e:
                results[i] = e.value
        active = nxt
    return results


def _row(scenario, method, res, usage_summary=None):
    m = metrics_from_result(res)
    row = {
        "scenario_id": scenario.scenario_id,
        "policy": method,
        "sku_type": scenario.sku_type,
        "demand_type": scenario.demand_type,
        "lead_time_type": scenario.lead_time_type,
        "budget_level": scenario.budget_level,
        "stockout_cost_level": scenario.stockout_cost_level,
        **{k: round(v, 4) for k, v in m.items()},
        "llm_calls": 0, "input_tokens": 0, "output_tokens": 0, "est_cost_usd": 0.0,
    }
    if usage_summary:
        row.update(
            llm_calls=usage_summary["calls"],
            input_tokens=usage_summary["input_tokens"],
            output_tokens=usage_summary["output_tokens"],
            est_cost_usd=usage_summary["est_cost_usd"],
        )
    return row


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scenarios", default="data/scenarios")
    ap.add_argument("--seeds", type=int, default=3, help="seed-indices per cell")
    ap.add_argument("--out", default="data/results_llm.csv")
    ap.add_argument("--model", default="claude-sonnet-5")
    ap.add_argument("--methods", nargs="+",
                    default=["rule_based", "forecast_safety", "cost_min",
                             "llm_only", "hybrid"])
    ap.add_argument("--live", action="store_true", help="actually call the API")
    ap.add_argument("--stub", action="store_true", help="offline smoke test")
    args = ap.parse_args()

    scenarios = load_subsample(args.scenarios, args.seeds)
    if not scenarios:
        raise SystemExit(f"no scenarios found in {args.scenarios} — run "
                         "generate_scenarios.py first")
    horizon = scenarios[0].horizon_days
    llm_methods = [m for m in args.methods if m in ("llm_only", "hybrid")]

    print(f"Subsample: {len(scenarios)} scenarios ({args.seeds} seed(s)/cell), "
          f"horizon={horizon}\n")
    if llm_methods:
        print(format_table(llm_methods, len(scenarios), args.model, horizon))
        print()

    if llm_methods and not (args.live or args.stub):
        print("DRY RUN — no API calls. Re-run with --live to execute the batched "
              "sweep, or --stub for an offline smoke test.")
        return
    if args.live and not args.stub and not os.environ.get("ANTHROPIC_API_KEY"):
        raise SystemExit("ANTHROPIC_API_KEY not set — cannot run --live.")

    inject = _StubClient() if args.stub else None
    from opensupply.agents import LLMClient, HybridAgent

    rows = []
    # deterministic baselines (free)
    for name in [m for m in args.methods if m in BASELINE_POLICIES]:
        for sc in scenarios:
            rows.append(_row(sc, name, Simulator(sc).run(BASELINE_POLICIES[name]())))

    # llm_only — batched lockstep (Batch API, 50% off)
    if "llm_only" in args.methods:
        llm = LLMClient(model=args.model, client=inject)
        for sc, res in zip(scenarios, run_llm_only_batched(scenarios, llm)):
            rows.append(_row(sc, "llm_only", res, llm.usage.summary(args.model)))
        print(f"llm_only done: {llm.usage.summary(args.model)}")

    # hybrid — synchronous, shared client (cheap)
    if "hybrid" in args.methods:
        llm = LLMClient(model=args.model, client=inject)
        for sc in scenarios:
            res = Simulator(sc).run(HybridAgent(client=llm))
            rows.append(_row(sc, "hybrid", res, llm.usage.summary(args.model)))
        print(f"hybrid done: {llm.usage.summary(args.model)}")

    add_oracle_regret(rows, {sc.scenario_id: sc for sc in scenarios})
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    with open(args.out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    print(f"\nWrote {len(rows)} rows -> {args.out}")


class _StubClient:
    """Canned client for --stub: never touches the network."""

    def __init__(self):
        self.messages = self
        self._n = 0

    def parse(self, **kwargs):
        from opensupply.agents.schemas import (
            LLMDecision, ScenarioClassification, ReviewDecision)
        self._n += 1
        schema = kwargs["output_format"]
        if schema is LLMDecision:
            out = LLMDecision(order_quantity=80 + (self._n % 5), reason="stub")
        elif schema is ScenarioClassification:
            out = ScenarioClassification(
                scenario_type="viral_spike", demand_uncertainty="high",
                supplier_risk="high", recommended_policy="cost_min",
                safety_multiplier=1.5, reason="stub")
        else:
            out = ReviewDecision(approved=True, adjusted_order_quantity=0, reason="stub")

        class _R:
            parsed_output = out
            usage = {"input_tokens": 320, "output_tokens": 40}
        return _R()


if __name__ == "__main__":
    main()
