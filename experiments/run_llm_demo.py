"""Phase 4 demo: LLM-only + hybrid agent vs. the baselines (needs API access).

Runs one scenario through all five methods, prints the cost table plus token
cost / latency for the LLM methods, and probes LLM-only decision *instability*
by asking the model the same question several times (Step 4.1's motivation).

Requires the `anthropic` SDK and credentials (ANTHROPIC_API_KEY or `ant auth
login`). Model defaults to claude-sonnet-5; pass --model claude-opus-4-8 for
the harder-reasoning variant.

Usage:
    python experiments/run_llm_demo.py
    python experiments/run_llm_demo.py --model claude-opus-4-8 --demand viral_spike
"""

from __future__ import annotations

import argparse
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from opensupply import Scenario, Simulator
from opensupply.simulator import Observation
from opensupply.policies import (
    ReorderPointPolicy,
    ForecastSafetyStockPolicy,
    CostMinimizationPolicy,
)
from opensupply.evaluation import metrics_from_result, decision_stability


class _StubClient:
    """Canned client for `--stub`: returns schema-appropriate objects and NEVER
    touches the network. Order quantities vary slightly per call so the offline
    output (and the stability probe) look representative of a real run."""

    def __init__(self):
        self.messages = self
        self._n = 0

    def parse(self, **kwargs):
        from opensupply.agents.schemas import (
            LLMDecision, ScenarioClassification, ReviewDecision,
        )
        self._n += 1
        jitter = [0, 4, -3, 7, -5, 2][self._n % 6]  # deterministic pseudo-noise
        schema = kwargs["output_format"]
        if schema is LLMDecision:
            out = LLMDecision(order_quantity=max(0, 80 + jitter), reason="stub decision")
        elif schema is ScenarioClassification:
            out = ScenarioClassification(
                scenario_type="viral_spike", demand_uncertainty="high",
                supplier_risk="high", recommended_policy="cost_min",
                safety_multiplier=1.6, reason="stub classification")
        else:
            out = ReviewDecision(approved=True, adjusted_order_quantity=0,
                                 reason="stub review")

        class _R:
            parsed_output = out
            usage = {"input_tokens": 420, "output_tokens": 55}
        return _R()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="claude-sonnet-5")
    ap.add_argument("--demand", default="viral_spike")
    ap.add_argument("--lead", default="delayed")
    ap.add_argument("--horizon", type=int, default=60)
    ap.add_argument("--stability-samples", type=int, default=5)
    ap.add_argument("--stub", action="store_true",
                    help="Offline smoke test: use a canned client, make NO API calls.")
    args = ap.parse_args()

    try:
        from opensupply.agents import LLMOnlyPolicy, HybridAgent, LLMClient
    except ImportError as e:
        raise SystemExit(f"Install Phase-4 deps first: pip install anthropic pydantic ({e})")

    inject = _StubClient() if args.stub else None
    if args.stub:
        print("[--stub] Offline mode: no API calls, canned responses.\n")

    scenario = Scenario(
        scenario_id=f"llm_demo_{args.demand}",
        seed=7,
        horizon_days=args.horizon,
        demand_type=args.demand,
        lead_time_type=args.lead,
        stockout_cost=8.0,
        business_note="Small shop; one product is trending on social media.",
        supplier_note="Overseas supplier, occasionally several days late.",
    )
    print(f"Scenario: {scenario.scenario_id} (lead={scenario.lead_time_type}), "
          f"model={args.model}\n")

    non_llm = {
        "rule_based": ReorderPointPolicy(),
        "forecast_safety": ForecastSafetyStockPolicy(),
        "cost_min": CostMinimizationPolicy(),
    }
    llm_only = LLMOnlyPolicy(model=args.model, client=inject)
    hybrid = HybridAgent(model=args.model, client=inject)

    print(f"{'policy':<18}{'total':>9}{'stockout':>10}{'lost':>7}{'orders':>8}"
          f"{'tokens':>9}{'$cost':>8}{'lat(s)':>8}")
    print("-" * 77)

    for name, policy in non_llm.items():
        res = Simulator(scenario).run(policy)
        m = metrics_from_result(res)
        print(f"{name:<18}{m['total_cost']:>9.0f}{m['stockout_cost']:>10.0f}"
              f"{int(m['lost_sales']):>7}{int(m['num_orders']):>8}{'-':>9}{'-':>8}{'-':>8}")

    for name, policy in [("llm_only", llm_only), ("hybrid", hybrid)]:
        res = Simulator(scenario).run(policy)
        m = metrics_from_result(res)
        u = policy.usage.summary(args.model)
        print(f"{name:<18}{m['total_cost']:>9.0f}{m['stockout_cost']:>10.0f}"
              f"{int(m['lost_sales']):>7}{int(m['num_orders']):>8}"
              f"{u['input_tokens'] + u['output_tokens']:>9}{u['est_cost_usd']:>8.3f}"
              f"{u['latency_s']:>8.1f}")

    _stability_probe(scenario, args, inject)


def _stability_probe(scenario, args, inject=None):
    """Ask LLM-only the SAME state N times; report order-quantity variability.
    High variance = the instability that motivates the hybrid design."""
    from opensupply.agents import LLMOnlyPolicy

    demand = Simulator(scenario).demand[:30]
    obs = Observation(
        day=30,
        on_hand=15,
        on_order=0,
        sales_history=np.array(demand),
        demand_history=np.array(demand),
        scenario=scenario,
        expected_lead=5.0,
    )
    qtys = []
    probe = LLMOnlyPolicy(model=args.model, client=inject)
    for _ in range(args.stability_samples):
        qtys.append(probe.decide(obs))
    cv = decision_stability(qtys)
    print(f"\nLLM-only stability probe ({args.stability_samples} identical queries): "
          f"orders={[int(q) for q in qtys]}, "
          f"mean={np.mean(qtys):.1f}, CV={cv:.2f} (lower = more stable)")


if __name__ == "__main__":
    main()
