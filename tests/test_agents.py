"""Agent plumbing tests — no network, no API key.

A stub client stands in for anthropic.Anthropic(), returning canned schema
instances. This verifies the LLM policies drop into the Simulator, honor the
Policy interface, and accumulate usage — without calling the real API.

Run: python tests/test_agents.py   (or via pytest)
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from opensupply import Scenario, Simulator
from opensupply.agents import LLMOnlyPolicy, HybridAgent, LLMClient
from opensupply.agents.schemas import (
    LLMDecision,
    ScenarioClassification,
    ReviewDecision,
)


class _StubResponse:
    def __init__(self, parsed_output):
        self.parsed_output = parsed_output
        self.usage = {"input_tokens": 120, "output_tokens": 30}


class _StubMessages:
    def parse(self, **kwargs):
        schema = kwargs["output_format"]
        if schema is LLMDecision:
            out = LLMDecision(order_quantity=42, reason="stub")
        elif schema is ScenarioClassification:
            out = ScenarioClassification(
                scenario_type="viral_spike",
                demand_uncertainty="high",
                supplier_risk="high",
                recommended_policy="cost_min",
                safety_multiplier=1.8,
                reason="stub",
            )
        elif schema is ReviewDecision:
            out = ReviewDecision(approved=True, adjusted_order_quantity=0, reason="stub")
        else:
            raise AssertionError(f"unexpected schema {schema}")
        return _StubResponse(out)


class _StubClient:
    def __init__(self):
        self.messages = _StubMessages()


def test_llm_only_runs():
    policy = LLMOnlyPolicy(client=_StubClient())
    res = Simulator(Scenario(seed=1, horizon_days=30, demand_type="stable")).run(policy)
    assert res.total_cost >= 0
    assert policy.usage.calls == 30  # one LLM call per day
    assert policy.usage.input_tokens == 30 * 120


def test_hybrid_runs_and_calls_llm_less_than_daily():
    policy = HybridAgent(client=_StubClient(), classify_every=14)
    res = Simulator(Scenario(seed=2, horizon_days=60, demand_type="viral_spike")).run(policy)
    assert res.total_cost >= 0
    # hybrid classifies on a cadence + reviews only significant orders, so it
    # must make far fewer LLM calls than the 60-day LLM-only path would.
    assert 0 < policy.usage.calls < 60


def test_usage_cost_accounting():
    u = LLMClient(model="claude-sonnet-5", client=_StubClient()).usage
    u.record({"input_tokens": 1_000_000, "output_tokens": 0}, 1.0)
    assert abs(u.cost_usd("claude-sonnet-5") - 3.0) < 1e-6  # $3 / 1M input


if __name__ == "__main__":
    test_llm_only_runs()
    test_hybrid_runs_and_calls_llm_less_than_daily()
    test_usage_cost_accounting()
    print("all agent tests passed")
