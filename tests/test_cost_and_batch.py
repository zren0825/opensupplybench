"""Cost-optimization tests — all offline, no API.

Covers: the stepper refactor stays equivalent to run(); the cost estimator's
arithmetic (batch = half of sync); and the batched lockstep driver produces
byte-identical simulation results to the synchronous path when decisions match.
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from opensupply import Scenario, Simulator
from opensupply.policies import CostMinimizationPolicy
from opensupply.agents import LLMClient, LLMOnlyPolicy
from opensupply.agents.schemas import LLMDecision
from opensupply.agents.cost import estimate, calls_per_scenario


class _ConstClient:
    """Deterministic stub: always orders the same quantity, so batched and
    synchronous drivers must produce identical simulations."""

    def __init__(self, qty=50):
        self.messages = self
        self.qty = qty

    def parse(self, **kwargs):
        out = LLMDecision(order_quantity=self.qty, reason="const")

        class _R:
            parsed_output = out
            usage = {"input_tokens": 300, "output_tokens": 40}
        return _R()


def test_stepper_matches_run():
    # run() drives stepper() internally; a hand-driven stepper must match.
    sc = Scenario(seed=3, horizon_days=45, demand_type="seasonal")
    via_run = Simulator(sc).run(CostMinimizationPolicy())

    policy = CostMinimizationPolicy()
    policy.reset() if hasattr(policy, "reset") else None
    gen = Simulator(sc).stepper()
    obs = next(gen)
    result = None
    try:
        while True:
            obs = gen.send(policy.decide(obs))
    except StopIteration as e:
        result = e.value
    assert abs(result.total_cost - via_run.total_cost) < 1e-9
    assert result.num_orders == via_run.num_orders


def test_cost_estimator_batch_is_half():
    rows_sync, sync = estimate(["llm_only"], 100, horizon=90, batch=False)
    rows_batch, batch = estimate(["llm_only"], 100, horizon=90, batch=True)
    assert rows_sync[0].calls == 100 * 90
    assert abs(batch - 0.5 * sync) < 1e-9


def test_calls_per_scenario():
    assert calls_per_scenario("llm_only", 90) == 90
    assert calls_per_scenario("hybrid", 90) < 90  # far fewer than daily
    assert calls_per_scenario("rule_based", 90) == 0


def test_batched_equals_synchronous():
    from experiments.run_llm_benchmark import run_llm_only_batched

    scenarios = [
        Scenario(scenario_id=f"s{i}", seed=i, horizon_days=30, demand_type=d)
        for i, d in enumerate(["stable", "viral_spike", "intermittent"])
    ]
    # synchronous, one policy per scenario
    sync = [
        Simulator(sc).run(LLMOnlyPolicy(client=_ConstClient(50))) for sc in scenarios
    ]
    # batched lockstep, shared client (parse_batch falls back to sequential
    # parse against the stub, which has no messages.batches)
    llm = LLMClient(client=_ConstClient(50))
    batched = run_llm_only_batched(scenarios, llm)

    for a, b in zip(sync, batched):
        assert abs(a.total_cost - b.total_cost) < 1e-9
        assert a.num_orders == b.num_orders


if __name__ == "__main__":
    test_stepper_matches_run()
    test_cost_estimator_batch_is_half()
    test_calls_per_scenario()
    test_batched_equals_synchronous()
    print("all cost/batch tests passed")
