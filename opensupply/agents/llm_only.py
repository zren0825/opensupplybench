"""Baseline 4: LLM-only replenishment (Step 4.1).

The LLM sees the raw state each day and directly outputs an order quantity.
This is the baseline that (per the plan) usually shows direct LLM decisions are
unstable — run `decision_stability()` over repeated calls to quantify it.
"""

from __future__ import annotations

from opensupply.policies.base import Policy
from opensupply.simulator import Observation
from opensupply.agents.llm_client import LLMClient
from opensupply.agents.schemas import LLMDecision
from opensupply.agents.prompts import LLM_ONLY_SYSTEM, build_decision_prompt


class LLMOnlyPolicy(Policy):
    name = "llm_only"

    def __init__(self, client=None, model: str = "claude-sonnet-5"):
        self.llm = client if isinstance(client, LLMClient) else LLMClient(model=model, client=client)
        self.last_reason: str = ""

    @property
    def usage(self):
        return self.llm.usage

    def decide(self, obs: Observation) -> float:
        decision = self.llm.parse(LLM_ONLY_SYSTEM, build_decision_prompt(obs), LLMDecision)
        self.last_reason = decision.reason
        return float(max(0, decision.order_quantity))
