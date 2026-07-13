"""LLM methods (Phase 4).

    LLMOnlyPolicy         - Step 4.1: LLM emits {order_quantity, reason} directly.
    LLMScenarioClassifier - Step 4.2: classify scenario type + demand/supplier risk.
    HybridAgent           - Step 4.3: classifier -> forecast + cost-min optimizer
                            -> LLM reviewer/explainer (the proposed method).

All policies implement the same `Policy` interface as the Phase-3 baselines, so
they drop straight into `Simulator` and the benchmark harness. `LLMClient`
tracks token cost + latency for the Phase-5.2 metrics.

The `anthropic` SDK is imported lazily inside `LLMClient`, so importing this
module (or the schemas) does not require the SDK or an API key.
"""

from opensupply.agents.llm_client import LLMClient, LLMUsage
from opensupply.agents.schemas import (
    LLMDecision,
    ScenarioClassification,
    ReviewDecision,
    SCENARIO_TYPES,
)
from opensupply.agents.llm_only import LLMOnlyPolicy
from opensupply.agents.classifier import LLMScenarioClassifier
from opensupply.agents.hybrid import HybridAgent

LLM_POLICIES = {
    "llm_only": LLMOnlyPolicy,
    "hybrid": HybridAgent,
}

__all__ = [
    "LLMClient",
    "LLMUsage",
    "LLMDecision",
    "ScenarioClassification",
    "ReviewDecision",
    "SCENARIO_TYPES",
    "LLMOnlyPolicy",
    "LLMScenarioClassifier",
    "HybridAgent",
    "LLM_POLICIES",
]
