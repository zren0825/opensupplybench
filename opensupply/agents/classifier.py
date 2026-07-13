"""LLM scenario classifier (Step 4.2) — the hybrid agent's first module.

Turns the messy state into a structured read: scenario type, demand
uncertainty, supplier risk, a recommended policy family, and a safety
multiplier the optimizer's output is scaled by. This is where the LLM adds
value the pure optimizer can't — reading business/supplier notes and trend.
"""

from __future__ import annotations

from opensupply.simulator import Observation
from opensupply.agents.llm_client import LLMClient
from opensupply.agents.schemas import ScenarioClassification
from opensupply.agents.prompts import CLASSIFIER_SYSTEM, build_classifier_prompt


class LLMScenarioClassifier:
    def __init__(self, llm: LLMClient):
        self.llm = llm

    def classify(self, obs: Observation) -> ScenarioClassification:
        return self.llm.parse(
            CLASSIFIER_SYSTEM, build_classifier_prompt(obs), ScenarioClassification
        )
