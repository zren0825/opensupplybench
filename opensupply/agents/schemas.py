"""Structured-output schemas for the LLM methods (Phase 4).

These Pydantic models are passed to `client.messages.parse(...)` so the API
constrains Claude's response to valid JSON — no brittle string parsing, and
the model is forced to emit exactly the fields each stage needs.
"""

from __future__ import annotations

from typing import Literal
from pydantic import BaseModel, Field

# Scenario taxonomy the classifier chooses from (Step 4.2). Broader than the
# demand generator's set on purpose — the LLM sees messy real-ish context and
# may recognize "new_product" / "declining" that the synthetic labels lump in.
SCENARIO_TYPES = [
    "stable",
    "seasonal",
    "promotion_spike",
    "viral_spike",
    "declining",
    "new_product",
    "intermittent",
]

RiskLevel = Literal["low", "medium", "high"]
RecommendedPolicy = Literal["rule_based", "forecast_safety", "cost_min"]


class LLMDecision(BaseModel):
    """Step 4.1 — direct LLM replenishment decision."""

    order_quantity: int = Field(description="Units to order today. 0 = do not order.")
    reason: str = Field(description="One or two sentences justifying the quantity.")


class ScenarioClassification(BaseModel):
    """Step 4.2 — the hybrid agent's first module."""

    scenario_type: Literal[
        "stable",
        "seasonal",
        "promotion_spike",
        "viral_spike",
        "declining",
        "new_product",
        "intermittent",
    ]
    demand_uncertainty: RiskLevel
    supplier_risk: RiskLevel
    recommended_policy: RecommendedPolicy
    safety_multiplier: float = Field(
        description="How much to scale the tool's baseline order to hedge this "
        "situation. 1.0 = trust the optimizer, >1 = carry more buffer "
        "(spikes, unreliable supplier), <1 = trim (declining demand). "
        "Keep within roughly 0.5-2.5."
    )
    reason: str


class ReviewDecision(BaseModel):
    """Step 4.3 — the LLM reviewer's verdict on a tool-computed order."""

    approved: bool = Field(description="True if the proposed quantity is sound.")
    adjusted_order_quantity: int = Field(
        description="Final quantity. Equal to the proposal when approved; a "
        "corrected value when not."
    )
    reason: str
