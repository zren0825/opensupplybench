"""The proposed method: Hybrid LLM replenishment agent (Step 4.3).

Division of labor (the paper's core claim):
  * LLM classifier  -> reads messy context, frames the situation, sets a buffer.
  * Math tools       -> forecast + cost-minimization optimizer do the arithmetic.
  * LLM reviewer     -> sanity-checks the tool's number against the context.

The LLM never computes the order quantity directly (that's what makes LLM-only
unstable); it frames and reviews, while the optimizer does the inventory math.
To stay cost-aware, the LLM is called on a cadence (classify) and only on
significant orders (review) — not every simulated day.
"""

from __future__ import annotations

import numpy as np

from opensupply.policies.base import Policy
from opensupply.policies.cost_min import CostMinimizationPolicy
from opensupply.simulator import Observation
from opensupply.agents.llm_client import LLMClient
from opensupply.agents.classifier import LLMScenarioClassifier
from opensupply.agents.schemas import ReviewDecision
from opensupply.agents.prompts import REVIEWER_SYSTEM, build_review_prompt


class HybridAgent(Policy):
    name = "hybrid"

    def __init__(
        self,
        client=None,
        model: str = "claude-sonnet-5",
        classify_every: int = 14,
        use_reviewer: bool = True,
    ):
        self.llm = client if isinstance(client, LLMClient) else LLMClient(model=model, client=client)
        self.classifier = LLMScenarioClassifier(self.llm)
        self.optimizer = CostMinimizationPolicy()
        self.classify_every = classify_every
        self.use_reviewer = use_reviewer
        self.reset()

    @property
    def usage(self):
        return self.llm.usage

    def reset(self) -> None:
        self._classification = None
        self._last_classify_day = -(10 ** 9)
        self.last_reason = ""

    # --- ablation hooks: flip these off to isolate each module's contribution
    use_classifier: bool = True

    def decide(self, obs: Observation) -> float:
        # 1. LLM classifier (on a cadence, once there is some history)
        if (
            self.use_classifier
            and len(obs.demand_history) >= 7
            and (
                self._classification is None
                or obs.day - self._last_classify_day >= self.classify_every
            )
        ):
            try:
                self._classification = self.classifier.classify(obs)
                self._last_classify_day = obs.day
            except Exception:
                pass  # never let an LLM hiccup break the simulation

        cls = self._classification

        # 2. Tools: cost-minimization optimizer is the numeric backbone.
        base_q = self.optimizer.best_order(obs)
        q = self._apply_classification(base_q, cls)

        # 3. LLM reviewer, only for significant orders (cost-aware gating).
        if self.use_reviewer and q > 0 and self._is_significant(q, obs):
            q = self._review(q, obs, cls)

        return float(max(0, q))

    def _apply_classification(self, base_q: int, cls) -> int:
        if cls is None:
            return base_q
        mult = float(np.clip(cls.safety_multiplier, 0.5, 2.5))
        if cls.supplier_risk == "high":
            mult += 0.25  # unreliable delivery -> carry a little more
        return int(round(base_q * mult))

    def _is_significant(self, q: int, obs: Observation) -> bool:
        """Review orders that are large relative to typical lead-time demand —
        the ones where a mistake is expensive. Keeps reviewer call volume low."""
        hist = obs.demand_history
        mean_d = float(np.mean(hist[-14:])) if len(hist) >= 3 else 20.0
        return q > 1.5 * mean_d * max(1.0, obs.expected_lead)

    def _review(self, proposed_q: int, obs: Observation, cls) -> int:
        try:
            verdict = self.llm.parse(
                REVIEWER_SYSTEM,
                build_review_prompt(obs, proposed_q, cls),
                ReviewDecision,
            )
            self.last_reason = verdict.reason
            return proposed_q if verdict.approved else max(0, verdict.adjusted_order_quantity)
        except Exception:
            return proposed_q
