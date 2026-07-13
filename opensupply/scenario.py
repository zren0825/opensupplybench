"""Scenario definition.

A Scenario is a fully-specified, reproducible replenishment problem for a
single SKU over a fixed horizon. Everything a policy is allowed to see plus
everything the simulator needs to roll the world forward is captured here,
so that `hash(seed + fields)` uniquely determines an experiment.
"""

from __future__ import annotations

from dataclasses import dataclass, asdict, field
from typing import Optional
import json

# Discrete "levels" used when generating the benchmark grid (Phase 5). Keeping
# them as named levels (rather than raw floats) makes scenario breakdown tables
# in the paper readable.
BUDGET_LEVELS = {"low": 200.0, "medium": 1000.0, "high": 5000.0}
STOCKOUT_COST_LEVELS = {"low": 1.0, "medium": 4.0, "high": 12.0}


@dataclass
class Scenario:
    """A single reproducible replenishment problem.

    Cost model (what the simulator scores):
        total_cost = holding + stockout + ordering
    where holding accrues on end-of-day on-hand, stockout is a per-unit penalty
    on lost sales (lost-sales model, not backorder), and ordering is a fixed fee
    per order placed. Cost of goods sold is *excluded* — it is demand-determined
    and identical across policies facing the same demand — while `budget_per_order`
    is a hard operational constraint on a single order's spend.

    The stylized daily `holding_cost` together with the `stockout_cost` levels
    imply a newsvendor service-level target over the review interval; see
    `implied_service_level()`. This makes the parameter choice interpretable
    rather than arbitrary.
    """

    # Identity / reproducibility
    scenario_id: str = "unnamed"
    seed: int = 0
    horizon_days: int = 90

    # Demand + supply structure (see opensupply.demand / opensupply.leadtime)
    demand_type: str = "stable"
    lead_time_type: str = "fixed"
    base_demand: float = 20.0       # mean daily demand scale (SKU volume)
    dispersion: str = "medium"      # NB overdispersion: low | medium | high
    mean_lead: int = 5              # mean supplier lead time (days)
    review_period: int = 7          # planning cadence used by policies / oracle
    service_level: float = 0.95     # target cycle service level for safety-stock policies

    # Economics
    unit_cost: float = 1.0          # $ paid per unit ordered
    selling_price: float = 3.0      # $ earned per unit sold (margin context)
    holding_cost: float = 0.05      # $ per unit per day held at end of day
    stockout_cost: float = 4.0      # $ per unit of lost sales
    order_cost: float = 5.0         # fixed $ per order placed (transaction overhead)

    # Operational constraints
    budget_per_order: float = 1000.0   # max $ spend on a single order
    moq: int = 0                       # minimum order quantity (0 = none)
    case_pack: int = 1                 # orders rounded up to a multiple of this
    initial_inventory: int = 20

    def implied_service_level(self) -> float:
        """Newsvendor critical ratio over the review interval:
        SL* = p / (p + h·R), with p=stockout penalty, h=daily holding, R=review.
        The three benchmark stockout levels span roughly 0.74 / 0.92 / 0.97."""
        denom = self.stockout_cost + self.holding_cost * self.review_period
        return self.stockout_cost / denom if denom > 0 else 0.0

    # Free-text "messy" context the LLM agent (Phase 4) is meant to exploit.
    business_note: str = ""
    supplier_note: str = ""

    # Level tags, retained so scenario-breakdown analysis (Phase 6.2) is trivial.
    budget_level: str = "medium"
    stockout_cost_level: str = "medium"

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict) -> "Scenario":
        known = {f: d[f] for f in cls.__dataclass_fields__ if f in d}
        return cls(**known)

    def to_json(self) -> str:
        return json.dumps(self.to_dict(), indent=2, ensure_ascii=False)

    @classmethod
    def from_json(cls, s: str) -> "Scenario":
        return cls.from_dict(json.loads(s))
