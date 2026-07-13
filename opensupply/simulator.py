"""Single-SKU day-by-day simulator (Phase 2.1).

This is the most important piece of infrastructure in the project. It rolls
one SKU forward for `horizon_days`, letting a policy place orders each day,
and records everything needed to compute the benchmark metrics.

Daily loop (matches the plan):
    receive arrived orders
        -> observe demand
        -> sell min(demand, on_hand); record lost sales
        -> accrue holding + stockout cost
        -> policy decides reorder -> place order (arrival = today + lead time)
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Dict, Any
import numpy as np

from opensupply.scenario import Scenario
from opensupply.demand import DemandGenerator
from opensupply.leadtime import LeadTime


@dataclass
class Observation:
    """Everything a policy is allowed to see on a given day."""

    day: int
    on_hand: int
    on_order: int                 # units in the pipeline, not yet arrived
    sales_history: np.ndarray     # realized (censored) sales up to yesterday
    demand_history: np.ndarray    # true demand up to yesterday (incl. lost sales)
    scenario: Scenario
    expected_lead: float

    @property
    def inventory_position(self) -> int:
        return self.on_hand + self.on_order


@dataclass
class SimulationResult:
    scenario_id: str
    total_cost: float
    holding_cost: float
    stockout_cost: float
    ordering_cost: float
    stockout_days: int
    lost_sales: int
    units_sold: int
    ending_inventory: int
    avg_inventory: float
    num_orders: int
    order_history: List[Dict[str, Any]] = field(default_factory=list)
    daily: List[Dict[str, Any]] = field(default_factory=list)

    def summary(self) -> Dict[str, Any]:
        return {
            "scenario_id": self.scenario_id,
            "total_cost": round(self.total_cost, 2),
            "holding_cost": round(self.holding_cost, 2),
            "stockout_cost": round(self.stockout_cost, 2),
            "ordering_cost": round(self.ordering_cost, 2),
            "stockout_days": self.stockout_days,
            "lost_sales": self.lost_sales,
            "units_sold": self.units_sold,
            "ending_inventory": self.ending_inventory,
            "avg_inventory": round(self.avg_inventory, 2),
            "num_orders": self.num_orders,
        }


class Simulator:
    def __init__(self, scenario: Scenario, mean_lead: int | None = None,
                 base_level: float | None = None):
        # Config lives on the scenario (single source of truth); the optional
        # args are legacy overrides kept for backward compatibility.
        self.scenario = scenario
        self.mean_lead = mean_lead if mean_lead is not None else scenario.mean_lead
        self.demand = DemandGenerator(
            base_level=base_level if base_level is not None else scenario.base_demand
        ).generate(
            scenario.demand_type, scenario.horizon_days, scenario.seed,
            dispersion=scenario.dispersion,
        )
        self.lead_time = LeadTime(scenario.lead_time_type, self.mean_lead, scenario.seed)

    def _round_order(self, qty: float) -> int:
        """Apply MOQ, case pack, and per-order budget constraints."""
        s = self.scenario
        q = max(0, int(round(qty)))
        if q == 0:
            return 0
        if s.moq and q < s.moq:
            q = s.moq
        if s.case_pack > 1:
            q = int(np.ceil(q / s.case_pack) * s.case_pack)
        # budget cap (rounded down to a legal case-pack multiple)
        if s.unit_cost > 0:
            max_by_budget = int(s.budget_per_order // s.unit_cost)
            if q > max_by_budget:
                q = max_by_budget
                if s.case_pack > 1:
                    q = int((q // s.case_pack) * s.case_pack)
        return max(0, q)

    def run(self, policy) -> SimulationResult:
        """Drive `stepper()` with a synchronous policy (the common path)."""
        if hasattr(policy, "reset"):
            policy.reset()
        gen = self.stepper()
        try:
            obs = next(gen)
            while True:
                obs = gen.send(policy.decide(obs))
        except StopIteration as e:
            return e.value

    def stepper(self):
        """Generator form of the day loop, decoupled from decision-making.

        Yields an `Observation` each day and receives that day's raw order
        quantity via `.send(qty)`; returns the `SimulationResult` as the
        `StopIteration.value` when the horizon ends. This lets an external
        driver collect observations across many scenarios and batch the LLM
        calls (Batch API) instead of calling one-at-a-time. `run()` above is
        the trivial synchronous driver, so both paths share identical logic.
        """
        s = self.scenario
        on_hand = s.initial_inventory
        pipeline: List[List[int]] = []  # [arrival_day, qty]

        realized_sales: List[int] = []
        true_demand: List[int] = []
        daily: List[Dict[str, Any]] = []
        order_history: List[Dict[str, Any]] = []

        holding = stockout = ordering = 0.0
        lost_sales = units_sold = stockout_days = 0
        inv_track: List[int] = []

        for day in range(s.horizon_days):
            # 1. receive arrivals
            arrived = 0
            remaining = []
            for arrival_day, qty in pipeline:
                if arrival_day == day:
                    on_hand += qty
                    arrived += qty
                else:
                    remaining.append([arrival_day, qty])
            pipeline = remaining

            # 2. observe demand
            d = int(self.demand[day])

            # 3. sell what we can
            sold = min(d, on_hand)
            lost = d - sold
            on_hand -= sold
            units_sold += sold
            lost_sales += lost
            if lost > 0:
                stockout_days += 1

            # 4. accrue cost on end-of-day state
            holding += s.holding_cost * on_hand
            stockout += s.stockout_cost * lost
            inv_track.append(on_hand)

            realized_sales.append(sold)
            true_demand.append(d)

            # 5. policy decides reorder (sees history up to and including today)
            on_order = sum(q for _, q in pipeline)
            obs = Observation(
                day=day,
                on_hand=on_hand,
                on_order=on_order,
                sales_history=np.array(realized_sales),
                demand_history=np.array(true_demand),
                scenario=s,
                expected_lead=self.lead_time.expected_lead,
            )
            raw_qty = yield obs
            qty = self._round_order(raw_qty)
            if qty > 0:
                ordering += s.order_cost
                lead = self.lead_time.sample()
                arrival = day + lead
                pipeline.append([arrival, qty])
                order_history.append(
                    {"day": day, "qty": qty, "lead": lead, "arrival": arrival}
                )

            daily.append(
                {
                    "day": day,
                    "arrived": arrived,
                    "demand": d,
                    "sold": sold,
                    "lost": lost,
                    "on_hand": on_hand,
                    "on_order": on_order,
                    "ordered": qty,
                }
            )

        total = holding + stockout + ordering
        result = SimulationResult(
            scenario_id=s.scenario_id,
            total_cost=total,
            holding_cost=holding,
            stockout_cost=stockout,
            ordering_cost=ordering,
            stockout_days=stockout_days,
            lost_sales=lost_sales,
            units_sold=units_sold,
            ending_inventory=on_hand,
            avg_inventory=float(np.mean(inv_track)) if inv_track else 0.0,
            num_orders=len(order_history),
            order_history=order_history,
            daily=daily,
        )
        return result
