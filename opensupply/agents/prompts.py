"""Prompt construction for the LLM methods (Phase 4).

One place to turn a simulator Observation into the natural-language state
summary the LLM sees. Keeping demand history compact (recent window + summary
stats) controls token cost while preserving the signal the model needs.
"""

from __future__ import annotations

import numpy as np

from opensupply.simulator import Observation


def summarize_state(obs: Observation, recent_days: int = 14) -> str:
    s = obs.scenario
    hist = obs.demand_history
    recent = hist[-recent_days:] if len(hist) else hist
    recent_str = ", ".join(str(int(x)) for x in recent) if len(recent) else "(none yet)"

    if len(hist) >= 2:
        stats = (
            f"mean/day={np.mean(hist):.1f}, std={np.std(hist):.1f}, "
            f"last-7 mean={np.mean(hist[-7:]):.1f}, max={int(np.max(hist))}"
        )
    else:
        stats = "insufficient history"

    return (
        f"Day {obs.day} of a single-SKU replenishment simulation.\n"
        f"On hand: {obs.on_hand} units. In transit (on order): {obs.on_order}. "
        f"Inventory position: {obs.inventory_position}.\n"
        f"Expected supplier lead time: ~{obs.expected_lead:.1f} days "
        f"(type: {s.lead_time_type}).\n"
        f"Recent daily demand (oldest->newest): {recent_str}\n"
        f"Demand stats: {stats}\n"
        f"Economics: unit cost ${s.unit_cost:.2f}, holding ${s.holding_cost:.3f}/unit/day, "
        f"stockout penalty ${s.stockout_cost:.2f}/unit lost, order fee ${s.order_cost:.2f}.\n"
        f"Constraints: budget ${s.budget_per_order:.0f}/order, MOQ {s.moq}, "
        f"case pack {s.case_pack}.\n"
        f"Business note: {s.business_note or '(none)'}\n"
        f"Supplier note: {s.supplier_note or '(none)'}"
    )


LLM_ONLY_SYSTEM = (
    "You are an inventory replenishment agent for a small business. Each day you "
    "decide how many units of a single product to order from the supplier. Ordering "
    "too much wastes money on holding cost; ordering too little (or too late, given "
    "lead time) loses sales at the stockout penalty. Weigh the two and return an "
    "order quantity for today (0 to skip). Be decisive and numerically careful."
)


def build_decision_prompt(obs: Observation) -> str:
    return summarize_state(obs) + (
        "\n\nDecide today's order quantity (integer units, 0 if none). Keep the "
        "reason under 12 words."
    )


CLASSIFIER_SYSTEM = (
    "You are a demand/supply analyst for small-business inventory. Given a product's "
    "recent sales history and context, classify the situation so a downstream "
    "optimizer can size the replenishment order. Focus on: what kind of demand "
    "pattern this is, how uncertain near-term demand is, how risky the supplier's "
    "lead time is, which policy family fits, and how much safety buffer to carry "
    "relative to a cost-optimal baseline."
)


def build_classifier_prompt(obs: Observation) -> str:
    return summarize_state(obs) + (
        "\n\nClassify this SKU's current situation. safety_multiplier scales the "
        "optimizer's cost-minimal order: use >1 when a spike or unreliable supplier "
        "means a stockout is likely and costly, <1 when demand is fading and "
        "overstock is the bigger risk."
    )


REVIEWER_SYSTEM = (
    "You are a decision reviewer for an inventory replenishment agent. A numerical "
    "optimizer has proposed an order quantity. Sanity-check it against the situation "
    "and the messy business/supplier context the optimizer cannot read. Approve it if "
    "sound; otherwise return a corrected quantity. Do not micro-tune — only override "
    "when the proposal is clearly wrong given the context."
)


def build_review_prompt(obs: Observation, proposed_qty: int, classification) -> str:
    cls = ""
    if classification is not None:
        cls = (
            f"\nAnalyst read: scenario={classification.scenario_type}, "
            f"demand_uncertainty={classification.demand_uncertainty}, "
            f"supplier_risk={classification.supplier_risk}, "
            f"recommended_policy={classification.recommended_policy}."
        )
    return (
        summarize_state(obs)
        + cls
        + f"\n\nThe optimizer proposes ordering {proposed_qty} units today. "
        "Approve or correct it."
    )
