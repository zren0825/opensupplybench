"""SKU archetypes (v0.4).

A small set of **fixed, interpretable SKU types**, each representing a different
kind of product with a distinct cost/operational profile. The benchmark runs
*every* SKU through *every* demand/lead condition ("season"), so it is a clean
factorial:

    SKU archetype  ×  demand pattern  ×  lead-time regime  ×  seed

This is what lets the analysis ask questions like "does the hybrid agent help
more on high-stockout SKUs during a viral spike?" — SKU *type* is a controlled
factor, not entangled with the scenario.

The key economic tension the archetypes span: **some products are cheap to hold
but a stockout barely matters (commodity), others are dear to hold (perishable /
bulky), others lose a lot of money per stockout (premium / critical).** With a
single fixed SKU that balance is impossible to represent.

Economic conventions (v0.3 fixes, kept):
  * holding_cost = unit_cost × holding_rate  (a per-day fraction of item value)
  * stockout_cost = (unit margin) × stockout_multiple  (tethered to real margin)
  * budget_per_order = base_demand × unit_cost × days_of_supply  (days of supply)
  * each SKU's implied newsvendor service level = stockout/(stockout+holding·R)
    is used as its policies' cost-aware target. It is scale-invariant (depends on
    markup, stockout multiple, holding rate, review — not on item dollar value).

All values are stylized (not calibrated to a dataset) but chosen to be
interpretable and to span the holding-vs-stockout tradeoff.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List


def _bin(value, low_hi, med_hi):
    return "low" if value < low_hi else ("medium" if value < med_hi else "high")


@dataclass(frozen=True)
class SKU:
    name: str
    description: str
    unit_cost: float          # $ per unit
    markup: float             # unit margin = unit_cost × markup
    holding_rate: float       # holding cost / unit / day, as fraction of unit_cost
    stockout_multiple: float  # stockout penalty = margin × multiple (goodwill if >1)
    order_cost: float         # fixed $ per purchase order
    base_demand: float        # mean units / day (SKU volume)
    dispersion: str           # inherent demand lumpiness: low | medium | high
    mean_lead: int            # supplier speed (days)
    review_period: int        # ordering cadence (days)
    days_of_supply: int       # per-order budget, in days of demand

    def economics(self) -> dict:
        margin = self.unit_cost * self.markup
        holding = round(self.unit_cost * self.holding_rate, 4)
        stockout = round(margin * self.stockout_multiple, 4)
        implied_sl = stockout / (stockout + holding * self.review_period)
        return dict(
            sku_type=self.name,
            unit_cost=self.unit_cost,
            selling_price=round(self.unit_cost + margin, 2),
            holding_cost=holding,
            stockout_cost=stockout,
            order_cost=self.order_cost,
            mean_lead=self.mean_lead,
            review_period=self.review_period,
            base_demand=self.base_demand,
            dispersion=self.dispersion,
            days_of_supply=self.days_of_supply,
            budget_per_order=round(self.base_demand * self.unit_cost * self.days_of_supply, 2),
            service_level=round(min(max(implied_sl, 0.50), 0.99), 4),
            initial_inventory=int(self.base_demand),
            holding_level=_bin(self.holding_rate, 0.010, 0.018),
            stockout_cost_level=_bin(self.stockout_multiple, 1.5, 2.75),
            budget_level=_bin(self.days_of_supply, 5, 20),
        )


# Six archetypes spanning the holding↔stockout tradeoff. The implied service
# level (shown) ranges ≈0.59–0.98: bulky/perishable are holding-dominated (you'd
# rather stock out than hold), premium/critical are stockout-dominated.
SKU_ARCHETYPES: List[SKU] = [
    SKU("staple", "High-volume commodity: cheap to hold, easily substituted.",
        unit_cost=3.0, markup=0.40, holding_rate=0.006, stockout_multiple=1.0,
        order_cost=5.0, base_demand=120, dispersion="low", mean_lead=4,
        review_period=7, days_of_supply=14),            # implied SL ≈ 0.90
    SKU("perishable", "Spoils fast: expensive to hold, moderate stockout cost.",
        unit_cost=6.0, markup=0.60, holding_rate=0.030, stockout_multiple=1.5,
        order_cost=4.0, base_demand=60, dispersion="medium", mean_lead=3,
        review_period=7, days_of_supply=5),             # implied SL ≈ 0.81
    SKU("premium", "High-margin signature item: low volume, stockout hurts.",
        unit_cost=40.0, markup=1.20, holding_rate=0.008, stockout_multiple=3.0,
        order_cost=10.0, base_demand=12, dispersion="medium", mean_lead=7,
        review_period=14, days_of_supply=20),           # implied SL ≈ 0.97
    SKU("bulky", "Takes shelf space: dear to hold, low stockout penalty.",
        unit_cost=15.0, markup=0.50, holding_rate=0.025, stockout_multiple=1.0,
        order_cost=8.0, base_demand=25, dispersion="low", mean_lead=5,
        review_period=14, days_of_supply=10),           # implied SL ≈ 0.59
    SKU("critical_import", "Must-have, slow unreliable supplier, severe stockout.",
        unit_cost=20.0, markup=0.90, holding_rate=0.010, stockout_multiple=4.0,
        order_cost=10.0, base_demand=20, dispersion="high", mean_lead=10,
        review_period=7, days_of_supply=20),            # implied SL ≈ 0.98
    SKU("impulse", "Trendy fast-mover: don't over-buy (goes out of fashion), lumpy.",
        unit_cost=8.0, markup=1.00, holding_rate=0.020, stockout_multiple=0.75,
        order_cost=5.0, base_demand=40, dispersion="high", mean_lead=5,
        review_period=7, days_of_supply=7),             # implied SL ≈ 0.84
]

SKU_BY_NAME = {s.name: s for s in SKU_ARCHETYPES}
