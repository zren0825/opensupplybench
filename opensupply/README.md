# `opensupply/` — the core simulation library

Everything here models **one small-business replenishment problem** and rolls it
forward in time. If you read one folder to understand the project, read this one.

The subpackages have their own guides:
[policies](policies/README.md) · [agents](agents/README.md) ·
[evaluation](evaluation/README.md).

---

## The problem in one paragraph

A shop sells one product. Each day some customers buy it; when stock runs low the
shop reorders from a supplier, but the order takes several days to arrive (the
**lead time**). Order too little and you stock out and lose the sale (a
**stockout cost**); order too much and cash sits on the shelf (a **holding
cost**). Demand is uncertain and the supplier is sometimes late. The single daily
decision — *order how many, if any?* — is the whole problem.

## The pieces (and the file for each)

| Concept | File | What it is |
|---|---|---|
| Problem spec | `scenario.py` | A `Scenario`: one fully-specified, reproducible problem (costs, constraints, seed). |
| SKU economics | `skus.py` | Samples a **distinct SKU** per scenario (see below). |
| Demand | `demand.py` | Negative-Binomial count demand in 7 recognizable shapes. |
| Supply | `leadtime.py` | Fixed / random / delayed lead-time models. |
| Forecasting | `forecast.py` | `forecast_tool` — a mean + uncertainty band from history. |
| The world | `simulator.py` | The day-by-day loop that scores a policy. |

---

## How a scenario is built

A scenario is three things combined: a **problem spec** (`Scenario`), a **demand
trajectory** (`DemandGenerator`), and a **lead-time behavior** (`LeadTime`). Every
part is seeded, so a scenario is byte-for-byte reproducible.

`experiments/generate_scenarios.py` stamps out the benchmark as a **factorial**:

```
SKU archetype  ×  demand pattern  ×  lead-time regime  ×  seed
   (6 types)        (7 patterns)      (3 regimes)
```

Every SKU archetype is run through **every** demand/lead condition ("season"), so
the benchmark covers all product-type × situation combinations. For a given
(demand, lead, seed) cell all SKUs share the same random seed, so they face the
same underlying season — a paired comparison. This makes **SKU type a controlled
factor** in the analysis (e.g. "does the hybrid help more on high-stockout SKUs
during a viral spike?").

## What makes SKUs different from each other

`skus.py` defines a small set of fixed, interpretable **SKU archetypes** — each a
different kind of product. A single fixed SKU can't represent the real tension
that *some products are cheap to hold but a stockout barely matters, while others
are dear to hold or lose a lot of money per stockout*. The archetypes span that:

| SKU | Profile | Implied service level* |
|---|---|---|
| `staple` | high-volume commodity, cheap to hold, substitutable | ~0.90 |
| `perishable` | spoils fast → dear to hold, moderate stockout | ~0.81 |
| `premium` | high-margin signature item, low volume, stockout hurts | ~0.97 |
| `bulky` | takes shelf space → dear to hold, low stockout penalty | ~0.59 |
| `critical_import` | must-have, slow unreliable supplier, severe stockout | ~0.98 |
| `impulse` | trendy fast-mover, don't over-buy, lumpy demand | ~0.84 |

\* Each SKU's **implied newsvendor service level**
(`stockout / (stockout + holding·review)`) is the cost-optimal target given its
economics; policies use it as their service target. `bulky` (0.59, holding-
dominated — you'd rather stock out than hold) to `critical_import` (0.98,
stockout-dominated) span the tradeoff.

Each archetype fixes: unit cost, margin, holding rate, stockout multiple, order
cost, volume, dispersion, supplier speed, review cadence, and budget. Three
conventions keep the economics honest and interpretable:

- **Holding scales with unit cost** (a per-day fraction of item value).
- **Stockout is tethered to margin** (`stockout_cost = margin × multiple`), not an
  arbitrary number — losing a sale costs at least the profit you'd have made.
- **Budget is measured in *days of supply*** (`base_demand × unit_cost × days`),
  so "tight budget" means the same for a low- and a high-volume SKU. Handily, the
  implied service level is *scale-invariant* (depends on markup, stockout
  multiple, holding rate, review — not on dollar value), so a $3 staple and a
  $40 premium item are both well-posed.

## How demand is generated (`demand.py`)

Demand is **integer counts** from a Negative-Binomial distribution around a
time-varying mean path. A `dispersion` knob sets the overdispersion
(`CV² = 1/µ + 1/r`; NB → Poisson as `r` → ∞). The seven shapes match the LLM
classifier's taxonomy:

`stable` · `seasonal` (weekly) · `promotion_spike` (short bump) ·
`viral_spike` (jump to 5–10× then decay) · `declining` (gradual decay) ·
`new_product` (logistic growth ramp) · `intermittent` (many zero-demand days).

The count draw is separated from the mean path, so any shape can run at any
dispersion and any volume.

## How supply is modeled (`leadtime.py`)

An order placed today arrives after a sampled lead time:
`fixed` (constant), `random` (jitter around the mean), or `delayed` (usually on
time, with a fat tail of late deliveries — the "I ordered it but it showed up
late and I stocked out" case). Demand noise and supply noise use independent
seeded streams.

## The simulator loop (`simulator.py`)

Each day: **receive** arrived orders → **observe** demand → **sell** `min(demand,
on-hand)` (unmet demand is lost) → accrue **holding + stockout** cost → ask the
**policy** to order (arriving after the lead time; a fixed order fee applies). The
scored objective is `total = holding + stockout + ordering` (cost of goods is
excluded — it's demand-determined; the budget is a hard constraint).

`Simulator.run(policy)` is the normal path; it drives a `stepper()` generator that
yields the day's `Observation` and receives the order quantity — the same
generator lets an external driver run many scenarios in lockstep and **batch** the
LLM calls (see [agents](agents/README.md)).

## What a policy sees

Every policy implements one method — `decide(obs) -> order_quantity` — given an
`Observation`: `on_hand`, `on_order`, `inventory_position` (= on-hand +
on-order, the number that actually matters), the demand/sales history so far, the
expected lead time, and the full `Scenario`. This thin interface is why the
classical baselines and the LLM agents are interchangeable inside the simulator.
