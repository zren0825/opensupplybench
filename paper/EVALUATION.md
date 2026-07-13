# Research-Readiness Review of OpenSupplyBench (v0.1 → v0.2)

A self-critical audit of the benchmark's demand model, baselines, economics, and
single-SKU scope against what a workshop / applied-AI reviewer would expect, plus
the concrete changes made to reach a defensible level. Written to be honest about
what is fixed and what remains explicit scope.

## Verdict on v0.1

The v0.1 scaffold *runs and is reproducible*, but three things would have drawn
reviewer fire, and one is a genuine scientific-validity problem:

| # | Issue | Severity | Status |
|---|-------|----------|--------|
| 1 | **Baselines are strawmen.** `rule_based` carried *zero* safety stock (≈50% cycle service by construction, 78% fill); `forecast_safety` was a base-stock policy with **no reorder trigger**, so it re-ordered almost daily (45 orders/90d) and was dominated by the fixed order fee rather than by inventory logic. Beating weak baselines proves little. | **High** | Fixed |
| 2 | **Demand was Gaussian, not count data.** `Normal(µ, 0.15µ)` with a fixed CV, clipped and rounded. Real retail demand is counts with overdispersion; safety-stock theory is distributional. No way to vary dispersion (a real experimental axis). | **High** | Fixed |
| 3 | **Generator/classifier taxonomy mismatch.** The LLM classifier could output `declining` / `new_product`, but the generator produced neither (`sudden_drop` instead, no growth ramp). | Medium | Fixed |
| 4 | **Un-interpretable economics.** `holding_cost = 0.05/unit/day` reads as ~1825%/yr if taken literally; magnitudes were magic numbers with no stated implied service level. | Medium | Documented + reframed |
| 5 | **Weak regret baseline.** Regret was measured vs. the best *evaluated* policy — circular, and it flatters whichever method happens to win. No oracle / lower bound. | Medium | Fixed (clairvoyant oracle) |
| 6 | **No achieved-service-level metric.** Cost alone hides *how* a method fails (overstock vs. stockout). Fill rate is the standard service KPI. | Medium | Fixed |
| 7 | **Single hidden scale.** Every SKU had `base_demand = 20`; no low-volume/high-volume heterogeneity, where intermittency and integer effects differ most. | Low–Med | Fixed (scale + dispersion randomized per scenario) |

## What changed in v0.2

### Demand (`opensupply/demand.py`)
- **Negative-Binomial counts** with a `dispersion` knob (`low`/`medium`/`high` →
  overdispersion parameter *r*). NB→Poisson as *r*→∞, so this spans the
  Poisson-to-overdispersed regimes retail actually exhibits. `CV² = 1/µ + 1/r`.
- **Taxonomy aligned to the classifier** (7 types): `stable`, `seasonal`,
  `promotion_spike`, `viral_spike`, `declining` (gradual exponential decay),
  `new_product` (logistic growth ramp), `intermittent` (zero-inflated NB).
- Mean *paths* are separated from the count draw, so any shape can be run at any
  dispersion and any scale.

### Baselines (`opensupply/policies/`)
Now four textbook-correct policies, each matched to the fixed-order-cost economics
(they order on a trigger/cadence, not daily):
- `rule_based` — continuous-review **(s, S)** with a modest safety factor.
- `forecast_safety` — **(s, S)** with the reorder point and order-up-to level
  derived from an exponential-smoothing forecast and a service-level *z*
  (`safety = z·σ·√L`). Now has a real reorder trigger → no daily dribble.
- `periodic_review` — **(R, S)** base-stock, ordering only every *R* days (the
  canonical OR benchmark; strong under a fixed ordering cost).
- `cost_min` — simulation-based optimizer (also the hybrid agent's tool).

### Economics (`opensupply/scenario.py`)
- Cost model stated explicitly: `total = holding + stockout + ordering`; COGS is
  excluded (demand-determined) and `budget` is a hard operational constraint.
- Holding scales with `unit_cost`. The stylized daily holding cost + the three
  stockout levels imply a **newsvendor service-level spread of ≈ 0.74 / 0.92 /
  0.97** over the review interval — documented via `Scenario.implied_service_level()`
  so the parameter choice is defensible, not magic.
- Config consolidated onto the `Scenario` (single source of truth): `base_demand`,
  `dispersion`, `mean_lead`, `review_period`, `service_level`.

### Metrics (`opensupply/evaluation/`)
- **Fill rate** (achieved service level) added to every result.
- **Clairvoyant oracle** (`oracle.py`): a look-ahead policy that knows the realized
  demand path and orders just-in-time; `regret_vs_oracle` measures cost above this
  reference. Labeled a *strong heuristic reference*, not a proven optimum (the true
  optimum under stochastic lead time is a DP; stated as such). This replaces the
  circular best-of-evaluated regret.

## What is still explicit scope (state it in the paper, don't hide it)

- **Single SKU.** Deliberate for the first paper (no cross-SKU budget coupling,
  substitution, or assortment). Multi-SKU shared-budget replenishment is named as
  future work, not silently ignored.
- **Lost-sales (not backorder) model.** Unmet demand is lost, not queued — the
  small-retailer-relevant case. Backorder is a one-line variant left for later.
- **Synthetic demand.** Characteristic shapes, not calibrated to a real dataset;
  the Phase-7 case study grounds applicability on de-identified real data rather
  than claiming the synthetic generator is realistic.
- **Oracle is heuristic**, not a provable lower bound. Reported as such.

## Bottom line

v0.1 was a runnable scaffold; v0.2 is defensible for a workshop / applied-AI
submission: count-based demand with a dispersion axis, textbook baselines matched
to the cost structure, interpretable economics with a stated service-level spread,
and a fill-rate + clairvoyant-regret evaluation. The remaining limitations are
scoping choices to declare, not holes to hide.
