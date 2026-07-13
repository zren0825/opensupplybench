# OpenSupplyBench — 1-Page Project Proposal (Phase 0)

**Title:** OpenSupplyBench: A Scenario-Based Benchmark for Small-Business
Inventory Replenishment Agents

**Research question:** Can cost-aware *hybrid* LLM agents improve
replenishment decisions under demand and lead-time uncertainty, for small
businesses with messy data and tight budgets — where a pure optimizer is
brittle to context and a pure LLM is numerically unstable?

**Scope (what this paper is and is NOT):**
- IS: a *replenishment-decision benchmark* + a *hybrid agent* method paper.
- IS NOT: a forecasting-accuracy paper, and NOT a full supply-chain app.
- First paper stays on single-SKU inventory replenishment under uncertainty.

**Inputs (what a policy sees each day):** sales history, current inventory,
on-order pipeline, lead-time estimate, unit/holding/stockout costs, per-order
budget, MOQ/case-pack, and free-text business & supplier notes.

**Outputs:** order quantity, reorder timing, and (for LLM methods) a
natural-language explanation of the decision.

**Methods compared:**
1. Rule-based continuous-review *(s, S)* — baseline 1
2. Forecast + safety stock *(s, S)* — baseline 2
3. Periodic-review *(R, S)* base-stock — baseline 3
4. Cost-minimization optimizer — baseline 4
5. LLM-only (direct decision) — baseline 5
6. **Hybrid LLM agent** (classifier + forecast + optimizer + reviewer) — proposed

**Metrics:** total inventory cost (holding + stockout + ordering), fill rate
(achieved service level), stockout rate / lost sales, overstock, regret vs. a
clairvoyant oracle, decision stability, and — for LLM methods — token cost and
latency.

**Benchmark:** a factorial of **6 interpretable SKU archetypes × 7 demand
patterns × 3 lead-time regimes × seeds**. Archetypes (staple, perishable,
premium, bulky, critical_import, impulse) span the holding-vs-stockout tradeoff
(implied service levels ≈ 0.59–0.98), with holding scaled to unit cost, stockout
tethered to margin, and days-of-supply budgets. Every SKU is run through every
condition, so SKU type is a controlled factor. Single-SKU-per-scenario (no shared
budget across SKUs — future work). See `EVALUATION.md` for the design audit.

**Research questions for the experiments:**
- RQ1: Does the hybrid agent reduce total cost vs. baselines?
- RQ2: In which scenarios (viral spike, supplier delay, low budget, high
  stockout penalty) is it most valuable — and where are baselines enough?
- RQ3: Why is LLM-only unstable?
- RQ4: Does each hybrid module contribute (ablation)?

**Target output:** open-source repo + reproducible benchmark (≥1000 scenarios)
+ arXiv preprint + workshop / applied-AI venue submission.
