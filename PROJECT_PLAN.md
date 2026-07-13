# OpenSupplyBench — Roadmap & Status

Derived from `OpenSupplyBench_Paper_Plan.pdf`. Target: arXiv + GitHub +
workshop submission in ~6 months (9 if busy). **Guiding principle:** don't
start with the LLM agent, don't read 50 papers, don't design a full app —
get the single-SKU simulator + 3 baselines running first, *then* decide.

**v0.2–v0.4 research-hardening (done):** count-based (Negative-Binomial) demand
with a dispersion axis; 4 textbook baselines matched to the fixed-order cost (no
daily-dribble bug) + a **VBC (per-scenario best classical)** bar and a train/test
tuned baseline; scale-normalized, explicitly-weighted evaluation vs. a clairvoyant
oracle; economic fixes (margin-tethered stockout, days-of-supply budget); and a
**factorial of interpretable SKU archetypes × conditions** (`opensupply/skus.py`)
so SKU type is a controlled analysis factor. Rationale + the full v0.1→v0.4
diff are in [`paper/EVALUATION.md`](paper/EVALUATION.md). Single-SKU-per-scenario
(no shared budget), lost-sales, and synthetic demand remain **explicit scope**.

## Phase status

| Phase | Goal | Done-when | Status |
|-------|------|-----------|--------|
| 0 | Fix minimal research question | 1-page proposal, scope locked | ✅ `paper/proposal.md` |
| 1 | Minimal background: replenishment + forecasting | can implement forecast baseline + policy | ✅ `forecast.py`, policies |
| 2 | Minimal single-SKU simulator | 90-day sim → cost/stockout/order history | ✅ `simulator.py` + demand/leadtime |
| 3 | Baseline policies | (s,S), forecast+safety, (R,S), cost-min | ✅ `policies/` (v0.2: 4 strong policies) |
| 4 | Add LLM | LLM-only baseline + hybrid agent | ✅ `opensupply/agents/` (offline-tested; live run needs API key) |
| 5 | Build benchmark | ≥1000 reproducible scenarios | ✅ `generate_scenarios.py` (factorial: 6 SKU × 7 demand × 3 lead × seeds) |
| 6 | Experiments | main table + scenario breakdown + ablation | ⏳ harness ready (`run_benchmark.py` sync, `run_llm_benchmark.py` batched); analysis TODO |
| 7 | Real small-business case study | run on de-identified data | ⏳ TODO |
| 8 | Open-source repo | others can clone & reproduce | 🔵 in progress (structure + README done) |
| 9 | Write paper | full draft + figures + limitations | ⏳ TODO `paper/` |
| 10 | arXiv + submission | preprint + GitHub public + venue | ⏳ TODO |

## Phase 4 status (LLM work — done, pending a live run)

All three implement the `Policy` interface and drop into `Simulator` unchanged.
Verified offline with a stub client (`tests/test_agents.py`, no API call):

1. **`LLMOnlyPolicy` (4.1)** — LLM reads state, returns `{order_quantity, reason}`
   via structured outputs (`messages.parse`). `run_llm_demo.py` includes a
   repeated-query stability probe to quantify instability.
2. **`LLMScenarioClassifier` (4.2)** — classifies scenario type + demand
   uncertainty + supplier risk + recommended policy + a `safety_multiplier`.
3. **`HybridAgent` (4.3)** — classifier → `CostMinimizationPolicy` optimizer
   (numeric backbone) → LLM reviewer on significant orders. LLM frames + reviews;
   tools do the math. Cost-aware: classifier on a cadence, reviewer only on large
   orders (far fewer calls than daily).

Model: `claude-sonnet-5` default (cheap, high-volume) / `claude-opus-4-8` for hard
cases. Token cost + latency tracked in `agents/llm_client.py` (`LLMUsage`).

**Not yet run against the live API** — `experiments/run_llm_demo.py` needs an
API key and costs tokens; run it manually when ready. `HybridAgent` exposes
`use_classifier` / `use_reviewer` flags for the Phase-6.3 ablation.

## Immediate next steps

- Run `run_llm_demo.py` (with API key) to get the first real LLM-vs-baseline numbers.
- Wire the LLM policies into `run_benchmark.py` on a *subsampled* scenario set
  (LLM calls are expensive) for the Phase-6.1 main table.
- Phase 6.2 breakdown + 6.3 ablation; Phase 7 real case study.

## Experiments to run (Phase 6)

- **6.1 Main:** all 5 methods × benchmark → cost/stockout/overstock/regret/
  stability/token-cost table. (Baselines runnable now via `run_benchmark.py`.)
- **6.2 Breakdown:** stable / viral spike / supplier delay / low budget / high
  stockout penalty — find where hybrid actually helps vs. where baselines suffice.
- **6.3 Ablation:** full hybrid vs. −classifier vs. −reviewer vs. −cost-aware
  constraints vs. LLM-only.

## Timeline

M1 background + formulation + single-SKU sim · M2 demand/lead-time + baselines ·
M3 LLM-only + hybrid v1 · M4 benchmark + main experiments · M5 ablation + case
study + repo cleanup · M6 paper draft + figures + arXiv + workshop submission.
