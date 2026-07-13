# OpenSupplyBench

**A scenario-based benchmark for small-business inventory replenishment agents.**

Can cost-aware *hybrid* LLM agents make better replenishment decisions than
traditional policies when demand is uncertain, supplier lead times are
unstable, and the budget is tight? OpenSupplyBench provides a reproducible
simulator, a scenario benchmark, and a set of baseline and LLM-based policies
to answer that question.

> This is a research artifact, not a product. See [`paper/proposal.md`](paper/proposal.md)
> for the 1-page scope and [`PROJECT_PLAN.md`](PROJECT_PLAN.md) for the full roadmap.

## Quickstart

```bash
pip install -r requirements.txt

# 1. Run the first experiment: one 90-day scenario, the baseline policies,
#    prints cost/stockout/orders and saves an inventory curve figure.
python experiments/run_first_experiment.py

# 2. Generate a (small) reproducible benchmark set and run the free baselines.
python experiments/generate_scenarios.py --seeds 3 --out data/scenarios
python experiments/run_benchmark.py --scenarios data/scenarios --out data/results.csv
```

The full benchmark is a **factorial**: 6 SKU archetypes × 7 demand × 3 lead-time
× seeds. With `--seeds 25` that's **3150 reproducible scenarios**, where every
SKU type is run through every demand/lead condition.

> See [`paper/EVALUATION.md`](paper/EVALUATION.md) for the research-readiness
> review (what was hardened v0.1 → v0.4 and why).

## What's in the benchmark

- **Demand (Negative-Binomial counts):** stable, seasonal, promotion spike,
  viral spike, declining, new-product ramp, intermittent — aligned with the LLM
  classifier's taxonomy. A `dispersion` knob spans Poisson→overdispersed
  (`CV² = 1/µ + 1/r`). `opensupply/demand.py`
- **Lead-time types:** fixed, random, delayed (fat-tailed) — `opensupply/leadtime.py`
- **SKU archetypes** (`opensupply/skus.py`): a fixed set of interpretable product
  types — `staple`, `perishable`, `premium`, `bulky`, `critical_import`,
  `impulse` — spanning the holding-vs-stockout tradeoff (implied service levels
  ≈ 0.59–0.98). Each archetype is run through every condition, so **SKU type is a
  controlled factor** in the analysis, not entangled with the scenario.
- **Economics:** `total = holding + stockout + ordering` (lost-sales). Holding
  scales with unit cost; **stockout is tethered to unit margin**
  (`margin × multiple`); **budget is expressed in days of supply**
  (`base_demand × unit_cost × days`), so it is not confounded with SKU scale.
  Each SKU's implied newsvendor service level is its policies' cost-aware target
  (`Scenario.implied_service_level()`). `opensupply/scenario.py`

## How it works (and where to read more)

The project models **one decision** — *how many units to reorder today?* — and
scores strategies for making it. A day at a time, the simulator receives arrived
orders, observes demand, sells what it can (unmet demand is a lost sale), charges
holding + stockout + ordering cost, and asks the current **policy** to order
(arriving after a supplier lead time). Run 90 days and you get a total cost. A
**benchmark** — every SKU archetype run through every demand/lead condition —
lets strategies be compared fairly, from simple rules up to the hybrid LLM agent.

Each subsystem has a short high-level guide:

| Read this | To understand |
|---|---|
| [`opensupply/README.md`](opensupply/README.md) | The core model: how a scenario is built, **how SKUs differ**, how demand & lead time are generated, the day loop |
| [`opensupply/policies/README.md`](opensupply/policies/README.md) | **How each classical baseline works** ((s,S), (R,S), cost-min, tuned) |
| [`opensupply/agents/README.md`](opensupply/agents/README.md) | The LLM-only baseline and the **hybrid agent** design |
| [`opensupply/evaluation/README.md`](opensupply/evaluation/README.md) | Metrics, the clairvoyant oracle, the Virtual Best Classical bar, and honest weighting |
| [`experiments/README.md`](experiments/README.md) | What each runnable script does and the end-to-end workflow |
| [`paper/EVALUATION.md`](paper/EVALUATION.md) | The research-readiness audit (v0.1 → v0.3 and why) |

## Methods

| # | Policy | Status |
|---|--------|--------|
| 1 | Rule-based continuous-review *(s, S)* | ✅ `policies/rule_based.py` |
| 2 | Forecast + safety stock *(s, S)* | ✅ `policies/forecast_safety.py` |
| 3 | Periodic-review *(R, S)* base-stock | ✅ `policies/periodic_review.py` |
| 4 | Cost-minimization optimizer | ✅ `policies/cost_min.py` |
| 5 | LLM-only (direct decision) | ✅ `agents/llm_only.py` |
| 6 | **Hybrid LLM agent** (proposed) | ✅ `agents/hybrid.py` |

The four baselines are textbook policies matched to the fixed-order-cost
economics (they order on a trigger/cadence, not daily). The LLM methods use
structured outputs via the Anthropic SDK and are verified offline with a stub
client. Running them against the live API needs credentials and spends tokens:

```bash
pip install anthropic pydantic
python experiments/estimate_llm_cost.py --scenarios 486   # price a run first (free)
python experiments/run_llm_demo.py --stub                 # offline smoke test ($0)
python experiments/run_llm_demo.py                        # live, ~$0.12
python experiments/run_llm_benchmark.py --seeds 3 --live  # batched sweep (50% off)
```

## Evaluation

Metrics: total cost (holding + stockout + ordering) · **fill rate** (achieved
service level) · stockout rate / lost sales · overstock · decision stability ·
(LLM) token cost & latency.

Aggregation is **scale-normalized** (cost ÷ clairvoyant oracle) so scenarios of
different volume count comparably, and reported under both **uniform** and a
**documented prevalence-weighted** scheme. The strong bar is the **Virtual Best
Classical** (per-scenario best over the classical policies) — the paper's test is
whether the hybrid beats VBC, not any single policy. `opensupply/evaluation/`

```bash
# tune the classical safety factor on a train split (offline)
python experiments/tune_baselines.py --scenarios data/scenarios
# VBC + normalized + weighted tables + per-demand-type breakdown
python experiments/analyze_results.py --results data/results.csv
```

## Repo structure

```
opensupply/
  scenario.py       # Scenario spec (single source of config) + cost model
  skus.py           # heterogeneous SKU economics sampler
  demand.py         # Negative-Binomial demand generator (7 types, dispersion)
  leadtime.py       # supplier lead-time models
  forecast.py       # forecast_tool (moving avg / exp smoothing / seasonal)
  simulator.py      # single-SKU day loop; run() + steppable generator
  policies/         # 4 baselines (s,S / R,S / cost-min) + tuned (s,S) + util
  agents/           # LLM-only, classifier, hybrid + llm_client, cost, prompts
  evaluation/       # metrics, oracle regret, VBC, weighting, train/test tuning
experiments/
  run_first_experiment.py   # Day 6-7 milestone
  generate_scenarios.py     # Phase 5.1 — build the benchmark grid
  run_benchmark.py          # Phase 6.1 — synchronous run, results CSV
  run_llm_benchmark.py      # batched (Batch API, 50% off) LLM sweep
  estimate_llm_cost.py      # price any LLM run before spending
  tune_baselines.py         # fit tuned (s,S) on a train split
  analyze_results.py        # VBC + normalized/weighted tables + breakdown
paper/              # proposal, EVALUATION.md, figures
tests/              # smoke + agents + cost/batch + v0.2 tests
```

## Reproduce

Every scenario is fully determined by its seed and fields, so
`generate_scenarios.py` + `run_benchmark.py` reproduce the results exactly.
Run the offline tests with:

```bash
python tests/test_smoke.py && python tests/test_v02.py && \
python tests/test_skus.py && python tests/test_agents.py && \
python tests/test_cost_and_batch.py && python tests/test_analysis.py
```

## Citation

```bibtex
@misc{opensupplybench,
  title  = {OpenSupplyBench: A Scenario-Based Benchmark for Small-Business
            Inventory Replenishment Agents},
  year   = {2026},
  note   = {Preprint / code: https://github.com/zren0825/opensupplybench}
}
```
