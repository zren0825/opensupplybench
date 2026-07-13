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

The full benchmark is `--seeds 40` → 7 demand × 3 lead-time × 3 budget ×
3 stockout-cost × 40 seeds = **7560 reproducible scenarios**, each also assigned
a SKU scale and demand dispersion (drawn reproducibly per scenario).

> See [`paper/EVALUATION.md`](paper/EVALUATION.md) for the research-readiness
> review (what was hardened from v0.1 → v0.2 and why).

## What's in the benchmark

- **Demand (Negative-Binomial counts):** stable, seasonal, promotion spike,
  viral spike, declining, new-product ramp, intermittent — aligned with the LLM
  classifier's taxonomy. A `dispersion` knob spans Poisson→overdispersed
  (`CV² = 1/µ + 1/r`). `opensupply/demand.py`
- **Lead-time types:** fixed, random, delayed (fat-tailed) — `opensupply/leadtime.py`
- **SKU heterogeneity:** demand scale (low/med/high volume) × dispersion, drawn
  per scenario.
- **Economics/constraints:** unit / holding / stockout / order costs, per-order
  budget, MOQ, case pack. Cost model `total = holding + stockout + ordering`
  (lost-sales); the parameters imply a documented service-level spread
  (`Scenario.implied_service_level()`). `opensupply/scenario.py`

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

## Evaluation metrics

total cost (holding + stockout + ordering) · **fill rate** (achieved service
level) · stockout rate / lost sales · overstock · **regret vs. a clairvoyant
oracle** · decision stability · (LLM) token cost & latency —
`opensupply/evaluation/`

## Repo structure

```
opensupply/
  scenario.py       # Scenario spec (single source of config) + cost model
  demand.py         # Negative-Binomial demand generator (7 types, dispersion)
  leadtime.py       # supplier lead-time models
  forecast.py       # forecast_tool (moving avg / exp smoothing / seasonal)
  simulator.py      # single-SKU day loop; run() + steppable generator
  policies/         # 4 baseline policies (s,S / R,S / cost-min) + util
  agents/           # LLM-only, classifier, hybrid + llm_client, cost, prompts
  evaluation/       # metrics (fill rate) + clairvoyant oracle regret
experiments/
  run_first_experiment.py   # Day 6-7 milestone
  generate_scenarios.py     # Phase 5.1 — build the benchmark grid
  run_benchmark.py          # Phase 6.1 — synchronous run, results CSV
  run_llm_benchmark.py      # batched (Batch API, 50% off) LLM sweep
  estimate_llm_cost.py      # price any LLM run before spending
paper/              # proposal, EVALUATION.md, figures
tests/              # smoke + agents + cost/batch + v0.2 tests
```

## Reproduce

Every scenario is fully determined by its seed and fields, so
`generate_scenarios.py` + `run_benchmark.py` reproduce the results exactly.
Run the offline tests with:

```bash
python tests/test_smoke.py && python tests/test_v02.py && \
python tests/test_agents.py && python tests/test_cost_and_batch.py
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
