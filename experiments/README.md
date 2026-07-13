# `experiments/` — runnable scripts

The end-to-end workflow, from building the benchmark to the analysis tables. The
baseline path is free; only the LLM scripts spend API tokens (and they default to
safe dry-runs).

| Script | What it does | Cost |
|---|---|---|
| `run_first_experiment.py` | One 90-day scenario, all baselines, prints costs + saves an inventory-curve figure | free |
| `generate_scenarios.py` | Build the reproducible benchmark (7 demand × 3 lead × seeds, a distinct SKU each) | free |
| `run_benchmark.py` | Run policies over the benchmark → results CSV (with oracle regret) | free (baselines) / paid (`--methods llm`) |
| `estimate_llm_cost.py` | Price an LLM run before spending anything | free |
| `run_llm_benchmark.py` | Batched LLM sweep (Batch API, 50% off); dry-run by default | free preview / paid `--live` |
| `tune_baselines.py` | Fit the tuned `(s,S)` safety factor on a train split | free |
| `analyze_results.py` | VBC + normalized/weighted tables + per-demand-type breakdown | free |

---

## Typical flow

```bash
# 1. Build the benchmark (6 SKU archetypes × 7 demand × 3 lead × seeds)
python experiments/generate_scenarios.py --seeds 25 --out data/scenarios

# 2. Free baselines over the whole benchmark
python experiments/run_benchmark.py --scenarios data/scenarios --out data/results.csv

# 3. Fit the tuned classical baseline (train/test split)
python experiments/tune_baselines.py --scenarios data/scenarios

# 4. Analysis: VBC bar + normalized cost, uniform vs prevalence weighting
python experiments/analyze_results.py --results data/results.csv
```

## Adding the LLM methods (spends tokens)

```bash
# price it first (free) — cells are 126 (6 SKU × 7 demand × 3 lead),
# so K seeds/cell = 126·K scenarios
python experiments/estimate_llm_cost.py --scenarios 252   # e.g. K=2

# offline smoke test of the whole pipeline ($0)
python experiments/run_llm_benchmark.py --scenarios data/scenarios --seeds 2 --stub

# the real batched sweep (needs ANTHROPIC_API_KEY; ~50% off via Batch API)
python experiments/run_llm_benchmark.py --scenarios data/scenarios --seeds 2 --live
```

Safety: LLM scripts print a cost estimate and refuse to call the API unless you
pass `--live`, and `run_benchmark.py --methods llm` enforces a scenario cap.
