# `opensupply/evaluation/` — metrics, references, and honest aggregation

How a policy is scored, and how the aggregate numbers are made trustworthy. The
subtle choices here are what separate "we beat a baseline" from a claim that
survives review.

| File | Provides |
|---|---|
| `metrics.py` | Per-run metrics (cost breakdown, **fill rate**, relative regret) |
| `oracle.py` | **Clairvoyant oracle** — a strong per-scenario reference for regret |
| `vbc.py` | **Virtual Best Classical** — the per-scenario best classical policy |
| `weighting.py` | Scale-normalized, explicitly-weighted aggregation |
| `tuning.py` | Train/test fitting of the tuned baseline |

---

## Per-run metrics (`metrics.py`)

For one policy on one scenario: `total_cost` (= holding + stockout + ordering) and
its breakdown, **`fill_rate`** (achieved service level = units filled ÷ demanded —
the standard service KPI), `stockout_rate`, `overstock`, `num_orders`. For LLM
methods, token cost and latency come from `agents/LLMUsage`.

## Two reference points for "how good is this?"

Raw cost is meaningless without a reference. There are two:

**1. Clairvoyant oracle (`oracle.py`) — the lower bar.** A policy that *knows the
realized demand path* and orders just-in-time to cover it, sweeping a few
cadences to pick the cheapest. `regret = policy_cost − oracle_cost` measures how
far above the best-possible you are. Honest caveat, stated in the paper: it's
clairvoyant about demand but still faces stochastic lead times, so it's a **strong
heuristic reference, not a provable optimum**.

**2. Virtual Best Classical (`vbc.py`) — the real bar.** No single classical
policy wins everywhere, so the strong opponent is the **per-scenario minimum**
across all classical policies: "the best a traditional method could do if you
always picked the right one." The paper's central test is whether the hybrid
agent beats VBC — i.e. whether the LLM adds value *beyond* selecting/tuning the
right classical policy. Keeping a diverse policy pool makes VBC a *tougher* bar,
not an easier one.

## Aggregation: normalize, then weight (`weighting.py`)

Two traps in averaging across scenarios, both fixed here:

- **Scale.** A premium high-volume SKU can cost 50× a cheap low-volume one, so a
  plain mean of raw dollars just measures the expensive scenarios. Fix: aggregate
  the **cost ratio to oracle** (`cost ÷ oracle_cost`), so every scenario counts
  comparably. (This alone can flip the ranking — raw cost favored `cost_min`;
  normalized favors `forecast_safety`.)
- **Prevalence.** A balanced grid gives `viral_spike` and `stable` equal weight,
  but viral spikes are rare in reality — equal weighting over-represents the hard
  cases and flatters an adaptive method. Fix: report the headline under **both**
  a `uniform` scheme and a **documented prevalence prior** (`PREVALENCE_PRIOR`:
  common patterns up-weighted, spikes/launches down-weighted). If the hybrid wins
  under both, the claim is strong; if only under uniform, that's honestly reported
  as "helps disproportionately on the rarer, harder scenarios."

`experiments/analyze_results.py` produces these tables plus a per-demand-type
breakdown.

## Fair baseline tuning (`tuning.py`)

`train_test_split` divides scenarios by seed parity; `tune_safety_factor`
grid-searches the tuned baseline's buffer on the **train** split (minimizing mean
cost-ratio-to-oracle) and it is evaluated on **held-out** seeds. This makes the
classical baseline genuinely fitted rather than a raw formula — without hindsight
overfitting.
