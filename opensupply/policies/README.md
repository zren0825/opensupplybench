# `opensupply/policies/` — the classical (non-LLM) baselines

A **policy** answers one question each day: *how many units to order?* It
implements `decide(obs) -> quantity` (see [base.py](base.py)); the simulator
applies MOQ / case-pack / budget rounding to the result. These are the textbook
strategies the LLM agent must beat — deliberately **strong** ones, because
beating weak baselines proves nothing.

All four order on a **trigger or cadence** (not every day), so they respect the
fixed per-order fee. Notation: **s** = reorder point (when to order), **S** =
order-up-to level (target), **L** = lead time, **R** = review period, **z** =
safety factor from the target service level.

| Policy | File | One-liner |
|---|---|---|
| `rule_based` | `rule_based.py` | Simple continuous-review *(s, S)*, modest fixed buffer |
| `forecast_safety` | `forecast_safety.py` | *(s, S)* with a forecast + cost-aware safety stock |
| `periodic_review` | `periodic_review.py` | *(R, S)* base-stock, orders every R days |
| `cost_min` | `cost_min.py` | Simulation-based optimizer (also the hybrid agent's tool) |
| `tuned_ss` | `tuned.py` | Like `forecast_safety`, but its buffer is **fitted** on a train split |

---

## 1. Rule-based `(s, S)` — the honest simple baseline

```
mean, σ  = running stats of recent demand
s (reorder point) = mean·L + z·σ·√L          # z ≈ 1 (a plain buffer)
S (order-up-to)   = mean·(L+R) + z·σ·√L
if inventory_position ≤ s:  order  S − position
```

Reacts to a running average with a small safety cushion. Robust and cheap, but
blind — it can't anticipate a spike. This is the naive reference.

## 2. Forecast + safety stock `(s, S)`

Same shape, but smarter about *both* terms:
- the mean comes from an **exponential-smoothing forecast** (leans on recent
  data), and
- the safety stock is sized to the **SKU's cost-aware service level**
  (`z = Φ⁻¹(service_level)`, where `service_level` is the SKU's implied newsvendor
  ratio — see the [core guide](../README.md)).

The reorder trigger (order only when position ≤ s) is what keeps it from
re-ordering every day and drowning in the fixed order fee.

## 3. Periodic-review `(R, S)` base-stock

The canonical operations-research benchmark: **look only every R days**, and on a
review day top up to a base-stock level covering the protection interval:

```
if day % R == 0:
    S = mean·(L+R) + z·σ·√(L+R)              # note √(L+R): exposure is lead + review
    order  S − inventory_position
```

Because it orders on a fixed cadence it is naturally efficient under a fixed order
cost. (Subtle but correct: continuous-review policies hedge over `√L`, periodic
review over `√(L+R)`, because you can't reorder until the next review.)

## 4. Cost-minimization optimizer

Instead of a formula, this one **brute-forces the objective**:

```
build ~50 plausible future demand paths by bootstrapping recent demand
for each candidate order size q (0 … recent-peak-based cap):
    simulate the next L+R days over all paths, total holding+stockout+order cost
pick the q with the lowest expected cost
```

The strongest single non-LLM method — it optimizes the actual cost directly, with
no service-level guess. It's also the exact tool the hybrid agent calls. Its
blind spot: it only optimizes the numbers in front of it (recent demand), so it
can't foresee a spike the history doesn't hint at, and it can't read business
context — which is the opening for the LLM.

## 5. Tuned `(s, S)` — the *fairly* tuned classical baseline

Identical to `forecast_safety`, but the safety factor `k` is a free knob that is
**grid-searched on a training split of scenarios and evaluated on held-out
seeds** (`evaluation/tuning.py`, `experiments/tune_baselines.py`). This closes the
"you only beat an under-tuned textbook formula" critique — the classical side is
genuinely fitted, without hindsight overfitting.

---

## Why keep several, if the point is "strong not many"?

No single policy wins everywhere — across the benchmark the per-scenario winner is
split roughly rule-based 33% / forecast 32% / cost-min 27% / periodic 7%. So the
real bar is the **Virtual Best Classical**: the per-scenario *best* of this pool.
Keeping a diverse pool makes that bar *stronger*, and the paper's headline test is
whether the hybrid agent beats it — see [evaluation](../evaluation/README.md).
