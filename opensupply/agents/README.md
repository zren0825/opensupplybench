# `opensupply/agents/` — the LLM methods (Phase 4)

The proposed contribution. Two policies, built on the same `Policy` interface as
the classical baselines so they drop straight into the simulator and benchmark.

> **Cost note:** these are the only parts of the project that call the Anthropic
> API and spend tokens. The SDK is imported lazily, so importing this package
> needs no key; the offline tests use a stub client. Always price a run first
> (`experiments/estimate_llm_cost.py`) and prefer the batched sweep.

| Module | Role |
|---|---|
| `llm_client.py` | Thin Anthropic wrapper: structured outputs, token/latency/cost accounting, Batch API |
| `schemas.py` | Pydantic output schemas (forces valid JSON, no parsing) |
| `prompts.py` | Turns an `Observation` into the state summary the model reads |
| `llm_only.py` | **Baseline** — LLM emits the order quantity directly |
| `classifier.py` | Classifies the scenario (hybrid's first module) |
| `hybrid.py` | **The proposed method** — classifier → optimizer → reviewer |
| `cost.py` | Per-run cost estimator |

---

## The core idea

An LLM is unreliable as a *calculator* (ask it the same inventory question twice,
get two different numbers) but valuable as a *reader of messy context* (it
understands "this product is going viral" or "supplier flaky lately"). The design
uses it only for the latter.

## `LLMOnlyPolicy` — the baseline (what *not* to do)

Every day, the LLM sees the raw state and returns `{order_quantity, reason}` as
strict JSON. It **is** the calculator — one API call per simulated day. This is
the baseline the paper expects to be unstable and mediocre; `run_llm_demo.py`
includes a probe that asks the identical question several times to quantify the
instability.

## `HybridAgent` — the proposed method

The LLM frames and reviews; **proven inventory math does the arithmetic**:

```
1. Classify (occasionally, on a cadence — not daily):
     LLM reads the state + business/supplier notes → scenario type,
     demand uncertainty, supplier risk, a safety multiplier.
2. Compute (every day, free):
     the CostMinimization optimizer produces the cost-optimal order,
     scaled by the classifier's safety multiplier.
3. Review (only on unusually large orders):
     LLM sanity-checks the tool's number against the messy context;
     approves it or corrects it.
```

The result is **~6 API calls over 90 days instead of 90** — far cheaper than
LLM-only and far more stable, because the LLM never picks the number itself. The
`use_classifier` / `use_reviewer` flags let the Phase-6.3 ablation turn each module
off to measure its contribution.

## Why structured outputs

Both policies call `client.messages.parse(...)` with a Pydantic schema, so the
model is constrained to emit exactly the fields needed — no brittle text parsing,
and the model retries on a schema mismatch at the API layer.

## Cost controls (all in `llm_client.py`)

- **Batch API** (`parse_batch`) — 50% cheaper; used by the lockstep batched sweep.
- **Usage accounting** (`LLMUsage`) — tokens, latency, and $ cost (sync and batch
  priced separately), so token cost is a first-class benchmark metric.
- **Prompt caching** flag — present but *inert* at current prompt sizes (~320
  tokens, below the cache minimum); it only helps if prompts grow.

Model defaults to `claude-sonnet-5` (cheap, high-volume); pass
`claude-opus-4-8` for the hardest cases.
