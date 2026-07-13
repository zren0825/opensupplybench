"""Per-run LLM cost estimator (Phase 4/6 planning).

Predicts API cost for an LLM benchmark run *before* spending anything, so every
paid sweep has a known price tag. Models the call pattern of each method:

  * llm_only: one call per simulated day  -> horizon calls / scenario.
  * hybrid:   classifier on a cadence + a few reviewer calls (see hybrid.py),
              far fewer than daily.

Token sizes default to measured values for this project's prompts; override
them if the prompt changes. Batch API applies a 0.5 multiplier; intro pricing
uses the promotional Sonnet-5 rate.
"""

from __future__ import annotations

from dataclasses import dataclass

from opensupply.agents.llm_client import pricing_for

# Measured defaults for this repo's prompts (system + state summary ~ 320 tok
# input; short JSON with a <=12-word reason ~ 40 tok output).
DEFAULT_INPUT_TOKENS = 320
DEFAULT_OUTPUT_TOKENS = 40


def calls_per_scenario(method: str, horizon: int, classify_every: int = 14,
                       avg_reviews: int = 2) -> int:
    if method == "llm_only":
        return horizon
    if method == "hybrid":
        # one classifier call per cadence window (once history exists) + a
        # couple of reviewer calls on significant orders.
        return max(1, horizon // classify_every) + avg_reviews
    return 0


@dataclass
class CostEstimate:
    method: str
    n_scenarios: int
    calls: int
    input_tokens: int
    output_tokens: int
    cost_usd: float


def estimate(
    methods,
    n_scenarios: int,
    model: str = "claude-sonnet-5",
    horizon: int = 90,
    input_tokens: int = DEFAULT_INPUT_TOKENS,
    output_tokens: int = DEFAULT_OUTPUT_TOKENS,
    batch: bool = False,
    intro: bool = False,
):
    """Return (list[CostEstimate], total_cost_usd)."""
    in_p, out_p = pricing_for(model, intro)
    factor = 0.5 if batch else 1.0
    rows, total = [], 0.0
    for m in methods:
        cps = calls_per_scenario(m, horizon)
        if cps == 0:
            continue
        calls = n_scenarios * cps
        it, ot = calls * input_tokens, calls * output_tokens
        cost = factor * (it * in_p + ot * out_p) / 1e6
        rows.append(CostEstimate(m, n_scenarios, calls, it, ot, cost))
        total += cost
    return rows, total


def format_table(methods, n_scenarios: int, model: str = "claude-sonnet-5",
                 horizon: int = 90) -> str:
    """A compact comparison across sync/batch and sticker/intro pricing."""
    lines = [f"Cost estimate — {n_scenarios} scenarios x {horizon}d, model={model}",
             f"{'method':<12}{'calls':>10}", "-" * 22]
    rows, _ = estimate(methods, n_scenarios, model, horizon)
    for r in rows:
        lines.append(f"{r.method:<12}{r.calls:>10,}")
    lines.append("")
    lines.append(f"{'pricing':<22}{'total $':>10}")
    lines.append("-" * 32)
    combos = [
        ("sync, sticker", dict(batch=False, intro=False)),
        ("sync, intro", dict(batch=False, intro=True)),
        ("BATCH, sticker", dict(batch=True, intro=False)),
        ("BATCH, intro", dict(batch=True, intro=True)),
    ]
    for label, kw in combos:
        _, total = estimate(methods, n_scenarios, model, horizon, **kw)
        lines.append(f"{label:<22}{total:>10.2f}")
    return "\n".join(lines)
