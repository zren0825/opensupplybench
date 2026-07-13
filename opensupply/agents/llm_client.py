"""Anthropic client wrapper with usage/latency tracking + cost controls (Phase 4).

Keeps LLM plumbing in one place so every agent shares token-cost and latency
accounting (Phase 5.2 metrics). Three cost levers live here:

  * Batch API (`parse_batch`) — 50% cheaper, for the non-latency-sensitive
    benchmark sweep. Falls back to sequential calls when the injected client
    has no `messages.batches` (e.g. test stubs), so offline tests still work.
  * Prompt caching (`cache_system=True`) — adds `cache_control` to the system
    prompt. NOTE: our prompts are ~320 tokens, well below the ~2048-token cache
    minimum, so this is INERT today; it only starts paying off if the system
    prompt grows large (e.g. few-shot examples). Kept correct and off by default.
  * Cost estimation — `LLMUsage.cost_usd()` prices sync and batch tokens
    separately (batch at half rate).

The `anthropic` SDK is imported lazily, so importing this module needs no SDK
and no API key.
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass

# Sticker input/output price per 1M tokens (see claude-api model table).
_PRICING = {
    "claude-sonnet-5": (3.0, 15.0),
    "claude-opus-4-8": (5.0, 25.0),
    "claude-haiku-4-5": (1.0, 5.0),
}
# Introductory Sonnet-5 pricing, active through 2026-08-31.
_PRICING_INTRO = {"claude-sonnet-5": (2.0, 10.0)}


def pricing_for(model: str, intro: bool = False):
    if intro and model in _PRICING_INTRO:
        return _PRICING_INTRO[model]
    return _PRICING.get(model, (3.0, 15.0))


@dataclass
class LLMUsage:
    calls: int = 0
    input_tokens: int = 0
    output_tokens: int = 0
    batch_calls: int = 0
    batch_input_tokens: int = 0
    batch_output_tokens: int = 0
    latency_s: float = 0.0

    def record(self, usage, dt: float = 0.0, batch: bool = False) -> None:
        get = (lambda k: getattr(usage, k, 0)) if not isinstance(usage, dict) else usage.get
        it = get("input_tokens") or 0
        ot = get("output_tokens") or 0
        self.latency_s += dt
        if batch:
            self.batch_calls += 1
            self.batch_input_tokens += it
            self.batch_output_tokens += ot
        else:
            self.calls += 1
            self.input_tokens += it
            self.output_tokens += ot

    def cost_usd(self, model: str = "claude-sonnet-5", intro: bool = False) -> float:
        in_p, out_p = pricing_for(model, intro)
        sync = (self.input_tokens * in_p + self.output_tokens * out_p) / 1e6
        # Batch API is billed at 50% of standard rates.
        batch = 0.5 * (self.batch_input_tokens * in_p + self.batch_output_tokens * out_p) / 1e6
        return sync + batch

    def summary(self, model: str = "claude-sonnet-5", intro: bool = False) -> dict:
        return {
            "calls": self.calls + self.batch_calls,
            "input_tokens": self.input_tokens + self.batch_input_tokens,
            "output_tokens": self.output_tokens + self.batch_output_tokens,
            "batch_calls": self.batch_calls,
            "latency_s": round(self.latency_s, 2),
            "est_cost_usd": round(self.cost_usd(model, intro), 4),
        }


def pydantic_to_json_schema(model_cls) -> dict:
    """Convert a Pydantic model to a structured-outputs JSON schema.

    `messages.parse` does this internally, but the Batch API takes raw
    `messages.create` params, so we build the schema ourselves. Our schemas are
    flat (str/int/float/enum), which keeps this simple and within the
    structured-outputs feature set (object + additionalProperties:false + all
    fields required).
    """
    schema = model_cls.model_json_schema()
    schema.pop("title", None)
    props = schema.get("properties", {})
    for p in props.values():
        p.pop("title", None)
    schema["required"] = list(props.keys())
    schema["additionalProperties"] = False
    return schema


class LLMClient:
    def __init__(
        self,
        model: str = "claude-sonnet-5",
        max_tokens: int = 512,
        client=None,
        thinking_disabled: bool = True,
        cache_system: bool = False,
        intro_pricing: bool = False,
    ):
        # Default model is the cheaper Sonnet tier because these calls run
        # per-day across many scenarios; pass model="claude-opus-4-8" for the
        # hardest/messiest cases where the extra capability earns its cost.
        self.model = model
        self.max_tokens = max_tokens
        self.thinking_disabled = thinking_disabled
        self.cache_system = cache_system
        self.intro_pricing = intro_pricing
        self._client = client
        self.usage = LLMUsage()

    def _ensure_client(self):
        if self._client is None:
            import anthropic  # lazy: only needed when actually calling the API

            self._client = anthropic.Anthropic()
        return self._client

    def _system_field(self, system: str):
        if self.cache_system:
            return [{"type": "text", "text": system, "cache_control": {"type": "ephemeral"}}]
        return system

    def parse(self, system: str, user: str, schema):
        """One synchronous structured call; returns the parsed pydantic object."""
        client = self._ensure_client()
        kwargs = dict(
            model=self.model,
            max_tokens=self.max_tokens,
            system=self._system_field(system),
            messages=[{"role": "user", "content": user}],
            output_format=schema,
        )
        if self.thinking_disabled:
            kwargs["thinking"] = {"type": "disabled"}
        start = time.time()
        resp = client.messages.parse(**kwargs)
        self.usage.record(getattr(resp, "usage", {}), time.time() - start, batch=False)
        return resp.parsed_output

    def parse_batch(self, requests, schema, poll_seconds: int = 20):
        """Structured calls via the Batch API (50% cost), for the offline sweep.

        `requests` is a list of `(system, user)` tuples answered against one
        shared `schema`. Returns parsed objects in the same order (None for any
        request that errored). Falls back to sequential `parse()` when the
        client has no `messages.batches` (test stubs, or SDKs without batches),
        so this method is safe to call in offline tests.
        """
        client = self._ensure_client()
        if not hasattr(getattr(client, "messages", None), "batches"):
            return [self.parse(sys, usr, schema) for sys, usr in requests]

        from anthropic.types.message_create_params import MessageCreateParamsNonStreaming
        from anthropic.types.messages.batch_create_params import Request

        json_schema = pydantic_to_json_schema(schema)
        fmt = {"format": {"type": "json_schema", "schema": json_schema}}
        params_common = dict(
            model=self.model, max_tokens=self.max_tokens, output_config=fmt
        )
        if self.thinking_disabled:
            params_common["thinking"] = {"type": "disabled"}

        batch_requests = [
            Request(
                custom_id=f"req-{i}",
                params=MessageCreateParamsNonStreaming(
                    system=self._system_field(system),
                    messages=[{"role": "user", "content": user}],
                    **params_common,
                ),
            )
            for i, (system, user) in enumerate(requests)
        ]

        start = time.time()
        batch = client.messages.batches.create(requests=batch_requests)
        while True:
            b = client.messages.batches.retrieve(batch.id)
            if b.processing_status == "ended":
                break
            time.sleep(poll_seconds)
        dt = time.time() - start

        out = [None] * len(requests)
        for result in client.messages.batches.results(batch.id):
            idx = int(result.custom_id.split("-")[1])
            if result.result.type != "succeeded":
                continue
            msg = result.result.message
            self.usage.record(getattr(msg, "usage", {}), 0.0, batch=True)
            text = next((b.text for b in msg.content if b.type == "text"), "")
            try:
                out[idx] = schema(**json.loads(text))
            except Exception:
                out[idx] = None
        self.usage.latency_s += dt
        return out
