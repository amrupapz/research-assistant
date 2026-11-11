"""Langfuse tracing (D6 M1). The ONLY module that imports langfuse.

One trace per agent/flow run, with nested observations:

    agent (root, trace_id seeded from run_id)        flow / flow-resume / flow-approve (root)
      ├─ respond            generation                ├─ <node>                 chain
      └─ <tool name>        tool (decision, status)   │    ├─ <tool name>       tool
           ├─ human-approval span                     │    │    ├─ hybrid_search retriever
           ├─ hybrid_search retriever                 │    │    │    └─ embed    embedding
           │    └─ embed    embedding                 │    │    └─ ask          generation
           └─ ask           generation                │    └─ write_email       generation

trace_id = create_trace_id(seed=run_id): a resumed or approved run lands in the SAME trace.

Off unless settings.tracing_enabled and both keys are set; then every helper is a no-op that
yields NOOP. Tests never send: conftest turns it off, and tests that check spans install a
client with an in-memory exporter via use().
"""

import re
import sys
from collections.abc import Iterator
from contextlib import contextmanager
from typing import Any
from uuid import UUID

from research_assistant.config import Settings, get_settings

EMAIL = re.compile(r"[\w.+-]+@[\w-]+(\.[\w-]+)+")


class _Noop:
    """Stands in for an observation when tracing is off: accepts every call, records nothing."""

    trace_id = None

    def update(self, **kwargs) -> "_Noop":
        return self

    def end(self, **kwargs) -> "_Noop":
        return self


NOOP = _Noop()
_OFF = object()
_client: Any = None  # None = not built yet, _OFF = disabled, else a Langfuse client


def mask(*, data: Any, **kwargs) -> Any:
    """Runs on every input/output/metadata before it leaves the process: email addresses -> <email>."""
    if isinstance(data, str):
        return EMAIL.sub("<email>", data)
    if isinstance(data, dict):
        return {k: mask(data=v) for k, v in data.items()}
    if isinstance(data, list | tuple):
        return [mask(data=v) for v in data]
    return data


def _build(settings: Settings):
    if not settings.tracing_enabled:
        return _OFF
    if settings.langfuse_public_key is None or settings.langfuse_secret_key is None:
        print("[trace] TRACING_ENABLED but LANGFUSE_PUBLIC_KEY/SECRET_KEY missing: tracing off", file=sys.stderr)
        return _OFF
    from langfuse import Langfuse

    return Langfuse(
        public_key=settings.langfuse_public_key.get_secret_value(),
        secret_key=settings.langfuse_secret_key.get_secret_value(),
        base_url=settings.langfuse_base_url,
        mask=mask,
    )


def client():
    """The Langfuse client, or None when tracing is off. Built once, on first use."""
    global _client
    if _client is None:
        _client = _build(get_settings())
    return None if _client is _OFF else _client


def use(langfuse_client) -> None:
    """Install a client (tests: one with an in-memory exporter). None = tracing off."""
    global _client
    _client = _OFF if langfuse_client is None else langfuse_client


def trace_id_for(run_id: UUID) -> str | None:
    """The trace id for a run (deterministic), or None when tracing is off (nothing was sent)."""
    lf = client()
    return lf.create_trace_id(seed=str(run_id)) if lf is not None else None


def current_trace_id() -> str | None:
    lf = client()
    return lf.get_current_trace_id() if lf is not None else None


@contextmanager
def run_trace(run_id: UUID, name: str, *, as_type: str, input: Any, metadata: dict | None = None) -> Iterator[Any]:
    """The root observation of a run's trace. trace name = name; run_id goes in trace metadata."""
    lf = client()
    if lf is None:
        yield NOOP
        return
    from langfuse import propagate_attributes

    trace_meta = {"run_id": str(run_id), **(metadata or {})}
    with (
        lf.start_as_current_observation(
            name=name, as_type=as_type, input=input, trace_context={"trace_id": trace_id_for(run_id)}
        ) as root,
        propagate_attributes(trace_name=name, tags=[name.split("-")[0]], metadata=trace_meta),
        _errors_marked(root),
    ):
        yield root


@contextmanager
def observe(
    name: str,
    *,
    as_type: str = "span",
    input: Any = None,
    metadata: dict | None = None,
    not_errors: tuple[type[BaseException], ...] = (),
) -> Iterator[Any]:
    """A child observation of whatever is current (or a new trace if nothing is), made current.
    not_errors: exceptions that pass through without marking it ERROR (LangGraph's interrupt())."""
    lf = client()
    if lf is None:
        yield NOOP
        return
    paused = None
    with lf.start_as_current_observation(name=name, as_type=as_type, input=input, metadata=metadata) as obs:
        try:
            with _errors_marked(obs, not_errors):
                yield obs
        except not_errors as e:
            # Must NOT pass through the span: OpenTelemetry would set status=ERROR on the way out,
            # and Langfuse shows that as level ERROR (seen live: every paused approve was red).
            paused = e
    if paused is not None:
        raise paused


@contextmanager
def _errors_marked(obs, not_errors: tuple[type[BaseException], ...] = ()) -> Iterator[None]:
    """An exception escaping the block -> level ERROR + its message, then re-raised unchanged.
    BaseException: a Ctrl+C or SimulatedCrash is a run that didn't finish, so it shows too."""
    try:
        yield
    except not_errors:
        raise
    except BaseException as e:
        obs.update(level="ERROR", status_message=f"{type(e).__name__}: {e}"[:500])
        raise


def start_generation(name: str, model: str, input: Any = None, *, as_type: str = "generation"):
    """A model call: started before the call, finished with end_generation(). Never has children,
    so it isn't made current (llm.py keeps its try/finally shape)."""
    lf = client()
    if lf is None:
        return NOOP
    return lf.start_observation(name=name, as_type=as_type, model=model, input=input)


def usage_details(record: dict) -> dict[str, int]:
    """Run-log token counts -> Langfuse usage buckets. Buckets must NOT overlap: OpenAI's
    input_tokens already includes cached_tokens, so input = uncached only."""
    inp, cached, out = (record.get(k) or 0 for k in ("input_tokens", "cached_tokens", "output_tokens"))
    usage = {"input": inp - cached, "output": out}
    if cached:
        usage["input_cached_tokens"] = cached
    return usage


def cost_details(model: str, usage: dict[str, int], prices: dict[str, dict[str, float]]) -> dict[str, float] | None:
    """USD per bucket from the price table (per 1M tokens), or None if the model isn't priced."""
    price = prices.get(model)
    if price is None:
        return None
    per_token = {
        "input": price.get("input", 0.0),
        "input_cached_tokens": price.get("cached_input", price.get("input", 0.0)),
        "output": price.get("output", 0.0),
    }
    return {k: n * per_token[k] / 1_000_000 for k, n in usage.items()}


ERROR_STATUSES = {"error", "incomplete", "invalid_output", "ungrounded", "interrupted"}


def end_generation(gen, record: dict, settings: Settings) -> None:
    """Finish a model-call observation from its run-log record; also puts trace_id and cost_usd
    INTO the record, so the JSONL log can be joined to the trace (and M4 can sum cost from logs)."""
    usage = usage_details(record)
    cost = cost_details(record.get("model", ""), usage, settings.prices)
    if cost is not None:
        record["cost_usd"] = round(sum(cost.values()), 8)
    if gen is NOOP:
        return
    record["trace_id"] = gen.trace_id
    status = record.get("status")
    gen.update(
        usage_details=usage,
        cost_details=cost,
        level="ERROR" if status in ERROR_STATUSES else "DEFAULT",
        status_message=record.get("error"),
        metadata={k: record.get(k) for k in ("status", "prompt_version", "request_id", "retries", "reasoning_tokens")},
    ).end()


def tool_level(status: str | None) -> str:
    """denied/invalid/rejected are the gate WORKING (warning); error = the tool broke."""
    return {"ok": "DEFAULT", "error": "ERROR"}.get(status or "", "WARNING")


def shutdown() -> None:
    """Send everything queued, then stop the exporter threads. CLIs call this before exiting."""
    lf = client() if _client is not None else None
    if lf is not None:
        lf.shutdown()


# --- D6 M4: eval traces + reading traces back for the cost/latency report -----------------------


@contextmanager
def eval_trace(stage: str, case: str, category: str) -> Iterator[Any]:
    """One root per eval case for stages with no run root of their own (answers, qualify, judging),
    so retrieval + model calls nest in ONE trace. Tagged eval/<stage> for the Langfuse UI."""
    lf = client()
    if lf is None:
        yield NOOP
        return
    from langfuse import propagate_attributes

    name = f"eval/{stage}/{case}"
    with (
        lf.start_as_current_observation(name=name, as_type="chain", input={"case": case}) as root,
        propagate_attributes(
            trace_name=name, tags=["eval", stage], metadata={"case": case, "category": category}
        ),
        _errors_marked(root),
    ):
        yield root


def fetch_observations(trace_id: str) -> list[dict]:
    """Every observation of a trace from the Langfuse API, as plain dicts (perf.py stays pure):
    {id, parent_id, type (lowercase), name, start, end (epoch seconds; end None = unfinished), cost, tokens}."""
    lf = client()
    if lf is None:
        raise RuntimeError("tracing is off (TRACING_ENABLED / LANGFUSE_* keys): no traces to read")
    out, cursor = [], None
    while True:
        page = lf.api.observations.get_many(trace_id=trace_id, limit=100, cursor=cursor, fields="core,basic,usage")
        for o in page.data:
            d = o.dict(by_alias=True)
            out.append({
                "id": d["id"],
                "parent_id": d.get("parentObservationId"),
                "type": str(getattr(d["type"], "value", d["type"])).lower(),
                "name": d["name"],
                "start": d["startTime"].timestamp(),
                "end": d["endTime"].timestamp() if d.get("endTime") else None,
                "cost": _cost(d.get("costDetails") or {}),
                "tokens": _cost(d.get("usageDetails") or {}),  # total incl. cached: NOT cache-timing dependent (M5)
            })  # fmt: skip
        cursor = page.meta.cursor if page.data else None
        if not cursor:
            return out


def _cost(details: dict) -> float:
    """Langfuse costDetails -> USD. It has a `total` key only SOMETIMES: calls with cached tokens came
    back as {input, output, input_cached_tokens} with no total (found in M4: they summed to $0)."""
    if "total" in details:
        return details["total"] or 0.0
    return sum(v for v in details.values() if isinstance(v, int | float))
