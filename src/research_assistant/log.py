"""Structured run logs: one JSON line per model call in runs/YYYY-MM-DD.jsonl."""

import json
import sys
from datetime import UTC, datetime
from pathlib import Path

from research_assistant import tracing


def log_run(record: dict, runs_dir: Path) -> Path:
    """Append one record (plus a UTC timestamp) as a JSON line; return the file written."""
    now = datetime.now(UTC)
    record = {"ts": now.isoformat(timespec="milliseconds"), **record}
    if "trace_id" not in record and (trace_id := tracing.current_trace_id()):
        record["trace_id"] = trace_id  # joins this line to its Langfuse trace; absent = not traced

    runs_dir.mkdir(parents=True, exist_ok=True)
    path = runs_dir / f"{now:%Y-%m-%d}.jsonl"
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(record, default=str) + "\n")

    # Human-readable one-liner on stderr (stdout stays just the answer).
    if record.get("kind") == "tool_call":
        print(
            f"[tool] step {record.get('step')} {record.get('tool')} "
            f"decision={record.get('decision', '-')} status={record.get('status')} {record.get('latency_ms')}ms",
            file=sys.stderr,
        )
        return path
    if record.get("kind") == "api":
        print(
            f"[api] {record.get('method')} {record.get('path')} {record.get('status')} "
            f"{record.get('latency_ms')}ms {record.get('request_id')}",
            file=sys.stderr,
        )
        return path
    print(
        f"[run] {record.get('status')} {record.get('latency_ms')}ms "
        f"in={record.get('input_tokens')} out={record.get('output_tokens')} "
        f"cached={record.get('cached_tokens')} {record.get('request_id')}",
        file=sys.stderr,
    )
    return path
