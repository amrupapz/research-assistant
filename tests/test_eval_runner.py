"""D6 M0: `evals/run.py --all` with fake stages and a fake meter: no API, no DB."""

import importlib.util
from contextlib import contextmanager
from pathlib import Path

import pytest

_spec = importlib.util.spec_from_file_location("eval_run", Path(__file__).parents[1] / "evals" / "run.py")
eval_run = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(eval_run)


@contextmanager
def fake_meter():
    usage: dict = {}
    yield usage
    usage.update(model_calls=3, input_tokens=100, output_tokens=20, seconds=1.5)


def test_runs_every_stage_in_order_and_makes_one_row_each():
    order = []

    def stage(name, result):
        return lambda: order.append(name) or result

    rows = eval_run.run_all(
        {"retrieval": stage("retrieval", (9, 10)), "agent": stage("agent", (16, 16))}, meter=fake_meter
    )
    assert order == ["retrieval", "agent"]
    assert [(r["stage"], r["passed"], r["total"], r["error"]) for r in rows] == [
        ("retrieval", 9, 10, None),
        ("agent", 16, 16, None),
    ]
    assert rows[0]["model_calls"] == 3


def test_a_crashing_stage_is_reported_and_the_rest_still_run():
    def boom():
        raise RuntimeError("db down")

    rows = eval_run.run_all({"answers": boom, "flow": lambda: (8, 8)}, meter=fake_meter)
    assert rows[0]["error"] == "RuntimeError: db down" and rows[0]["total"] == 0
    assert (rows[1]["passed"], rows[1]["error"]) == (8, None)


@pytest.mark.parametrize(("passed", "total", "rate"), [(15, 16, "94%"), (0, 0, "-")])
def test_summary_table(passed, total, rate):
    rows = [{"stage": "agent", "passed": passed, "total": total, "error": None,
             "model_calls": 90, "input_tokens": 1000, "output_tokens": 200, "seconds": 12.3}]  # fmt: skip
    out = eval_run.summary_table(rows)
    assert out.splitlines()[0].split() == ["stage", "passed", "rate", "calls", "tokens", "seconds"]
    assert out.splitlines()[1].split() == ["agent", f"{passed}/{total}", rate, "90", "1200", "12.3"]


def test_the_all_command_covers_every_stage():
    assert list(eval_run.stages(conn=None)) == ["retrieval", "answers", "agent", "flow"]
