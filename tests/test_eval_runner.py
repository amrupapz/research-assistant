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
    assert list(eval_run.stages(conn=None)) == ["retrieval", "answers", "qualify", "agent", "flow"]


def test_category_rows_count_per_category_and_stage():
    results = [
        {"stage": "answers", "category": "freshness", "pass": True},
        {"stage": "agent", "category": "freshness", "pass": False},
        {"stage": "agent", "category": "freshness", "pass": True},
        {"stage": "qualify", "category": "prompt_injection", "pass": False},
    ]
    rows = {r["category"]: r for r in eval_run.category_rows(results)}
    assert (rows["freshness"]["passed"], rows["freshness"]["total"]) == (2, 3)
    assert rows["freshness"]["stages"] == {"answers": [1, 1], "agent": [1, 2]}
    assert rows["prompt_injection"]["stages"] == {"qualify": [0, 1]}


def test_category_table():
    rows = eval_run.category_rows([
        {"stage": "agent", "category": "freshness", "pass": True},
        {"stage": "agent", "category": "freshness", "pass": False},
    ])  # fmt: skip
    out = eval_run.category_table(rows, ["answers", "agent"])
    assert out.splitlines()[0].split() == ["category", "answers", "agent", "total", "rate"]
    assert out.splitlines()[1].split() == ["freshness", "-", "1/2", "1/2", "50%"]


def _answers_with_verdict(monkeypatch, verdict):
    from research_assistant.schemas import ClaimVerdict, ResearchAnswer

    chunk = type("C", (), {"chunk_id": "anthropic-funding#000"})()
    answer = ResearchAnswer(
        answer="Anthropic raised $22.5 billion [anthropic-funding#000].", confidence="high", assumptions=[],
        needs_sources=True, citations=[{"chunk_id": "anthropic-funding#000", "source_url": "https://x"}],
    )  # fmt: skip
    written = []
    monkeypatch.setattr(eval_run, "hybrid_search", lambda conn, q, k: [(chunk, 1.0)])
    monkeypatch.setattr(eval_run, "ask", lambda q, chunks: answer)
    monkeypatch.setattr(eval_run, "judge_claim", lambda claim, chunks: ClaimVerdict(verdict=verdict, reason="r"))
    monkeypatch.setattr(eval_run, "_write", written.append)
    result = eval_run.run_answers(None, only=["q06"], judge=True)
    return result, written[0]


def test_an_unsupported_claim_fails_an_answer_case(monkeypatch):
    (passed, total), record = _answers_with_verdict(monkeypatch, "unsupported")
    assert (passed, total) == (0, 1)
    assert record["no_unsupported_claims"] is False and record["pass"] is False


def test_a_partial_claim_only_warns(monkeypatch):
    (passed, total), record = _answers_with_verdict(monkeypatch, "partial")
    assert (passed, total) == (1, 1) and record["pass"] is True
    assert record["claim_warnings"] == ["partial: Anthropic raised $22.5 billion."]


def test_without_judge_flag_no_claim_check_is_added(monkeypatch):
    monkeypatch.setattr(eval_run, "judge_claim", lambda *a: pytest.fail("judged without --judge"))
    monkeypatch.setattr(eval_run, "hybrid_search", lambda conn, q, k: [])
    monkeypatch.setattr(eval_run, "_write", lambda r: None)
    from research_assistant.schemas import ResearchAnswer

    monkeypatch.setattr(eval_run, "ask", lambda q, c: ResearchAnswer(
        answer="The provided sources don't cover this.", confidence="low", assumptions=[], needs_sources=True,
        citations=[]))  # fmt: skip
    eval_run.run_answers(None, only=["q01"])


def test_calibration_refuses_to_run_until_everything_is_labelled(monkeypatch, tmp_path, capsys):
    path = tmp_path / "labels.jsonl"
    path.write_text('{"id": "j1", "claim": "c", "chunk_ids": [], "label": null, "origin": "t"}\n')
    monkeypatch.setattr(eval_run, "JUDGE_LABELS", path)
    monkeypatch.setattr(eval_run, "judge_claim", lambda *a: pytest.fail("judged before labelling"))
    assert eval_run.run_judge_calibration(conn=None) == (0, 0)
    assert "Label these first" in capsys.readouterr().out


def test_calibration_only_judges_the_selected_claims(monkeypatch, tmp_path):
    from research_assistant.schemas import ClaimVerdict

    path = tmp_path / "labels.jsonl"
    path.write_text(
        '{"id": "a1", "claim": "c", "chunk_ids": [], "label": "supported", "origin": "t", "labelled_by": "me"}\n'
        '{"id": "k1", "claim": "c", "chunk_ids": [], "label": null, "origin": "t"}\n'
    )
    judged = []
    monkeypatch.setattr(eval_run, "JUDGE_LABELS", path)
    monkeypatch.setattr(eval_run, "fetch_chunks", lambda conn, ids: [])
    monkeypatch.setattr(eval_run, "_write", lambda r: None)
    monkeypatch.setattr(eval_run, "judge_claim", lambda claim, chunks: judged.append(claim) or ClaimVerdict(verdict="supported", reason="r"))
    assert eval_run.run_judge_calibration(conn=None, only=["a1"]) == (1, 1)  # k1's missing label doesn't block a1
    assert eval_run.run_judge_calibration(conn=None, only=["k1"]) == (0, 0)  # ...but blocks itself
    assert len(judged) == 1


def test_each_answers_case_is_one_trace_and_records_its_id(monkeypatch, spans):
    """M4: retrieval + ask + judge nest under eval/answers/<case>, and the record points at that trace."""
    from research_assistant.schemas import ResearchAnswer

    written = []
    monkeypatch.setattr(eval_run, "hybrid_search", lambda conn, q, k: [])
    monkeypatch.setattr(eval_run, "ask", lambda q, c: ResearchAnswer(
        answer="The provided sources don't cover this.", confidence="low", assumptions=[], needs_sources=True,
        citations=[]))  # fmt: skip
    monkeypatch.setattr(eval_run, "_write", written.append)
    eval_run.run_answers(None, only=["q01"])
    (root,) = [s for s in spans() if s["parent"] is None]
    assert root["name"] == "eval/answers/q01"
    assert written[0]["trace_ids"] == [root["trace_id"]]
    assert root["attrs"]["langfuse.trace.tags"] == ("eval", "answers")


def test_records_carry_the_process_batch(monkeypatch, tmp_path):
    out = tmp_path / "evals.jsonl"
    monkeypatch.setattr(eval_run, "OUT", out)
    eval_run._write({"stage": "x"})
    import json

    assert json.loads(out.read_text())["batch"] == eval_run.BATCH
