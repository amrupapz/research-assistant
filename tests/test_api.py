# ruff: noqa: F811  (fake/saver are pytest fixtures imported from test_flow and used by name)
"""D7 M1: the HTTP API. Fake model + real test DB (rolled back), via FastAPI's TestClient.

TestClient runs background tasks right after the response, before the call returns. So
POST /runs → GET /runs/{id} sees the flow already paused at approve: the whole path, no sleeps.
"""

import json
import uuid

import pytest
from fastapi.testclient import TestClient
from test_agent import _settings
from test_flow import ANSWER, CHUNK, STRONG, fake, saver  # noqa: F401

from research_assistant import api, flow, llm, retrieval, state
from research_assistant.tools import ToolContext


class InlineRunner:
    """The real flow functions on the TEST connection (so everything rolls back)."""

    def __init__(self, ctx, saver):
        self.ctx, self.saver = ctx, saver

    def execute_flow(self, run_id, lead_id):
        self.ctx.run_id = run_id
        self.saver.threads.append(str(run_id))  # the saver fixture deletes these checkpoints
        flow.execute_flow(lead_id, self.ctx, self.saver)

    def continue_approval(self, run_id, draft_id, approved):
        self.ctx.run_id = run_id
        flow.continue_approval(draft_id, approved, self.ctx, self.saver)


@pytest.fixture
def settings(tmp_path, test_db_url):
    s = _settings(tmp_path / "runs")
    s.database_url = test_db_url
    return s


@pytest.fixture
def ctx(state_db, tmp_path):
    return ToolContext(conn=state_db, outbox_dir=tmp_path / "outbox")


@pytest.fixture
def client(settings, ctx, saver):
    app = api.create_app()
    app.dependency_overrides = {
        api.get_app_settings: lambda: settings,
        api.get_conn: lambda: ctx.conn,
        api.get_checkpointer: lambda: saver,
        api.get_runner: lambda: InlineRunner(ctx, saver),
    }
    with TestClient(app, raise_server_exceptions=False) as c:
        yield c


def _log(settings) -> list[dict]:
    return [json.loads(line) for f in sorted(settings.runs_dir.glob("*.jsonl")) for line in f.read_text().splitlines()]


# --- /health ------------------------------------------------------------------------------------


def test_health_ok(client):
    r = client.get("/health")
    assert (r.status_code, r.json()) == (200, {"db": "ok"})


def test_health_is_503_not_a_crash_when_postgres_is_down(client, settings):
    settings.database_url = "postgresql://localhost:1/nothing"  # nothing listens on port 1
    r = client.get("/health")
    assert r.status_code == 503
    assert r.json()["error"] == "database unavailable"
    assert "localhost:1" not in r.text  # no connection string, no driver message


# --- request id, errors, limits -----------------------------------------------------------------


def test_every_response_has_a_request_id_and_a_log_line(client, settings):
    r = client.get("/health", headers={"X-Request-ID": "abc-123"})
    assert r.headers["x-request-id"] == "abc-123"
    line = [x for x in _log(settings) if x["kind"] == "api"][-1]
    assert (line["request_id"], line["path"], line["status"]) == ("abc-123", "/health", 200)


def test_a_junk_request_id_is_replaced(client):
    r = client.get("/health", headers={"X-Request-ID": "x\ny" * 50})
    assert r.headers["x-request-id"] != "x\ny" * 50 and len(r.headers["x-request-id"]) == 32


def test_the_log_groups_by_route_not_by_id(client, settings):
    client.get(f"/runs/{uuid.uuid4()}")
    assert [x for x in _log(settings) if x["kind"] == "api"][-1]["path"] == "/runs/{run_id}"


def test_a_crash_returns_500_without_internals(client, settings, monkeypatch):
    def boom(*a, **k):
        raise RuntimeError("secret detail sk-live-123 in /Users/me/code")

    monkeypatch.setattr(api, "fetch_lead", boom)
    r = client.post("/runs", json={"lead_id": "L1"}, headers={"X-Request-ID": "crash-1"})
    assert (r.status_code, r.json()) == (500, {"error": "internal error", "request_id": "crash-1"})
    assert "sk-live" not in r.text and "Traceback" not in r.text
    logged = [x for x in _log(settings) if x["kind"] == "api_error"][-1]
    assert logged["request_id"] == "crash-1" and "secret detail" in logged["error"]  # the detail is in OUR log


def test_body_over_the_limit_is_413(client):
    r = client.post("/ask", content=b"{" + b" " * api.MAX_BODY_BYTES + b"}", headers={"content-type": "application/json"})
    assert r.status_code == 413 and "request_id" in r.json()


def test_chunked_body_over_the_limit_is_413(client):
    def chunks():
        for _ in range(api.MAX_BODY_BYTES // 1000 + 2):
            yield b" " * 1000

    r = client.post("/ask", content=chunks(), headers={"content-type": "application/json"})  # no Content-Length
    assert r.status_code == 413


@pytest.mark.parametrize(
    ("path", "body"),
    [
        ("/ask", {"question": "x" * 2001}),
        ("/ask", {"question": "ok?", "k": 500}),
        ("/ask", {"question": "ok?", "admin": True}),  # unknown field
        ("/runs", {"lead_id": "L1; DROP TABLE leads"}),
        ("/drafts/D1/approve", {"approved": "yes"}),  # StrictBool
        ("/drafts/X1/approve", {"approved": True}),  # bad id in the path
    ],
)
def test_invalid_input_is_422_without_echoing_it(client, path, body):
    r = client.post(path, json=body)
    assert r.status_code == 422
    assert "DROP TABLE" not in r.text and "x" * 100 not in r.text


# --- /ask ---------------------------------------------------------------------------------------


def test_ask_returns_the_answer_and_its_sources(client, monkeypatch):
    monkeypatch.setattr(retrieval, "hybrid_search", lambda conn, q, k: [(CHUNK, 1.0)])
    monkeypatch.setattr(llm, "ask", lambda q, chunks: ANSWER)
    r = client.post("/ask", json={"question": "What does Figma do?"})
    assert r.status_code == 200
    assert r.json()["result"]["answer"] == ANSWER.answer
    assert r.json()["sources"] == [{"chunk_id": "figma-about#000", "source_url": "https://figma.com/about", "fetched_at": "2026-11-01T00:00:00"}]


def test_ask_truncated_model_output_is_502(client, monkeypatch):
    def truncated(q, chunks):
        raise llm.IncompleteResponseError("max_output_tokens")

    monkeypatch.setattr(retrieval, "hybrid_search", lambda conn, q, k: [(CHUNK, 1.0)])
    monkeypatch.setattr(llm, "ask", truncated)
    r = client.post("/ask", json={"question": "What does Figma do?"})
    assert r.status_code == 502 and "max_output_tokens" not in r.text


# --- runs + approval, end to end ----------------------------------------------------------------


def _start(client, fake):
    fake["verdicts"].append(STRONG)
    r = client.post("/runs", json={"lead_id": "L1"})
    assert r.status_code == 202
    return r.json()["run_id"]


def test_start_returns_202_and_the_run_pauses_with_its_draft_visible(client, fake):
    run_id = _start(client, fake)
    r = client.get(f"/runs/{run_id}")
    assert r.status_code == 200
    view = r.json()
    assert (view["status"], view["lead_id"], view["fit"]) == ("waiting", "L1", "strong")
    assert view["draft"]["draft_id"] == "D1" and view["draft"]["to"] == "dana.kim@example.com"
    assert view["draft"]["body"].startswith("Hello Dana")  # read it BEFORE approving (P5 carry)


def test_approve_yes_sends_once_and_a_second_approve_is_409(client, fake, ctx):
    run_id = _start(client, fake)
    r = client.post("/drafts/D1/approve", json={"approved": True})
    assert (r.status_code, r.json()["run_id"]) == (202, run_id)
    view = client.get(f"/runs/{run_id}").json()
    assert view["status"] == "done" and view["draft"] is None
    assert (ctx.outbox_dir / "D1.eml").exists()
    again = client.post("/drafts/D1/approve", json={"approved": True})
    assert again.status_code == 409 and "not waiting" in again.json()["error"]
    assert ctx.conn.execute("SELECT count(*) FROM sends").fetchone()[0] == 1


def test_approve_no_sends_nothing(client, fake, ctx):
    run_id = _start(client, fake)
    client.post("/drafts/D1/approve", json={"approved": False})
    assert client.get(f"/runs/{run_id}").json()["outcome"] == "declined: D1 not sent"
    assert not (ctx.outbox_dir / "D1.eml").exists()


def test_approving_a_draft_that_does_not_exist_is_409(client):
    assert client.post("/drafts/D99/approve", json={"approved": True}).status_code == 409


def test_a_lost_race_cannot_approve(client, fake, ctx):
    """Someone else claimed the run between our checks and our UPDATE: only one claim wins."""
    run_id = _start(client, fake)
    assert state.claim_waiting(ctx.conn, uuid.UUID(run_id)) is True  # the other request won
    assert client.post("/drafts/D1/approve", json={"approved": True}).status_code == 409
    assert ctx.conn.execute("SELECT count(*) FROM approvals").fetchone()[0] == 0


def test_unknown_lead_is_404_and_creates_no_run(client, ctx):
    r = client.post("/runs", json={"lead_id": "L999"})
    assert r.status_code == 404
    assert ctx.conn.execute("SELECT count(*) FROM runs").fetchone()[0] == 0


def test_unknown_run_is_404(client):
    assert client.get(f"/runs/{uuid.uuid4()}").status_code == 404


def test_a_run_that_is_not_a_flow_run_is_404(client, ctx):
    run_id = state.start_run(ctx.conn, "agent task", "test-model", "agent-v5")
    assert client.get(f"/runs/{run_id}").status_code == 404


def test_lead_status(client):
    r = client.get("/leads/L1/status")
    assert r.status_code == 200 and r.json()["lead"]["company"] == "Figma"
    assert client.get("/leads/L999/status").status_code == 404
