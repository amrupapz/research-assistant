"""HTTP API (D7 M1). The ONLY file that imports fastapi.

    uv run api            → http://127.0.0.1:8000/docs

Long work never blocks a request: POST /runs creates the runs row and returns 202 + run_id; the
graph runs in a background task; GET /runs/{id} polls. Approve = claim (fast, atomic, here) +
continue (slow, background). Same flow.py functions as the CLI, so the same guarantees hold
(interrupt() approval, send_once, resume after a crash).

Errors never carry a stack trace or exception text: {"error", "request_id"}. The details go to the
runs/ log under the same request_id.

Auth (D7 M2): `Authorization: Bearer <key>` on everything except /health. The key → a Principal
(tenant + role); the request's connection is then set to that tenant, so every query only sees its
rows. Codes: 401 no/bad/revoked key · 404 missing OR another tenant's (same body, can't tell apart)
· 403 own tenant, role not allowed (a member approving).
"""

import re
import time
import traceback
import uuid
from collections.abc import Iterator
from contextlib import asynccontextmanager, contextmanager
from datetime import datetime
from typing import Annotated, Literal
from uuid import UUID

import psycopg
from fastapi import BackgroundTasks, Depends, FastAPI, HTTPException, Request
from fastapi import Path as PathParam
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from pydantic import BaseModel, ConfigDict, Field, StrictBool

from research_assistant import flow, state, tracing
from research_assistant.auth import Principal, Resource, can_access, find_principal
from research_assistant.config import Settings, get_settings
from research_assistant.db import connect, set_tenant
from research_assistant.flow import FlowError
from research_assistant.llm import IncompleteResponseError, UngroundedAnswerError
from research_assistant.log import log_run
from research_assistant.schemas import ResearchAnswer
from research_assistant.tools import (
    DRAFT_ID,
    LEAD_ID,
    LeadStatusArgs,
    ToolContext,
    lead_status,
)

MAX_BODY_BYTES = 16 * 1024  # our biggest real body is a ~2,000-char question
REQUEST_ID = re.compile(r"^[A-Za-z0-9_-]{1,64}$")  # a client's X-Request-ID goes into logs: no newlines, no junk


# --- request / response models -------------------------------------------------------------------
# extra="forbid": an unknown field is a 422, not silently ignored (catches typos and probing).


class AskRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")
    question: str = Field(min_length=3, max_length=2000)
    k: int = Field(default=5, ge=1, le=10)


class Source(BaseModel):
    chunk_id: str
    source_url: str
    fetched_at: datetime


class AskResponse(BaseModel):
    result: ResearchAnswer
    sources: list[Source]  # the cited chunks, with where and when they came from


class StartRunRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")
    lead_id: str = Field(pattern=LEAD_ID)


class Accepted(BaseModel):
    run_id: UUID
    status: Literal["running"] = "running"


class DraftView(BaseModel):
    draft_id: str
    to: str
    subject: str
    body: str


class RunView(BaseModel):
    run_id: UUID
    status: str  # running | waiting | done | failed
    lead_id: str | None
    outcome: str | None
    fit: str | None
    warnings: list[str]
    draft: DraftView | None  # only while waiting: read it BEFORE approving


class ApproveRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")
    approved: StrictBool  # StrictBool: "yes" / 1 are a 422, not a silent True


class Health(BaseModel):
    db: Literal["ok", "down"]


# --- middleware: request id, body limit, access log ----------------------------------------------


class BodyTooLarge(HTTPException):
    def __init__(self) -> None:
        super().__init__(413, f"request body over {MAX_BODY_BYTES} bytes")


class Guard:
    """Plain ASGI middleware (outermost), so it sees every request, including ones FastAPI rejects:

    - request id: the client's X-Request-ID if it's sane, else a new one; on request.state, in the
      response header, in every error body and log line.
    - body limit: Content-Length over the limit → 413 before reading anything; a body without
      Content-Length (chunked) is counted as it arrives and cut off at the limit.
    - access log: one runs/ line per request (method, route, status, latency) for M6's alerts."""

    def __init__(self, app, settings_fn) -> None:
        self.app = app
        self.settings_fn = settings_fn

    async def __call__(self, scope, receive, send):
        if scope["type"] != "http":
            return await self.app(scope, receive, send)
        headers = dict(scope["headers"])
        rid = headers.get(b"x-request-id", b"").decode("latin-1")
        rid = rid if REQUEST_ID.match(rid) else uuid.uuid4().hex
        scope.setdefault("state", {})["request_id"] = rid
        started = time.perf_counter()
        status = 500

        async def send_with_id(message):
            nonlocal status
            if message["type"] == "http.response.start":
                status = message["status"]
                message["headers"] = [*message.get("headers", []), (b"x-request-id", rid.encode())]
            await send(message)

        received = 0

        async def limited_receive():
            nonlocal received
            message = await receive()
            if message["type"] == "http.request":
                received += len(message.get("body", b""))
                if received > MAX_BODY_BYTES:
                    raise BodyTooLarge()  # FastAPI re-raises HTTPException from body reading → 413
            return message

        try:
            length = headers.get(b"content-length")
            if length is not None and (not length.isdigit() or int(length) > MAX_BODY_BYTES):
                await error(413, f"request body over {MAX_BODY_BYTES} bytes", rid)(scope, receive, send_with_id)
            else:
                await self.app(scope, limited_receive, send_with_id)
        finally:
            route = scope.get("route")
            principal = scope["state"].get("principal")  # set by get_principal; None = no/bad key
            log_run(
                {
                    "kind": "api",
                    "request_id": rid,
                    "key_id": principal.key_id if principal else None,  # never the key itself
                    "tenant_id": principal.tenant_id if principal else None,
                    "method": scope["method"],
                    "path": getattr(route, "path", scope["path"]),  # /runs/{run_id}, not the id: groups for p95
                    "status": status,
                    "latency_ms": round((time.perf_counter() - started) * 1000),
                },
                self.settings_fn().runs_dir,
            )


def error(status: int, message: str, request_id: str) -> JSONResponse:
    return JSONResponse({"error": message, "request_id": request_id}, status_code=status)


def _rid(request: Request) -> str:
    return getattr(request.state, "request_id", "-")


# --- dependencies (tests override these) ---------------------------------------------------------


def get_app_settings() -> Settings:
    return get_settings()


def get_conn(settings: Annotated[Settings, Depends(get_app_settings)]) -> Iterator[psycopg.Connection]:
    """One autocommit connection per request (each `with conn.transaction()` commits).
    NO tenant yet: tenant queries raise until get_tenant_conn sets it from the checked key."""
    try:
        conn = connect(settings.database_url, autocommit=True, tenant=None)
    except psycopg.OperationalError as e:
        raise HTTPException(503, "database unavailable") from e
    with conn:
        yield conn


UNAUTHORIZED = HTTPException(401, "missing or invalid API key", headers={"WWW-Authenticate": "Bearer"})


def get_principal(request: Request, conn: Annotated[psycopg.Connection, Depends(get_conn)]) -> Principal:
    """Who is calling. Unknown, revoked and malformed keys all get the same 401."""
    scheme, _, key = request.headers.get("authorization", "").partition(" ")
    if scheme.lower() != "bearer" or not key or len(key) > 128:
        raise UNAUTHORIZED
    principal = find_principal(conn, key.strip())
    if principal is None:
        raise UNAUTHORIZED
    request.state.principal = principal  # for the access log
    return principal


def get_tenant_conn(
    conn: Annotated[psycopg.Connection, Depends(get_conn)], principal: Annotated[Principal, Depends(get_principal)]
) -> psycopg.Connection:
    """The request's connection, scoped to the caller's tenant. Endpoints only ever get THIS one."""
    set_tenant(conn, principal.tenant_id)
    return conn


NOT_FOUND = HTTPException(404, "not found")  # no id, no state: nothing to learn from it


def check(conn: psycopg.Connection, user: Principal, kind: str, key, action: str) -> None:
    """404 if the resource is missing or another tenant's (indistinguishable); 403 if it is ours but
    the role can't do `action`. Two layers: the tenant-filtered lookup AND can_access()."""
    tenant_id = state.owner(conn, kind, key)
    if tenant_id is None or not can_access(user, Resource(tenant_id), "read"):
        raise NOT_FOUND
    if not can_access(user, Resource(tenant_id), action):
        raise HTTPException(403, f"your role can't {action}")


def get_checkpointer(settings: Annotated[Settings, Depends(get_app_settings)]):
    from langgraph.checkpoint.postgres import PostgresSaver

    try:
        cp_conn = flow.checkpointer_conn(settings.database_url)
    except psycopg.OperationalError as e:
        raise HTTPException(503, "database unavailable") from e
    with cp_conn:
        yield PostgresSaver(cp_conn)


class Runner:
    """Runs the slow half of a flow AFTER the response is sent. Each job opens its OWN connections:
    the request's connection is closed by then. A crash leaves the run 'running' or 'failed' in
    Postgres → `flow --resume <run_id>`, exactly like the CLI."""

    def __init__(self, settings: Settings) -> None:
        self.settings = settings

    @contextmanager
    def _ctx(self, run_id: UUID, tenant_id: str):
        from langgraph.checkpoint.postgres import PostgresSaver

        url = self.settings.database_url
        with connect(url, autocommit=True, tenant=tenant_id) as conn, flow.checkpointer_conn(url) as cp_conn:
            yield ToolContext(conn=conn, run_id=run_id), PostgresSaver(cp_conn)

    def execute_flow(self, run_id: UUID, lead_id: str, tenant_id: str) -> None:
        self._job(run_id, tenant_id, lambda ctx, cp: flow.execute_flow(lead_id, ctx, cp))

    def continue_approval(self, run_id: UUID, draft_id: str, approved: bool, tenant_id: str) -> None:
        self._job(run_id, tenant_id, lambda ctx, cp: flow.continue_approval(draft_id, approved, ctx, cp))

    def _job(self, run_id: UUID, tenant_id: str, fn) -> None:
        try:
            with self._ctx(run_id, tenant_id) as (ctx, cp):
                fn(ctx, cp)
        except Exception:  # noqa: BLE001 -- the run row already says 'failed'; nobody awaits this response
            log_run({"kind": "api_job", "run_id": run_id, "status": "error", "error": traceback.format_exc()}, self.settings.runs_dir)


def get_runner(settings: Annotated[Settings, Depends(get_app_settings)]) -> Runner:
    return Runner(settings)


Conn = Annotated[psycopg.Connection, Depends(get_tenant_conn)]  # always tenant-scoped
User = Annotated[Principal, Depends(get_principal)]
Checkpointer = Annotated[object, Depends(get_checkpointer)]
LeadId = Annotated[str, PathParam(pattern=LEAD_ID)]
DraftId = Annotated[str, PathParam(pattern=DRAFT_ID)]


# --- the app -------------------------------------------------------------------------------------


def create_app() -> FastAPI:
    @asynccontextmanager
    async def lifespan(app: FastAPI):
        yield
        tracing.shutdown()  # send queued spans before the process exits

    app = FastAPI(title="Research Assistant", version="0.7.0", lifespan=lifespan)
    app.add_middleware(Guard, settings_fn=lambda: app.dependency_overrides.get(get_app_settings, get_app_settings)())

    @app.exception_handler(HTTPException)
    async def http_error(request: Request, exc: HTTPException):
        response = error(exc.status_code, str(exc.detail), _rid(request))
        response.headers.update(exc.headers or {})  # e.g. WWW-Authenticate on a 401
        return response

    @app.exception_handler(RequestValidationError)
    async def invalid(request: Request, exc: RequestValidationError):
        # Location + message only: the default also echoes the INPUT back (could be huge, or a secret).
        detail = [{"loc": list(e["loc"]), "msg": e["msg"]} for e in exc.errors()]
        return JSONResponse({"error": "invalid request", "detail": detail, "request_id": _rid(request)}, status_code=422)

    @app.exception_handler(FlowError)
    async def conflict(request: Request, exc: FlowError):
        return error(409, str(exc), _rid(request))  # our own messages (ids + states), no internals

    @app.exception_handler(Exception)
    async def crashed(request: Request, exc: Exception):
        settings = app.dependency_overrides.get(get_app_settings, get_app_settings)()
        log_run(
            {"kind": "api_error", "request_id": _rid(request), "path": request.url.path, "error": traceback.format_exc()},
            settings.runs_dir,
        )
        return error(500, "internal error", _rid(request))

    @app.get("/health", responses={503: {"model": Health}})
    def health(settings: Annotated[Settings, Depends(get_app_settings)]) -> Health:
        """Can we reach Postgres? 503 (not a crash) when it's down. No versions, no error text."""
        try:
            with psycopg.connect(settings.database_url, connect_timeout=2) as conn:
                conn.execute("SELECT 1")
        except psycopg.Error:
            raise HTTPException(503, "database unavailable") from None
        return Health(db="ok")

    @app.post("/ask")
    def ask_endpoint(body: AskRequest, conn: Conn) -> AskResponse:
        """A grounded answer with citations. Synchronous: one retrieval + one model call."""
        from research_assistant.llm import ask
        from research_assistant.retrieval import hybrid_search

        with tracing.observe("ask", as_type="chain", input={"question": body.question, "k": body.k}) as span:
            chunks = [chunk for chunk, _ in hybrid_search(conn, body.question, k=body.k)]
            try:
                result = ask(body.question, chunks)
            except (IncompleteResponseError, UngroundedAnswerError) as e:
                raise HTTPException(502, "the model did not return a usable answer") from e
            span.update(output=result.answer)
        by_id = {c.chunk_id: c for c in chunks}
        cited = [by_id[c.chunk_id] for c in result.citations if c.chunk_id in by_id]
        return AskResponse(result=result, sources=[Source(**c.model_dump(include={"chunk_id", "source_url", "fetched_at"})) for c in cited])

    @app.post("/runs", status_code=202)
    def start_run(
        body: StartRunRequest,
        user: User,
        conn: Conn,
        background: BackgroundTasks,
        runner: Annotated[Runner, Depends(get_runner)],
        settings: Annotated[Settings, Depends(get_app_settings)],
    ) -> Accepted:
        """Start the workflow for a lead. Returns at once; poll GET /runs/{run_id}."""
        check(conn, user, "lead", body.lead_id, "start")  # before creating anything
        run_id = flow.start_flow_run(body.lead_id, ToolContext(conn=conn), settings)
        background.add_task(runner.execute_flow, run_id, body.lead_id, user.tenant_id)
        return Accepted(run_id=run_id)

    @app.get("/runs/{run_id}")
    def get_run(run_id: UUID, user: User, conn: Conn, checkpointer: Checkpointer) -> RunView:
        check(conn, user, "run", run_id, "read")
        view = flow.run_view(run_id, ToolContext(conn=conn), checkpointer)
        if view is None:  # ours, but not a workflow run (an `agent` CLI run)
            raise NOT_FOUND
        return RunView(**view)

    @app.post("/drafts/{draft_id}/approve", status_code=202)
    def approve(
        draft_id: DraftId,
        body: ApproveRequest,
        user: User,
        conn: Conn,
        checkpointer: Checkpointer,
        background: BackgroundTasks,
        runner: Annotated[Runner, Depends(get_runner)],
    ) -> Accepted:
        """Approve or decline a waiting draft. The claim (waiting → running) happens NOW, so a second
        call gets 409; sending happens in the background (send_once). Poll GET /runs/{run_id}.
        Not ours / missing → 404; a member → 403 (approvers only)."""
        check(conn, user, "draft", draft_id, "approve")
        run_id = flow.claim_approval(draft_id, ToolContext(conn=conn), checkpointer)  # FlowError → 409
        background.add_task(runner.continue_approval, run_id, draft_id, body.approved, user.tenant_id)
        return Accepted(run_id=run_id)

    @app.get("/leads/{lead_id}/status")
    def get_lead_status(lead_id: LeadId, user: User, conn: Conn) -> dict:
        """What we remember about a lead: outreach, replies, latest qualification and research."""
        check(conn, user, "lead", lead_id, "read")
        return lead_status(LeadStatusArgs(lead_id=lead_id), ToolContext(conn=conn))

    return app


app = create_app()


def main() -> None:
    import uvicorn

    uvicorn.run("research_assistant.api:app", host="127.0.0.1", port=8000)  # M4 (Docker) binds 0.0.0.0 inside the container
