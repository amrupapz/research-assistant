"""Tools the agent may PROPOSE. Our code validates and runs them; the model never does.

Each tool = a Pydantic args model (the full rules, enforced here) + a strict schema generated
from it (the shape the model sees) + a function. REGISTRY is the whitelist: nothing else runs.
"""

import re
from collections.abc import Callable
from dataclasses import dataclass, field
from email.message import EmailMessage
from pathlib import Path
from typing import Any

import psycopg
from pydantic import BaseModel, ConfigDict, Field, ValidationError, field_validator

from research_assistant.db import fetch_lead
from research_assistant.llm import ask
from research_assistant.retrieval import hybrid_search
from research_assistant.schemas import Draft, Lead

LEAD_ID = r"^L[0-9]{1,6}$"
DRAFT_ID = r"^D[0-9]{1,6}$"
ONE_LINE = r"^[^\r\n]*$"  # header values: a newline would let the model inject "Bcc: ..."

# Drafts are cold intros: they never need links or extra addresses, so ban both outright.
# Catches http(s)://, www., and bare domain+path ("evil.example/claim"); not obfuscations like
# "evil[.]example" -- those still face the human approval step.
LINK = re.compile(r"https?://|www\.|\b[a-z0-9-]+(?:\.[a-z0-9-]+)+/", re.IGNORECASE)
EMAIL = re.compile(r"[\w.+-]+@[\w-]+(?:\.[\w-]+)+")


class SearchCompanyArgs(BaseModel):
    model_config = ConfigDict(extra="forbid")  # -> "additionalProperties": false
    question: str = Field(
        min_length=3, max_length=300, description="A research question about a company, e.g. 'Who is Intel's CEO?'"
    )


class GetLeadArgs(BaseModel):
    model_config = ConfigDict(extra="forbid")
    lead_id: str = Field(pattern=LEAD_ID, description="Lead id, e.g. 'L1'.")


class DraftEmailArgs(BaseModel):
    model_config = ConfigDict(extra="forbid")
    lead_id: str = Field(pattern=LEAD_ID, description="Lead id of the recipient, e.g. 'L1'.")
    subject: str = Field(min_length=1, max_length=120, pattern=ONE_LINE, description="Email subject line (one line).")
    body: str = Field(
        min_length=1,
        max_length=2000,
        description="Plain-text email body. Use only facts returned by search_company. No links or email addresses.",
    )

    @field_validator("subject", "body")
    @classmethod
    def no_links_or_addresses(cls, value: str) -> str:
        """Error names the rule, never the matched text (it may be attacker-written)."""
        if LINK.search(value):
            raise ValueError("drafts may not contain links")
        if EMAIL.search(value):
            raise ValueError("drafts may not contain email addresses")
        return value


class SendEmailArgs(BaseModel):
    model_config = ConfigDict(extra="forbid")
    draft_id: str = Field(pattern=DRAFT_ID, description="Id of a draft from draft_email, e.g. 'D1'.")


class ToolError(Exception):
    """A problem the model could fix (unknown tool, bad args, unknown id). M2 sends it back as tool output."""


@dataclass
class ToolContext:
    """What tools may touch. Passed in, so tests can fake it."""

    conn: psycopg.Connection | Any
    drafts: dict[str, Draft] = field(default_factory=dict)  # in memory for now; Postgres in P4
    outbox_dir: Path = Path("outbox")  # "sending" = writing a .eml file here; never real email
    warnings: list[str] = field(default_factory=list)  # shown to the human at approval time


def _lead_or_error(ctx: ToolContext, lead_id: str) -> Lead:
    lead = fetch_lead(ctx.conn, lead_id)
    if lead is None:
        raise ToolError(f"no lead with id {lead_id!r}")
    return lead


def search_company(args: SearchCompanyArgs, ctx: ToolContext) -> dict:
    """Hybrid retrieval + the grounded P2 answer (a model call inside a tool call; both are logged)."""
    chunks = [chunk for chunk, _ in hybrid_search(ctx.conn, args.question, k=5)]
    result = ask(args.question, chunks)
    if result.needs_sources:  # missing / conflicting / outdated: the human should know before approving
        ctx.warnings.append(
            f"search_company({args.question!r}) flagged its sources as missing, conflicting, or unverified "
            f"(confidence={result.confidence}). Check any claims from it before sending."
        )
    return result.model_dump()


def get_lead(args: GetLeadArgs, ctx: ToolContext) -> dict:
    return _lead_or_error(ctx, args.lead_id).model_dump()


def _next_draft_id(ctx: ToolContext) -> str:
    """D1, D2, ... skipping ids already in memory OR already in outbox/ from an earlier run.

    Drafts live in memory (until P4), so every run restarts at D1; without the outbox check a
    new D1 would collide with last run's sent D1.eml.
    """
    n = len(ctx.drafts) + 1
    while f"D{n}" in ctx.drafts or (ctx.outbox_dir / f"D{n}.eml").exists():
        n += 1
    return f"D{n}"


def draft_email(args: DraftEmailArgs, ctx: ToolContext) -> dict:
    """Store a draft addressed to the lead's email from OUR table (the model never picks the address)."""
    lead = _lead_or_error(ctx, args.lead_id)
    draft = Draft(
        draft_id=_next_draft_id(ctx),
        lead_id=lead.lead_id,
        to=lead.email,
        subject=args.subject,
        body=args.body,
    )
    ctx.drafts[draft.draft_id] = draft
    return {**draft.model_dump(), "status": "draft (not sent)"}


def send_email(args: SendEmailArgs, ctx: ToolContext) -> dict:
    """"Send" a stored draft by writing outbox/<draft_id>.eml. Only runs after a human said yes.

    Sends exactly the stored draft: the model passes only a draft_id, so it can't change the
    recipient or text between your approval and the send. Each draft can be sent once.
    """
    draft = ctx.drafts.get(args.draft_id)
    if draft is None:
        raise ToolError(f"no draft with id {args.draft_id!r}")
    path = ctx.outbox_dir / f"{draft.draft_id}.eml"
    if path.exists():
        raise ToolError(f"draft {draft.draft_id} was already sent")
    msg = EmailMessage()
    msg["From"] = "sales-agent@example.com"
    msg["To"] = draft.to
    msg["Subject"] = draft.subject
    msg.set_content(draft.body)
    ctx.outbox_dir.mkdir(parents=True, exist_ok=True)
    path.write_bytes(bytes(msg))
    return {"draft_id": draft.draft_id, "to": draft.to, "status": "sent (written to outbox)", "path": str(path)}


@dataclass(frozen=True)
class Tool:
    name: str
    description: str
    args_model: type[BaseModel]
    fn: Callable[[Any, ToolContext], dict]


# Strict mode guarantees the SHAPE. Length limits etc. stay in Pydantic, enforced by us, so we
# only send keywords strict mode is known to accept.
STRICT_KEYWORDS = {"type", "description", "pattern"}


def tool_schema(tool: Tool) -> dict:
    """Strict Responses-API tool definition generated from the Pydantic args model (flat models only)."""
    js = tool.args_model.model_json_schema()
    props = {name: {k: v for k, v in prop.items() if k in STRICT_KEYWORDS} for name, prop in js["properties"].items()}
    if set(js.get("required", [])) != set(props):
        raise ValueError(f"{tool.name}: strict mode needs every field required (no defaults)")
    return {
        "type": "function",
        "name": tool.name,
        "description": tool.description,
        "parameters": {"type": "object", "properties": props, "required": list(props), "additionalProperties": False},
        "strict": True,
    }


REGISTRY: dict[str, Tool] = {
    t.name: t
    for t in [
        Tool(
            "search_company",
            "Research a company from our saved sources. Returns a cited answer and whether sources are missing.",
            SearchCompanyArgs,
            search_company,
        ),
        Tool("get_lead", "Look up a sales lead by id: name, title, company, email.", GetLeadArgs, get_lead),
        Tool(
            "draft_email",
            "Save an email draft to a lead. Does NOT send it. Returns the draft with its draft_id.",
            DraftEmailArgs,
            draft_email,
        ),
        Tool(
            "send_email",
            "Send a saved draft to its lead. Requires the user's approval; only use when the user asked to send.",
            SendEmailArgs,
            send_email,
        ),
    ]
}


def tool_definitions() -> list[dict]:
    """What we pass as `tools=` to respond()."""
    return [tool_schema(t) for t in REGISTRY.values()]


def _brief(e: ValidationError) -> str:
    """Field + reason per error, without echoing the (untrusted) input values."""
    return "; ".join(f"{'.'.join(map(str, err['loc'])) or '(root)'}: {err['msg']}" for err in e.errors())


def validate_call(name: str, raw_args: str) -> BaseModel:
    """Known tool + valid JSON-string arguments -> the parsed args model. ToolError otherwise."""
    tool = REGISTRY.get(name)
    if tool is None:
        raise ToolError(f"unknown tool {name!r}; available: {sorted(REGISTRY)}")
    try:
        return tool.args_model.model_validate_json(raw_args)
    except ValidationError as e:
        raise ToolError(f"invalid arguments for {name}: {_brief(e)}") from e


def execute(name: str, args: BaseModel, ctx: ToolContext) -> dict:
    """Run an already validated (and, in the loop, authorized) call."""
    return REGISTRY[name].fn(args, ctx)


def run_tool(name: str, raw_args: str, ctx: ToolContext) -> dict:
    """validate_call + execute, with NO permission check. Tests and scratch use only; the agent goes through authorize."""
    return execute(name, validate_call(name, raw_args), ctx)
