"""Tools the agent may PROPOSE. Our code validates and runs them; the model never does.

Each tool = a Pydantic args model (the full rules, enforced here) + a strict schema generated
from it (the shape the model sees) + a function. REGISTRY is the whitelist: nothing else runs.
"""

from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

import psycopg
from pydantic import BaseModel, ConfigDict, Field, ValidationError

from research_assistant.db import fetch_lead
from research_assistant.llm import ask
from research_assistant.retrieval import hybrid_search
from research_assistant.schemas import Draft, Lead

LEAD_ID = r"^L[0-9]{1,6}$"


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
    subject: str = Field(min_length=1, max_length=120, description="Email subject line.")
    body: str = Field(
        min_length=1,
        max_length=2000,
        description="Plain-text email body. Use only facts returned by search_company.",
    )


class ToolError(Exception):
    """A problem the model could fix (unknown tool, bad args, unknown id). M2 sends it back as tool output."""


@dataclass
class ToolContext:
    """What tools may touch. Passed in, so tests can fake it."""

    conn: psycopg.Connection | Any
    drafts: dict[str, Draft] = field(default_factory=dict)  # in memory for now; Postgres in P4


def _lead_or_error(ctx: ToolContext, lead_id: str) -> Lead:
    lead = fetch_lead(ctx.conn, lead_id)
    if lead is None:
        raise ToolError(f"no lead with id {lead_id!r}")
    return lead


def search_company(args: SearchCompanyArgs, ctx: ToolContext) -> dict:
    """Hybrid retrieval + the grounded P2 answer (a model call inside a tool call; both are logged)."""
    chunks = [chunk for chunk, _ in hybrid_search(ctx.conn, args.question, k=5)]
    return ask(args.question, chunks).model_dump()


def get_lead(args: GetLeadArgs, ctx: ToolContext) -> dict:
    return _lead_or_error(ctx, args.lead_id).model_dump()


def draft_email(args: DraftEmailArgs, ctx: ToolContext) -> dict:
    """Store a draft addressed to the lead's email from OUR table (the model never picks the address)."""
    lead = _lead_or_error(ctx, args.lead_id)
    draft = Draft(
        draft_id=f"D{len(ctx.drafts) + 1}",
        lead_id=lead.lead_id,
        to=lead.email,
        subject=args.subject,
        body=args.body,
    )
    ctx.drafts[draft.draft_id] = draft
    return {**draft.model_dump(), "status": "draft (not sent)"}


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
    ]
}


def tool_definitions() -> list[dict]:
    """What we pass as `tools=` to respond()."""
    return [tool_schema(t) for t in REGISTRY.values()]


def _brief(e: ValidationError) -> str:
    """Field + reason per error, without echoing the (untrusted) input values."""
    return "; ".join(f"{'.'.join(map(str, err['loc'])) or '(root)'}: {err['msg']}" for err in e.errors())


def run_tool(name: str, raw_args: str, ctx: ToolContext) -> dict:
    """Look up the tool, validate the model's JSON-string arguments, run it. ToolError on anything fixable."""
    tool = REGISTRY.get(name)
    if tool is None:
        raise ToolError(f"unknown tool {name!r}; available: {sorted(REGISTRY)}")
    try:
        args = tool.args_model.model_validate_json(raw_args)
    except ValidationError as e:
        raise ToolError(f"invalid arguments for {name}: {_brief(e)}") from e
    return tool.fn(args, ctx)
