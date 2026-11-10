"""Permission policy: which proposed (and already validated) tool calls may run."""

from dataclasses import dataclass
from typing import Literal

from pydantic import BaseModel

Decision = Literal["allow", "deny", "needs_approval"]

# Default-deny: only names listed here can ever run. A new tool is denied until someone adds it.
ALLOWED = frozenset(
    {"search_company", "get_lead", "lead_status", "list_followups", "qualify_lead", "draft_email"}
)  # read/draft: no outside effect
NEEDS_APPROVAL = frozenset({"send_email"})  # leaves the building: a human says y/n (M3)


@dataclass(frozen=True)
class ToolCall:
    """A call the model proposed, after validation. args is None if the tool has no args model yet."""

    name: str
    args: BaseModel | None = None


def authorize(call: ToolCall) -> Decision:
    """Decide whether this tool call may run.

    ✍️ YOU WRITE THIS (Project 3 core function).

    Contract:
      - "allow"          for the read/draft tools: search_company, get_lead, draft_email
      - "needs_approval" for send_email (M3: the CLI asks you y/n before it runs)
      - "deny"           for EVERYTHING else: unknown names, "", near-misses like
                         "Get_Lead" or "get_lead " (exact match only)
      - Always return one of the three strings above. Never raise for an unknown name.

    Hints:
      - Default-deny: list what's allowed and what needs approval; anything not listed is
        denied. Never write "deny if name in BAD_TOOLS": a tool you forgot to list would run.
      - Decide on call.name only for now. call.args is there for later rules
        (e.g. "drafts only to leads at companies we've researched").
      - ~6 lines. Run: uv run pytest tests/test_permissions.py -v
    """
    if call.name in ALLOWED:
        return "allow"
    if call.name in NEEDS_APPROVAL:
        return "needs_approval"
    return "deny"
