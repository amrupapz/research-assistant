import pytest

from research_assistant.permissions import ToolCall, authorize
from research_assistant.tools import GetLeadArgs


@pytest.mark.parametrize("name", ["search_company", "get_lead", "draft_email"])
def test_read_and_draft_tools_are_allowed(name):
    assert authorize(ToolCall(name)) == "allow"


def test_send_email_needs_approval():
    assert authorize(ToolCall("send_email")) == "needs_approval"


@pytest.mark.parametrize(
    "name",
    ["delete_lead", "run_shell", "", "Get_Lead", "get_lead ", "send_email\n", "SEND_EMAIL", "search_company2"],
)
def test_everything_else_is_denied(name):
    assert authorize(ToolCall(name)) == "deny"


def test_decision_ignores_args_for_now():
    assert authorize(ToolCall("get_lead", GetLeadArgs(lead_id="L1"))) == "allow"
