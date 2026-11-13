"""D7 M2: can_access(user, resource, action), pure, no DB. These fail until YOU write it (auth.py)."""

import pytest

from research_assistant.auth import Principal, Resource, can_access

MEMBER = Principal(key_id=1, tenant_id="acme", role="member")
APPROVER = Principal(key_id=2, tenant_id="acme", role="approver")
ACME = Resource(tenant_id="acme")
GLOBEX = Resource(tenant_id="globex")


@pytest.mark.parametrize("action", ["read", "start"])
def test_a_member_can_read_and_start_in_their_own_tenant(action):
    assert can_access(MEMBER, ACME, action) is True


def test_a_member_cannot_approve():
    assert can_access(MEMBER, ACME, "approve") is False


@pytest.mark.parametrize("action", ["read", "start", "approve"])
def test_an_approver_can_do_everything_in_their_own_tenant(action):
    assert can_access(APPROVER, ACME, action) is True


@pytest.mark.parametrize("user", [MEMBER, APPROVER])
@pytest.mark.parametrize("action", ["read", "start", "approve"])
def test_nobody_can_do_anything_in_another_tenant(user, action):
    assert can_access(user, GLOBEX, action) is False  # not even an approver; not even read


def test_an_unknown_role_is_denied():
    assert can_access(Principal(3, "acme", "admin"), ACME, "read") is False  # default-deny, no KeyError


def test_an_unknown_action_is_denied():
    assert can_access(APPROVER, ACME, "delete") is False


def test_tenant_ids_must_match_exactly():
    assert can_access(APPROVER, Resource(tenant_id="ACME"), "read") is False
    assert can_access(APPROVER, Resource(tenant_id="acme "), "read") is False
