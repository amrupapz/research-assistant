"""API keys, tenants, roles (D7 M2). Keys are managed from this CLI only; there is no admin endpoint.

    uv run keys add-tenant acme
    uv run keys create-key --tenant acme --role approver --label "Dana laptop"   (key shown ONCE)
    uv run keys list-keys
    uv run keys revoke-key 3

A key is 32 random bytes. The DB stores only its sha256: a leaked table gives nobody a working key.
A fast hash is fine HERE because the key is random (2^256 guesses). Passwords are guessable and
need a slow hash (bcrypt/argon2); random tokens don't.
"""

import argparse
import hashlib
import secrets
from dataclasses import dataclass
from typing import Literal

import psycopg

Role = Literal["member", "approver"]
Action = Literal["read", "start", "approve"]

# What each role may do INSIDE its own tenant. Approvers can do everything members can.
ROLE_ACTIONS: dict[str, set[str]] = {
    "member": {"read", "start"},
    "approver": {"read", "start", "approve"},
}

KEY_PREFIX = "ra_"  # recognisable in a leaked log or a secret scanner


@dataclass(frozen=True)
class Principal:
    """Who is calling: one API key = one principal."""

    key_id: int
    tenant_id: str
    role: str


@dataclass(frozen=True)
class Resource:
    """Anything a tenant owns (lead, run, draft). Only the owner matters for access."""

    tenant_id: str


def can_access(user: Principal, resource: Resource, action: str) -> bool:
    """✍️ YOURS (D7 M2). True only if `user` may do `action` on `resource`.

    Hint: two checks, both must pass. (1) Same tenant? Compare the two tenant_ids. (2) Does the
    user's role allow this action? ROLE_ACTIONS maps role -> allowed actions; an unknown role or
    action must be False (default-deny, like authorize()). Use .get() so a bad role can't KeyError.
    About 3 lines. tests/test_auth.py says what "right" means.
    """
    if user.tenant_id != resource.tenant_id:
        return False
    allowed = ROLE_ACTIONS.get(user.role, set())
    return action in allowed


# --- keys -----------------------------------------------------------------------------------


def new_key() -> str:
    return KEY_PREFIX + secrets.token_urlsafe(32)


def hash_key(key: str) -> str:
    return hashlib.sha256(key.encode()).hexdigest()


def find_principal(conn: psycopg.Connection, key: str) -> Principal | None:
    """The principal for a presented key, or None (unknown OR revoked: the caller can't tell which)."""
    row = conn.execute(
        "SELECT key_id, tenant_id, role FROM api_keys WHERE key_sha256 = %s AND revoked_at IS NULL", [hash_key(key)]
    ).fetchone()
    return Principal(*row) if row else None


def add_tenant(conn: psycopg.Connection, tenant_id: str) -> None:
    conn.execute("INSERT INTO tenants (tenant_id) VALUES (%s)", [tenant_id])


def create_key(conn: psycopg.Connection, tenant_id: str, role: str, label: str = "") -> tuple[int, str]:
    """(key_id, key). The key is returned once and never stored: lose it → revoke + create a new one."""
    key = new_key()
    key_id = conn.execute(
        "INSERT INTO api_keys (tenant_id, role, key_sha256, label) VALUES (%s, %s, %s, %s) RETURNING key_id",
        [tenant_id, role, hash_key(key), label],
    ).fetchone()[0]
    return key_id, key


def list_keys(conn: psycopg.Connection) -> list[dict]:
    cur = conn.execute("SELECT key_id, tenant_id, role, label, created_at, revoked_at FROM api_keys ORDER BY key_id")
    cols = [c.name for c in cur.description]
    return [dict(zip(cols, row, strict=True)) for row in cur.fetchall()]


def revoke_key(conn: psycopg.Connection, key_id: int) -> bool:
    cur = conn.execute("UPDATE api_keys SET revoked_at = now() WHERE key_id = %s AND revoked_at IS NULL", [key_id])
    return cur.rowcount == 1


# --- CLI ------------------------------------------------------------------------------------


def main(argv: list[str] | None = None) -> None:
    from research_assistant.db import connect

    p = argparse.ArgumentParser(prog="keys", description="Manage tenants and API keys (no HTTP endpoint for this).")
    sub = p.add_subparsers(dest="cmd", required=True)
    sub.add_parser("add-tenant").add_argument("tenant_id")
    c = sub.add_parser("create-key")
    c.add_argument("--tenant", required=True)
    c.add_argument("--role", required=True, choices=sorted(ROLE_ACTIONS))
    c.add_argument("--label", default="")
    sub.add_parser("list-keys")
    sub.add_parser("revoke-key").add_argument("key_id", type=int)
    args = p.parse_args(argv)

    try:
        with connect(tenant=None) as conn:  # tenants + keys aren't tenant-scoped; commits on clean exit
            if args.cmd == "add-tenant":
                add_tenant(conn, args.tenant_id)
                print(f"tenant {args.tenant_id} added")
            elif args.cmd == "create-key":
                key_id, key = create_key(conn, args.tenant, args.role, args.label)
                print(f"key {key_id} ({args.tenant}, {args.role}). Shown ONCE, store it now:\n{key}")
            elif args.cmd == "list-keys":
                for k in list_keys(conn):
                    state = f"revoked {k['revoked_at']:%Y-%m-%d}" if k["revoked_at"] else "active"
                    print(f"{k['key_id']:>4}  {k['tenant_id']:<16} {k['role']:<9} {state:<18} {k['label']}")
            elif args.cmd == "revoke-key":
                print(f"key {args.key_id} revoked" if revoke_key(conn, args.key_id) else f"no active key {args.key_id}")
    except psycopg.errors.ForeignKeyViolation:
        raise SystemExit(f"no tenant {args.tenant}: uv run keys add-tenant {args.tenant}") from None
    except psycopg.errors.UniqueViolation:
        raise SystemExit(f"tenant {args.tenant_id} already exists") from None
    except psycopg.errors.CheckViolation:
        raise SystemExit("tenant ids: 1-32 chars of a-z 0-9 _ -") from None
