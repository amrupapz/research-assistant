"""Send an approved draft exactly once (D4 M5), even if we crash halfway.

The problem: sending is an EXTERNAL effect. A database transaction can't undo an email, so the
email and the record of it can't commit together. The fix is a ledger + an idempotency key:

    1. CLAIM   write "about to send D7" (sends row, status=pending) and COMMIT it.
    2. SEND    deliver the email, stamped with the idempotency key "draft-7".
    3. RECORD  mark the row sent and COMMIT.

A crash can happen between any two steps; a retry reads the ledger and knows what to do:

    crash before 1 committed  -> no row        -> nothing was sent: claim and send now
    crash between 1 and 2     -> row pending   -> ask the channel: not delivered -> send now
    crash between 2 and 3     -> row pending   -> ask the channel: delivered -> just record it
    crash after 3             -> row sent      -> do nothing: "already sent"

A channel that can't answer "was this key delivered?" (most real email) must REFUSE a pending
retry and ask a human: sending twice is worse than not sending.
"""

import os
from email import message_from_bytes
from email.message import EmailMessage
from pathlib import Path

import psycopg

from research_assistant import state
from research_assistant.schemas import Draft

FROM = "sales-agent@example.com"
KEY_HEADER = "X-Idempotency-Key"


class SendError(Exception):
    """Not sent, and the reason is safe to show the model (not approved, stuck, unknown draft)."""


def idempotency_key(draft_id: str) -> str:
    """Same draft -> same key, on every attempt and every retry. (A uuid4 per call would be useless:
    the retry would get a NEW key and look like a different email.)"""
    return f"draft-{draft_id.removeprefix('D')}"


# --- the outbox channel: "sending" = writing outbox/<draft_id>.eml -------------------------


def deliver(outbox_dir: Path, draft: Draft, key: str) -> str:
    """Write the email atomically. Returns where it went (our stand-in for a provider message id)."""
    msg = EmailMessage()
    msg["From"] = FROM
    msg["To"] = draft.to
    msg["Subject"] = draft.subject
    msg[KEY_HEADER] = key
    msg.set_content(draft.body)

    outbox_dir.mkdir(parents=True, exist_ok=True)
    final = outbox_dir / f"{draft.draft_id}.eml"
    if final.exists():
        raise SendError(f"{final} already exists and is not this draft's send; refusing to overwrite it")
    tmp = outbox_dir / f".{draft.draft_id}.eml.tmp"
    tmp.write_bytes(bytes(msg))
    os.replace(tmp, final)  # atomic rename: the file is either fully there or not there at all
    return str(final)


def was_delivered(outbox_dir: Path, draft: Draft, key: str) -> bool:
    """Did the email with THIS key go out? For the outbox we can just look."""
    path = outbox_dir / f"{draft.draft_id}.eml"
    if not path.exists():
        return False
    return message_from_bytes(path.read_bytes())[KEY_HEADER] == key


# --- send_once ------------------------------------------------------------------------------


def send_once(conn: psycopg.Connection, draft: Draft, outbox_dir: Path, channel: str = "outbox") -> dict:
    """Send `draft` at most once. Returns what happened; raises SendError if it must not send."""
    key = idempotency_key(draft.draft_id)

    # 1. CLAIM -- its own short transaction, committed BEFORE anything leaves the building.
    with conn.transaction():
        approval = state.latest_approval(conn, draft.draft_id)
        if approval is None or approval[1] != "approved":
            raise SendError(f"draft {draft.draft_id} has no approval from the user")  # the FK can't check this
        claimed = state.claim_send(conn, draft.draft_id, approval[0], key, channel)

    if claimed is None:  # someone (an earlier attempt) already claimed this draft
        existing = state.get_send(conn, draft.draft_id)
        if existing["status"] == "sent":
            return {"draft_id": draft.draft_id, "to": draft.to, "status": "already sent", "sent_at": existing["sent_at"]}
        # status is pending: an earlier attempt died after claiming. Did its email go out?
        if channel != "outbox":
            raise SendError(f"draft {draft.draft_id} has an unfinished send; check manually before retrying")
        if was_delivered(outbox_dir, draft, key):  # died between SEND and RECORD: just record it
            with conn.transaction():
                state.mark_sent(conn, draft.draft_id, str(outbox_dir / f"{draft.draft_id}.eml"))
            return {"draft_id": draft.draft_id, "to": draft.to, "status": "already sent (recovered)"}
        # died between CLAIM and SEND: nothing went out, so it's safe to send now (fall through)

    # 2. SEND -- the external effect. No transaction open: we can't roll this back anyway.
    where = deliver(outbox_dir, draft, key)

    # 3. RECORD -- its own short transaction.
    with conn.transaction():
        state.mark_sent(conn, draft.draft_id, where)
    return {"draft_id": draft.draft_id, "to": draft.to, "status": "sent (written to outbox)", "path": where}
