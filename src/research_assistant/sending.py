"""Send an approved draft exactly once (D4 M5, made concurrency-safe in M7), even if we crash halfway.

The problem: sending is an EXTERNAL effect. A database transaction can't undo an email, so the
email and the record of it can't commit together. The fix is a ledger + an idempotency key + a lock:

    1. CLAIM   write "about to send D7" (sends row, status=pending) and COMMIT it.
    2. LOCK    in a new transaction, lock that row (FOR UPDATE NOWAIT) and re-read it.
    3. DECIDE  under the lock: sent -> nothing to do; pending -> was it delivered?
    4. SEND    deliver the email, stamped with the idempotency key "draft-7" (still holding the lock).
    5. RECORD  mark the row sent; COMMIT releases the lock.

Crash anywhere -> the lock is released with the dead connection, the row stays pending, and the
next attempt decides under the lock: delivered -> just record it; not delivered -> send now.
Two live processes at once -> only one gets the lock; the other is told to try again (M7: without
the lock both saw "pending, not delivered" and both sent).

A channel that can't answer "was this key delivered?" (most real email) must REFUSE a pending row
it didn't claim itself and ask a human: sending twice is worse than not sending.
"""

import os
from email import message_from_bytes
from email.message import EmailMessage
from pathlib import Path

import psycopg
from psycopg.errors import LockNotAvailable

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
    tmp = outbox_dir / f".{draft.draft_id}.eml.tmp"
    tmp.write_bytes(bytes(msg))
    try:
        os.link(tmp, final)  # atomic create-if-absent: complete file or nothing, and never overwrites
    except FileExistsError:
        raise SendError(f"{final} already exists and is not this draft's send; refusing to overwrite it") from None
    finally:
        tmp.unlink(missing_ok=True)
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
        claimed = state.claim_send(conn, draft.draft_id, approval[0], key, channel) is not None

    try:
        # 2-5. LOCK, DECIDE, SEND, RECORD -- one transaction. Every path goes through here, so
        # whoever holds the lock is the only one deciding.
        with conn.transaction():
            send = state.lock_send(conn, draft.draft_id)
            if send["status"] == "sent":
                return {"draft_id": draft.draft_id, "to": draft.to, "status": "already sent", "sent_at": send["sent_at"]}
            if channel == "outbox":
                if was_delivered(outbox_dir, draft, key):  # an earlier attempt died between SEND and RECORD
                    state.mark_sent(conn, draft.draft_id, str(outbox_dir / f"{draft.draft_id}.eml"))
                    return {"draft_id": draft.draft_id, "to": draft.to, "status": "already sent (recovered)"}
            elif not claimed:  # pending from an earlier attempt, and this channel can't tell if it went out
                raise SendError(f"draft {draft.draft_id} has an unfinished send; check manually before retrying")
            where = deliver(outbox_dir, draft, key)  # the external effect, under the lock
            state.mark_sent(conn, draft.draft_id, where)
            return {"draft_id": draft.draft_id, "to": draft.to, "status": "sent (written to outbox)", "path": where}
    except LockNotAvailable:
        raise SendError(f"another process is sending draft {draft.draft_id} right now; try again shortly") from None
