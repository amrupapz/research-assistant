"""Outreach status by hand (D4 M6). We don't read your inbox, so you tell us about replies.

uv run python -m research_assistant.outreach status
uv run python -m research_assistant.outreach replied L1 --note "said: call next week"
"""

import argparse

from research_assistant import state
from research_assistant.db import SEED_LEADS, connect, fetch_lead


def main() -> None:
    parser = argparse.ArgumentParser(
        prog="outreach",
        description="Who we emailed, who replied, who's due a follow-up.",
    )
    sub = parser.add_subparsers(dest="cmd", required=True)
    sub.add_parser("status", help="one line per lead")
    replied = sub.add_parser("replied", help="record that a lead replied (now)")
    replied.add_argument("lead_id")
    replied.add_argument("--note", default="")
    args = parser.parse_args()

    with connect(autocommit=True) as conn:
        if args.cmd == "replied":
            if fetch_lead(conn, args.lead_id) is None:
                parser.error(f"no lead {args.lead_id!r}")
            state.record_reply(conn, args.lead_id, args.note)
            print(f"recorded a reply from {args.lead_id}")
            return
        due = {f["lead_id"] for f in state.followups_due(conn, 3)}
        for lead in SEED_LEADS:
            o = state.outreach(conn, lead.lead_id)
            last = o["last_contacted_at"]
            contacted = (
                f"last emailed {last:%Y-%m-%d %H:%M}" if last else "never emailed"
            )
            replied_txt = (
                "REPLIED"
                if o["replied_since_last_contact"]
                else ("no reply" if last else "")
            )
            flag = "  <- follow-up due (3 days)" if lead.lead_id in due else ""
            print(
                f"{lead.lead_id}  {lead.name:<12} {lead.company:<8} {contacted:<28} {replied_txt:<8}{flag}"
            )


if __name__ == "__main__":
    main()
