"""Leads table against the local test Postgres (rolled back; skipped if Postgres isn't up)."""

from research_assistant.db import SEED_LEADS, fetch_lead, seed_leads


def test_seed_and_fetch(db):
    seed_leads(db)
    seed_leads(db)  # idempotent
    assert db.execute("SELECT count(*) FROM leads").fetchone()[0] == len(SEED_LEADS)
    assert fetch_lead(db, "L1") == SEED_LEADS[0]
    assert fetch_lead(db, "L999") is None


def test_seed_updates_changed_lead(db):
    seed_leads(db)
    seed_leads(db, [SEED_LEADS[0].model_copy(update={"title": "SVP Engineering"})])
    assert fetch_lead(db, "L1").title == "SVP Engineering"
