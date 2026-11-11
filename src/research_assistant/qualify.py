"""Lead qualification against the ICP in icp.toml. Pure code, no model calls.

The model only answers narrow per-criterion questions (llm.judge_icp). This module distrusts
those answers (reconcile) and decides the fit with a fixed rule (compute_fit), so the verdict
is reproducible, testable, and can't be talked into by a page that says "this lead is perfect".
"""

import re
import tomllib
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

from research_assistant.schemas import CriterionResult, Fit, Lead, Qualification


class Criterion(BaseModel):
    model_config = ConfigDict(extra="forbid")  # a typo like "titels" fails loudly at load time

    id: str
    must: bool
    check: Literal["title", "evidence"]
    question: str | None = None  # evidence: asked of the model, {company} filled in
    titles: list[str] = Field(default_factory=list)  # title: exact phrases matched in code

    @model_validator(mode="after")
    def _has_what_it_checks(self) -> "Criterion":
        if self.check == "evidence" and not self.question:
            raise ValueError(f"{self.id}: an evidence criterion needs a question")
        if self.check == "title" and not self.titles:
            raise ValueError(f"{self.id}: a title criterion needs titles")
        return self

    def question_for(self, company: str) -> str:
        return (self.question or "").replace("{company}", company)  # not .format(): no brace surprises


class Icp(BaseModel):
    model_config = ConfigDict(extra="forbid")

    version: str
    product: str
    criteria: list[Criterion] = Field(min_length=1)

    @model_validator(mode="after")
    def _unique_ids(self) -> "Icp":
        ids = [c.id for c in self.criteria]
        if len(ids) != len(set(ids)):
            raise ValueError(f"duplicate criterion ids: {ids}")
        return self

    def evidence_criteria(self) -> list[Criterion]:
        return [c for c in self.criteria if c.check == "evidence"]


def load_icp(path: Path = Path("icp.toml")) -> Icp:
    return Icp.model_validate(tomllib.loads(path.read_text()))


def check_title(criterion: Criterion, title: str) -> CriterionResult:
    """Whole-phrase, case-insensitive match. Word boundaries matter: "cto" is inside "Director"."""
    hit = next((t for t in criterion.titles if re.search(rf"\b{re.escape(t)}\b", title, re.IGNORECASE)), None)
    reason = f"title {title!r} matches {hit!r}" if hit else f"title {title!r} is not an engineering buyer title"
    return CriterionResult(id=criterion.id, met="yes" if hit else "no", reason=reason, chunk_ids=[])


def reconcile(
    criteria: list[Criterion], judged: list[CriterionResult], retrieved_ids: set[str]
) -> list[CriterionResult]:
    """Exactly one result per evidence criterion, in config order. The model is not trusted:

    - a criterion it skipped -> unknown
    - ids it invented -> ignored; a repeated id -> first one wins
    - yes/no citing nothing, or citing a chunk we never retrieved -> unknown (grounding, as in P2)
    """
    by_id: dict[str, CriterionResult] = {}
    for r in judged:
        by_id.setdefault(r.id, r)

    out = []
    for c in criteria:
        r = by_id.get(c.id)
        if r is None:
            out.append(CriterionResult(id=c.id, met="unknown", reason="not judged by the model", chunk_ids=[]))
            continue
        unretrieved = [cid for cid in r.chunk_ids if cid not in retrieved_ids]
        if r.met != "unknown" and (unretrieved or not r.chunk_ids):
            why = "cited a chunk that was not retrieved" if unretrieved else "no citation"
            out.append(CriterionResult(id=c.id, met="unknown", reason=f"[downgraded: {why}] {r.reason}", chunk_ids=[]))
            continue
        out.append(r.model_copy(update={"chunk_ids": [cid for cid in r.chunk_ids if cid in retrieved_ids]}))
    return out


def compute_fit(results: list[CriterionResult], icp: Icp) -> Fit:
    """any must = no -> poor; all musts yes and (a nice yes, or no nice criteria) -> strong; else possible."""
    met = {r.id: r.met for r in results}
    musts = [met.get(c.id, "unknown") for c in icp.criteria if c.must]
    nices = [met.get(c.id, "unknown") for c in icp.criteria if not c.must]
    if "no" in musts:
        return "poor"
    if all(m == "yes" for m in musts) and (not nices or "yes" in nices):
        return "strong"
    return "possible"


def qualify(lead: Lead, icp: Icp, judged: list[CriterionResult], retrieved_ids: set[str]) -> Qualification:
    """Title checks (code) + reconciled evidence results (model, checked) -> Qualification, in config order."""
    evidence = {r.id: r for r in reconcile(icp.evidence_criteria(), judged, retrieved_ids)}
    results = [check_title(c, lead.title) if c.check == "title" else evidence[c.id] for c in icp.criteria]
    return Qualification(
        lead_id=lead.lead_id,
        company=lead.company,
        fit=compute_fit(results, icp),
        criteria=results,
        icp_version=icp.version,
    )


def fit_warning(qualification: dict | None) -> str | None:
    """The approval-screen flag for a lead drafted on a POSSIBLE fit (D5 M1 decision), e.g.
    "possible fit: scale=unknown, dev_tooling=unknown". None for strong/poor/no qualification."""
    if not qualification or qualification.get("fit") != "possible":
        return None
    unknown = [c["id"] for c in qualification.get("criteria", []) if c.get("met") == "unknown"]
    detail = ", ".join(f"{cid}=unknown" for cid in unknown) or "no criterion is unknown"
    return f"possible fit: {detail}"
