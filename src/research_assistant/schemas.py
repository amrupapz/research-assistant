"""Pydantic models: the shapes we force the model's output into."""

from datetime import date
from typing import Literal

from pydantic import BaseModel, Field


class Citation(BaseModel):
    chunk_id: str = Field(description="Exact id of a chunk from <untrusted_data>, e.g. 'figma-about#000'.")
    source_url: str = Field(description="That chunk's source_url, copied exactly.")


class ResearchAnswer(BaseModel):
    answer: str = Field(
        description="Concise answer for a salesperson. Put [chunk_id] after every factual claim. "
        "If the sources don't answer the question, say so."
    )
    confidence: Literal["low", "medium", "high"] = Field(
        description="How well the provided sources support the answer, including how current they are."
    )
    assumptions: list[str] = Field(
        description="Things you inferred rather than read in the sources. Empty list if none."
    )
    needs_sources: bool = Field(
        description="True if the provided sources are missing, conflicting, or outdated for this question."
    )
    citations: list[Citation] = Field(
        description="Every chunk cited in the answer. Empty only if the sources don't answer the question."
    )


class Page(BaseModel):
    """One saved source page from data/."""

    source_url: str
    fetched_at: date
    title: str
    text: str
    synthetic: bool = False


class Chunk(BaseModel):
    """A searchable piece of a Page. Never exists without its source."""

    chunk_id: str  # stable: "<file stem>#<index>", e.g. "figma-about#002"
    source_url: str
    fetched_at: date
    title: str
    text: str


class Lead(BaseModel):
    """One sales contact (fictional; emails are @example.com)."""

    lead_id: str
    name: str
    title: str
    company: str
    email: str


class Draft(BaseModel):
    """An email the agent drafted. Drafting never sends; send_email (M3) needs approval."""

    draft_id: str
    lead_id: str
    to: str
    subject: str
    body: str


Met = Literal["yes", "no", "unknown"]
Fit = Literal["strong", "possible", "poor"]


class CriterionResult(BaseModel):
    """One ICP criterion judged for one company. yes/no must cite chunks (checked in code)."""

    id: str = Field(description="The criterion id exactly as given, e.g. 'software_org'.")
    met: Met = Field(description="yes / no only if a chunk about this company clearly answers it; else unknown.")
    reason: str = Field(description="One short sentence: what the evidence shows, or that it is missing.")
    chunk_ids: list[str] = Field(description="Exact ids of the chunks that support yes/no. Empty for unknown.")


class IcpJudgment(BaseModel):
    """The model's structured output for qualify_lead: one result per criterion. It never picks the fit."""

    results: list[CriterionResult]


class Qualification(BaseModel):
    """A lead scored against the ICP. `fit` is computed in code (qualify.compute_fit), never by the model."""

    lead_id: str
    company: str
    fit: Fit
    criteria: list[CriterionResult]
    icp_version: str


Verdict = Literal["supported", "partial", "unsupported", "not_a_claim"]


class ClaimVerdict(BaseModel):
    """The judge's structured output (D6 M3): does the cited text SUPPORT the claim? (Not: is it true.)"""

    verdict: Verdict = Field(description="supported / partial / unsupported / not_a_claim.")
    reason: str = Field(description="One short sentence: what the chunks do or don't say.")


class EmailDraft(BaseModel):
    """The model's structured output for the workflow's draft node. Still goes through draft_email's
    validation (no links, no addresses, no invisible characters) before it's stored."""

    subject: str = Field(description="One-line subject, under 100 characters.")
    body: str = Field(description="Plain-text email body, under 150 words. No links, no email addresses.")
