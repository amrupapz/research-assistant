"""Pydantic models: the shapes we force the model's output into."""

from datetime import date
from typing import Literal

from pydantic import BaseModel, Field


class ResearchAnswer(BaseModel):
    answer: str = Field(description="Concise answer for a salesperson: a few sentences or a short list.")
    confidence: Literal["low", "medium", "high"] = Field(
        description="How sure you are the answer is correct AND current."
    )
    assumptions: list[str] = Field(
        description="Things you assumed or inferred rather than knew. Empty list if none."
    )
    needs_sources: bool = Field(
        description="True if the answer depends on recent or changing facts that must be verified."
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
