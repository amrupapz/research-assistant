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
