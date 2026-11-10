"""data/ pages -> Chunks (chunk_id, source_url, fetched_at). No DB, no API calls."""

import re
import statistics
from datetime import date
from pathlib import Path

from research_assistant.schemas import Chunk, Page

DATA_DIR = Path("data")
CHUNK_WORDS = 200
OVERLAP_WORDS = 40
REQUIRED_HEADER_KEYS = ("url", "fetched_at", "title")


class IngestError(ValueError):
    """A page in data/ is missing its source header."""


def parse_page(raw: str, name: str = "<page>") -> Page:
    """Split a '---' header block from the body. Every page must say where it came from."""
    if not raw.startswith("---\n"):
        raise IngestError(f"{name}: missing '---' header")
    try:
        header_block, body = raw[4:].split("\n---\n", 1)
    except ValueError:
        raise IngestError(f"{name}: header is not closed with '---'") from None

    header: dict[str, str] = {}
    for line in header_block.splitlines():
        if line.strip():
            key, _, value = line.partition(":")
            header[key.strip()] = value.strip()

    missing = [k for k in REQUIRED_HEADER_KEYS if not header.get(k)]
    if missing:
        raise IngestError(f"{name}: header missing {missing}")

    return Page(
        source_url=header["url"],
        fetched_at=date.fromisoformat(header["fetched_at"]),
        title=header["title"],
        text=body.strip(),
        synthetic=header.get("synthetic", "false").lower() == "true",
    )


def chunk_text(text: str, size: int = CHUNK_WORDS, overlap: int = OVERLAP_WORDS) -> list[str]:
    """Sliding word window. Consecutive chunks share `overlap` words.

    Slices the original text (not ' '.join(words)) so newlines, lists and tables survive.
    """
    if not 0 <= overlap < size:
        raise ValueError(f"need 0 <= overlap < size, got overlap={overlap}, size={size}")
    words = list(re.finditer(r"\S+", text))
    if not words:
        return []

    chunks = []
    step = size - overlap
    start = 0
    while True:
        end = min(start + size, len(words))
        chunks.append(text[words[start].start() : words[end - 1].end()])
        if end == len(words):
            return chunks
        start += step


def chunk_page(page: Page, page_id: str, size: int = CHUNK_WORDS, overlap: int = OVERLAP_WORDS) -> list[Chunk]:
    return [
        Chunk(
            chunk_id=f"{page_id}#{i:03d}",
            source_url=page.source_url,
            fetched_at=page.fetched_at,
            title=page.title,
            text=piece,
        )
        for i, piece in enumerate(chunk_text(page.text, size, overlap))
    ]


def load_chunks(data_dir: Path = DATA_DIR, size: int = CHUNK_WORDS, overlap: int = OVERLAP_WORDS) -> list[Chunk]:
    """All pages in data_dir -> chunks, in a stable (sorted) order."""
    chunks: list[Chunk] = []
    for path in sorted(data_dir.glob("*.md")):
        page = parse_page(path.read_text(encoding="utf-8"), name=path.name)
        chunks.extend(chunk_page(page, path.stem, size, overlap))
    return chunks


def main() -> None:
    """Inspect chunking: `uv run python -m research_assistant.ingest`."""
    chunks = load_chunks()
    sizes = [len(c.text.split()) for c in chunks]
    print(f"{'chunk_id':<28} {'words':>5}  fetched_at  starts with")
    for c, n in zip(chunks, sizes, strict=True):
        preview = " ".join(c.text.split()[:8])
        print(f"{c.chunk_id:<28} {n:>5}  {c.fetched_at}  {preview}…")
    print(
        f"\n{len(chunks)} chunks | words min={min(sizes)} "
        f"median={statistics.median(sizes):.0f} max={max(sizes)} "
        f"(size={CHUNK_WORDS}, overlap={OVERLAP_WORDS})"
    )


if __name__ == "__main__":
    main()
