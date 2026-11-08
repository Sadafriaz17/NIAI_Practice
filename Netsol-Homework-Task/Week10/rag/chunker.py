"""Step 2 - CHUNK: split long text into overlapping pieces that fit an embedding model."""
import re
from typing import List

from .models import Chunk, Document


def _hard_split(unit: str, size: int) -> List[str]:
    """Split a unit that is longer than `size` at whitespace."""
    pieces, current = [], ""
    for word in unit.split(" "):
        while len(word) > size:                       # a single huge "word" (e.g. a URL / table row)
            if current:
                pieces.append(current)
                current = ""
            pieces.append(word[:size])
            word = word[size:]
        if len(current) + len(word) + (1 if current else 0) > size:
            pieces.append(current)
            current = word
        else:
            current = f"{current} {word}" if current else word
    if current:
        pieces.append(current)
    return pieces


def _to_units(text: str, size: int) -> List[str]:
    """paragraphs -> sentences -> (if still too long) words."""
    units = []
    for paragraph in re.split(r"\n\s*\n", text):
        paragraph = " ".join(paragraph.split())
        if not paragraph:
            continue
        for sentence in re.split(r"(?<=[.!?])\s+", paragraph):
            if len(sentence) <= size:
                units.append(sentence)
            else:
                units.extend(_hard_split(sentence, size))
    return units


def split_text(text: str, chunk_size: int = 800, overlap: int = 150) -> List[str]:
    """Greedy packing of sentences into chunks of at most `chunk_size` characters.
    The end of every chunk is repeated (up to `overlap` characters) at the start of the next one,
    so that an answer that sits on a boundary is not lost."""
    if overlap >= chunk_size:
        raise ValueError("overlap must be smaller than chunk_size")
    chunks, current = [], []
    for unit in _to_units(text, chunk_size):
        if current and len(" ".join(current)) + 1 + len(unit) > chunk_size:
            chunks.append(" ".join(current))
            keep, total = [], 0                          # carry the tail of the chunk over
            for prev in reversed(current):
                if total + len(prev) + 1 > overlap:
                    break
                keep.insert(0, prev)
                total += len(prev) + 1
            while keep and len(" ".join(keep)) + 1 + len(unit) > chunk_size:
                keep.pop(0)
            current = keep
        current.append(unit)
    if current:
        chunks.append(" ".join(current))
    return [c.strip() for c in chunks if c.strip()]


def chunk_documents(docs: List[Document], chunk_size: int = 800, overlap: int = 150) -> List[Chunk]:
    chunks = []
    for doc in docs:
        for i, piece in enumerate(split_text(doc.text, chunk_size, overlap)):
            page_tag = f"p{doc.page}" if doc.page else "p0"
            chunks.append(Chunk(id=f"{doc.source}::{page_tag}::c{i}", text=piece, source=doc.source, page=doc.page))
    return chunks
