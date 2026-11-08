"""Research Assistant: summarise a paper with a dynamic prompt (map-reduce for long papers)."""
import math
import re
from typing import Optional

from .chunker import split_text
from .prompts import build_notes_messages, build_summary_messages


def clean_paper_text(text: str) -> str:
    text = text.replace("\x00", " ")
    text = re.sub(r"-\n(?=[a-z])", "", text)
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text).strip()
    start = int(len(text) * 0.6)                              # drop the bibliography (only searched in the last 40%)
    m = list(re.finditer(r"\n\s*(references|bibliography)\s*\n", text[start:], flags=re.I))
    return text[: start + m[0].start()] if m else text


def summarize(llm, text: str, title: str = "paper", style: str = "technical", length: str = "medium",
              fmt: str = "structured", focus: Optional[str] = None, max_chars: int = 12000, max_parts: int = 15) -> str:
    text = clean_paper_text(text)
    if len(text) < 200:
        raise ValueError("The text is too short to summarise.")
    if len(text) <= max_chars:
        return llm.chat(build_summary_messages(title, text, style, length, fmt, focus))

    size = max(max_chars, math.ceil(len(text) / max_parts))   # MAP: notes for every part
    parts = split_text(text, chunk_size=size, overlap=400)
    notes = []
    for i, part in enumerate(parts, start=1):
        notes.append(f"### Notes, part {i}\n" + llm.chat(build_notes_messages(title, i, len(parts), part), max_tokens=400))
    return llm.chat(build_summary_messages(title, "\n\n".join(notes), style, length, fmt, focus))   # REDUCE
