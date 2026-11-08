#!/usr/bin/env python
"""Week 9 - Task 1: Research Assistant that summarises papers with a DYNAMIC prompt.

The prompt is not fixed: it is assembled at run time from the user's choices
(style, length, output format, focus) with a LangChain ChatPromptTemplate.
Long papers are handled with map-reduce (notes per part -> final summary).

Examples
    python research_assistant.py paper.pdf
    python research_assistant.py paper.pdf --style beginner --length short --format bullets
    python research_assistant.py paper.pdf --style technical --length long --focus "methodology and datasets"
    python research_assistant.py a.pdf b.pdf --compare
    python research_assistant.py paper.pdf --interactive
"""
import argparse
import math
import re
from pathlib import Path
from typing import List, Optional

from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate

from llm_factory import get_llm

# --------------------------------------------------------------------------
# 1. The building blocks of the dynamic prompt
# --------------------------------------------------------------------------
STYLES = {
    "beginner": "Explain it for a beginner. Use plain language, avoid jargon (define any term you must use) and add one simple analogy if it helps.",
    "technical": "Write for a technical reader (graduate level). Keep the correct terminology, mention models, datasets, metrics and key numbers.",
    "executive": "Write for a busy decision maker. Focus on the problem, the main result, why it matters and possible applications. No equations or implementation details.",
}
LENGTHS = {
    "short": "about 100 words",
    "medium": "about 250 words",
    "long": "about 500 words",
}
FORMATS = {
    "paragraph": "Write it as flowing paragraphs.",
    "bullets": "Write it as concise bullet points.",
    "structured": "Use exactly these markdown headings: Problem, Method, Results, Limitations, Conclusion.",
}

SUMMARY_SYSTEM = """You are an expert research assistant that summarises scientific papers.

Audience and style: {style_instruction}
Length: {length_instruction}.
Format: {format_instruction}
{focus_instruction}

Rules:
- Use ONLY information that is in the text you are given. If something is not in the text, do not guess; say it is not stated.
- Do not invent numbers, datasets or citations.
- Start directly with the summary (no preamble)."""

NOTES_SYSTEM = """You are a research assistant. Read this part of a paper and write compact notes (at most 150 words, bullet points):
the problem, the method, datasets/experiments, key numbers and conclusions that appear in THIS part. Do not add anything that is not in the text."""

COMPARE_SYSTEM = """You are an expert research assistant. Compare the following papers using their summaries.
{style_instruction}
Format: a short markdown table (aspects as rows: problem, method, data, main result, limitations) followed by a 3-5 sentence conclusion about which approach is better for what.
Use ONLY the information in the summaries."""


def build_summary_prompt(style: str = "technical", length: str = "medium",
                         fmt: str = "structured", focus: Optional[str] = None) -> ChatPromptTemplate:
    """Assemble the prompt dynamically from the user's options."""
    if style not in STYLES or length not in LENGTHS or fmt not in FORMATS:
        raise ValueError(f"style must be one of {list(STYLES)}, length one of {list(LENGTHS)}, format one of {list(FORMATS)}")
    focus_instruction = (f"Pay special attention to: {focus.strip()}." if focus and focus.strip()
                         else "Cover the paper as a whole.")
    prompt = ChatPromptTemplate.from_messages([
        ("system", SUMMARY_SYSTEM),
        ("human", "Paper: {title}\n\nPaper text:\n{paper_text}"),
    ])
    # .partial() fixes the dynamic parts; title and paper_text are filled at invoke time
    return prompt.partial(style_instruction=STYLES[style], length_instruction=LENGTHS[length],
                          format_instruction=FORMATS[fmt], focus_instruction=focus_instruction)


# --------------------------------------------------------------------------
# 2. Reading papers
# --------------------------------------------------------------------------
def read_paper(path: str) -> str:
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(path)
    if p.suffix.lower() == ".pdf":
        from pypdf import PdfReader
        pages = [(pg.extract_text() or "") for pg in PdfReader(str(p)).pages]
        text = "\n".join(pages)
    else:
        text = p.read_text(encoding="utf-8", errors="ignore")
    text = clean_text(text)
    if len(text) < 200:
        raise ValueError("Almost no text could be extracted (scanned PDF?).")
    return text


def clean_text(text: str) -> str:
    text = text.replace("\x00", " ")
    text = re.sub(r"-\n(?=[a-z])", "", text)        # re-join words hyphenated at line ends
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return strip_references(text.strip())


def strip_references(text: str) -> str:
    """Cut the bibliography (saves tokens). Only looks in the last 40% of the paper."""
    start = int(len(text) * 0.6)
    matches = list(re.finditer(r"\n\s*(references|bibliography)\s*\n", text[start:], flags=re.I))
    return text[: start + matches[0].start()] if matches else text


def split_parts(text: str, size: int, overlap: int = 400) -> List[str]:
    parts, i = [], 0
    while i < len(text):
        end = min(i + size, len(text))
        if end < len(text):                           # try to cut at a paragraph/sentence end
            cut = max(text.rfind("\n\n", i, end), text.rfind(". ", i, end))
            if cut > i + size // 2:
                end = cut + 1
        parts.append(text[i:end].strip())
        if end >= len(text):
            break
        i = max(end - overlap, i + 1)
    return parts


# --------------------------------------------------------------------------
# 3. Summarising (single pass or map-reduce)
# --------------------------------------------------------------------------
def summarize_paper(text: str, title: str, style="technical", length="medium", fmt="structured",
                    focus: Optional[str] = None, llm=None, max_chars: int = 12000, max_parts: int = 15) -> str:
    llm = llm or get_llm(temperature=0.2)
    parser = StrOutputParser()
    summary_chain = build_summary_prompt(style, length, fmt, focus) | llm | parser

    if len(text) <= max_chars:
        return summary_chain.invoke({"title": title, "paper_text": text})

    # MAP: compact notes for every part of the paper
    size = max(max_chars, math.ceil(len(text) / max_parts))
    parts = split_parts(text, size)
    notes_prompt = ChatPromptTemplate.from_messages([
        ("system", NOTES_SYSTEM),
        ("human", "Part {part} of {total} of the paper \"{title}\":\n\n{text}"),
    ])
    notes_chain = notes_prompt | llm | parser
    notes = []
    for i, part in enumerate(parts, start=1):
        print(f"   notes for part {i}/{len(parts)} ...")
        notes.append(f"### Notes, part {i}\n" + notes_chain.invoke(
            {"part": i, "total": len(parts), "title": title, "text": part}))

    # REDUCE: final summary written from the notes with the dynamic prompt
    return summary_chain.invoke({"title": title, "paper_text": "\n\n".join(notes)})


def compare_summaries(summaries: dict, style="technical", llm=None) -> str:
    llm = llm or get_llm(temperature=0.2)
    prompt = ChatPromptTemplate.from_messages([
        ("system", COMPARE_SYSTEM),
        ("human", "{summaries}"),
    ]).partial(style_instruction=STYLES[style])
    body = "\n\n".join(f"## {name}\n{s}" for name, s in summaries.items())
    return (prompt | llm | StrOutputParser()).invoke({"summaries": body})


# --------------------------------------------------------------------------
# 4. Command line interface
# --------------------------------------------------------------------------
def ask_choice(question: str, options: list, default: str) -> str:
    answer = input(f"{question} {options} [{default}]: ").strip().lower()
    return answer if answer in options else default


def main():
    ap = argparse.ArgumentParser(description="Summarise research papers with a dynamic prompt.")
    ap.add_argument("papers", nargs="+", help="PDF / TXT / MD files")
    ap.add_argument("--style", choices=STYLES, default="technical")
    ap.add_argument("--length", choices=LENGTHS, default="medium")
    ap.add_argument("--format", dest="fmt", choices=FORMATS, default="structured")
    ap.add_argument("--focus", default=None, help='e.g. "methodology and datasets"')
    ap.add_argument("--compare", action="store_true", help="also compare the papers (needs 2+ files)")
    ap.add_argument("--interactive", action="store_true", help="ask for the options one by one")
    ap.add_argument("--out", default="summaries", help="folder for the .md results")
    args = ap.parse_args()

    if args.interactive:
        args.style = ask_choice("Style", list(STYLES), args.style)
        args.length = ask_choice("Length", list(LENGTHS), args.length)
        args.fmt = ask_choice("Format", list(FORMATS), args.fmt)
        args.focus = input("Focus (optional, e.g. 'limitations'): ").strip() or None

    llm = get_llm(temperature=0.2)
    out_dir = Path(args.out)
    out_dir.mkdir(exist_ok=True)
    summaries = {}

    for path in args.papers:
        title = Path(path).stem
        print(f"\n=== {title} ({args.style}, {args.length}, {args.fmt}) ===")
        summary = summarize_paper(read_paper(path), title, args.style, args.length, args.fmt, args.focus, llm=llm)
        summaries[title] = summary
        print(summary)
        (out_dir / f"{title}_{args.style}_{args.length}.md").write_text(f"# {title}\n\n{summary}\n", encoding="utf-8")

    if args.compare and len(summaries) > 1:
        print("\n=== COMPARISON ===")
        comparison = compare_summaries(summaries, args.style, llm=llm)
        print(comparison)
        (out_dir / "comparison.md").write_text(comparison + "\n", encoding="utf-8")
    print(f"\nSaved in ./{out_dir}/")


if __name__ == "__main__":
    main()
