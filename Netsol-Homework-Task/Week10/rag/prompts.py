"""All prompts in one place (the 'P' of RAG)."""
from typing import Dict, List, Optional

from .models import RetrievedChunk

RAG_SYSTEM = """You are a careful assistant that answers questions using ONLY the context passages provided by the user message.
Rules:
- Use only the information in the context. If the answer is not there, reply exactly: "I couldn't find that in the uploaded documents."
- Cite the passages you used with their numbers in square brackets, for example [1] or [2][3].
- Be concise and accurate. Never invent facts, numbers or citations."""

RESEARCH_SYSTEM = """You are a research assistant answering questions about the user's papers using ONLY the context passages provided.
Structure the answer as: a direct answer (2-4 sentences), then "Evidence" (bullet points with citations like [1]), then "Caveats" (limits or things the papers do not say).
If the context does not contain the answer, reply exactly: "I couldn't find that in the uploaded documents." Never invent citations or numbers."""

CONDENSE_SYSTEM = """Given a chat history and a follow-up question, rewrite the follow-up as ONE standalone question that can be understood without the history.
Return only the question. If it is already standalone, return it unchanged."""

STYLES = {
    "beginner": "Explain it for a beginner. Use plain language, avoid jargon (define any term you must use) and add one simple analogy if it helps.",
    "technical": "Write for a technical reader (graduate level). Keep the correct terminology, mention models, datasets, metrics and key numbers.",
    "executive": "Write for a busy decision maker. Focus on the problem, the main result, why it matters and possible applications. No equations or implementation details.",
}
LENGTHS = {"short": "about 100 words", "medium": "about 250 words", "long": "about 500 words"}
FORMATS = {
    "paragraph": "Write it as flowing paragraphs.",
    "bullets": "Write it as concise bullet points.",
    "structured": "Use exactly these markdown headings: Problem, Method, Results, Limitations, Conclusion.",
}


def format_context(retrieved: List[RetrievedChunk]) -> str:
    blocks = []
    for i, r in enumerate(retrieved, start=1):
        where = r.chunk.source + (f", page {r.chunk.page}" if r.chunk.page else "")
        blocks.append(f"[{i}] (source: {where})\n{r.chunk.text}")
    return "\n\n".join(blocks)


def build_rag_messages(question: str, retrieved: List[RetrievedChunk], history: Optional[List[Dict]] = None,
                       mode: str = "qa") -> List[Dict]:
    system = RESEARCH_SYSTEM if mode == "research" else RAG_SYSTEM
    messages = [{"role": "system", "content": system}]
    messages += [{"role": m["role"], "content": m["content"]} for m in (history or [])[-6:]]   # last 3 turns
    messages.append({"role": "user", "content": f"Context:\n{format_context(retrieved)}\n\nQuestion: {question}"})
    return messages


def build_condense_messages(question: str, history: List[Dict]) -> List[Dict]:
    transcript = "\n".join(f"{m['role']}: {m['content']}" for m in history[-6:])
    return [{"role": "system", "content": CONDENSE_SYSTEM},
            {"role": "user", "content": f"Chat history:\n{transcript}\n\nFollow-up question: {question}"}]


# ---- dynamic prompt of the Research Assistant (same idea as Week 9) --------------------
def build_summary_messages(title: str, text: str, style: str = "technical", length: str = "medium",
                           fmt: str = "structured", focus: Optional[str] = None) -> List[Dict]:
    if style not in STYLES or length not in LENGTHS or fmt not in FORMATS:
        raise ValueError(f"style must be one of {list(STYLES)}, length one of {list(LENGTHS)}, format one of {list(FORMATS)}")
    focus_line = f"Pay special attention to: {focus.strip()}." if focus and focus.strip() else "Cover the paper as a whole."
    system = (
        "You are an expert research assistant that summarises scientific papers.\n\n"
        f"Audience and style: {STYLES[style]}\n"
        f"Length: {LENGTHS[length]}.\n"
        f"Format: {FORMATS[fmt]}\n"
        f"{focus_line}\n\n"
        "Rules:\n- Use ONLY information that is in the text you are given. If something is not stated, say so.\n"
        "- Do not invent numbers, datasets or citations.\n- Start directly with the summary (no preamble)."
    )
    return [{"role": "system", "content": system},
            {"role": "user", "content": f"Paper: {title}\n\nPaper text:\n{text}"}]


def build_notes_messages(title: str, part: int, total: int, text: str) -> List[Dict]:
    system = ("You are a research assistant. Read this part of a paper and write compact notes (at most 150 words, bullet points): "
              "the problem, the method, datasets/experiments, key numbers and conclusions that appear in THIS part. "
              "Do not add anything that is not in the text.")
    return [{"role": "system", "content": system},
            {"role": "user", "content": f'Part {part} of {total} of the paper "{title}":\n\n{text}'}]
