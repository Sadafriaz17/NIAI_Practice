"""Offline smoke test: no API key, no model download, no network.
Run from the project folder:   python tests/test_core.py
"""
import os
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
tmp = tempfile.mkdtemp()
os.environ.update({"EMBEDDER": "hash", "DATA_DIR": tmp, "LLM_PROVIDER": "ollama",
                   "CHUNK_SIZE": "400", "CHUNK_OVERLAP": "80", "TOP_K": "3"})

from rag.chunker import split_text                     # noqa: E402
from rag.config import get_settings                    # noqa: E402
from rag.pipeline import RAGPipeline, NOT_FOUND, EMPTY_INDEX   # noqa: E402
from rag.research import summarize                     # noqa: E402


class FakeLLM:
    """Pretends to be the language model: returns the context it was given."""
    def __init__(self):
        self.calls = []

    def chat(self, messages, temperature=None, max_tokens=None):
        self.calls.append(messages)
        return "FAKE ANSWER based on: " + messages[-1]["content"][:60].replace("\n", " ")

    def stream(self, messages, temperature=None):
        yield from self.chat(messages).split(" ")


def test_chunker():
    text = " ".join(f"This is sentence number {i} about topic {i % 7}." for i in range(300))
    chunks = split_text(text, 400, 80)
    assert len(chunks) > 5
    assert all(len(c) <= 400 for c in chunks), max(len(c) for c in chunks)
    assert any(chunks[i][-30:] in chunks[i + 1] for i in range(len(chunks) - 1)), "no overlap found"
    assert split_text("", 400, 80) == []
    assert all(len(c) <= 100 for c in split_text("x" * 1000, 100, 10))     # very long word
    print("chunker ok:", len(chunks), "chunks")


def test_pipeline():
    llm = FakeLLM()
    pipe = RAGPipeline(get_settings(), llm=llm)
    assert pipe.ask("anything?").answer == EMPTY_INDEX

    data = (Path(__file__).resolve().parent.parent / "sample_docs" / "rag_overview.txt").read_bytes()
    n = pipe.ingest_bytes(data, "rag_overview.txt")
    assert n >= 3 and pipe.sources() == {"rag_overview.txt": n}
    assert pipe.ingest_bytes(data, "rag_overview.txt") == n and len(pipe.store) == n      # re-upload replaces

    hits = pipe.retrieve("What does the vector store do with cosine similarity?")
    assert hits and "vector store" in hits[0].chunk.text.lower(), hits[0].chunk.text
    print("top hit:", round(hits[0].score, 3), hits[0].chunk.text[:70], "...")

    ans = pipe.ask("What chunk size works well?")
    assert ans.answer.startswith("FAKE ANSWER") and ans.sources
    assert "Context:" in llm.calls[-1][-1]["content"] and "[1]" in llm.calls[-1][-1]["content"]

    followup = pipe.ask("and the top-k?", history=[{"role": "user", "content": "What chunk size works well?"},
                                                     {"role": "assistant", "content": "500 to 1000 characters."}])
    assert followup.answer                      # the condense step + answer both went through the fake LLM

    pieces = list(pipe.stream_answer(pipe.prepare("What is a loader?", rewrite=False)))
    assert pieces
    print("pipeline ok")

    # persistence: a new pipeline object finds the saved index
    pipe2 = RAGPipeline(get_settings(), llm=llm)
    assert len(pipe2.store) == n
    assert pipe2.delete_source("rag_overview.txt") == n and len(pipe2.store) == 0
    print("persistence ok")


def test_research():
    llm = FakeLLM()
    short = "Background and method. " * 40
    out = summarize(llm, short, "t", "beginner", "short", "bullets", "limitations")
    system = llm.calls[-1][0]["content"]
    assert "plain language" in system and "about 100 words" in system and "bullet points" in system and "limitations" in system
    long_text = ("The study evaluates many models on many datasets. " * 30 + "\n\n") * 30
    llm.calls.clear()
    summarize(llm, long_text, "long paper", max_chars=3000)
    assert len(llm.calls) >= 3, len(llm.calls)       # notes for several parts + the final summary
    print("research ok:", out[:30], "| calls for long paper:", len(llm.calls))


if __name__ == "__main__":
    test_chunker()
    test_pipeline()
    test_research()
    print("\nALL TESTS PASSED")
