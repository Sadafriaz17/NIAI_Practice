from dataclasses import dataclass, field, asdict
from typing import List, Optional


@dataclass
class Document:
    """Raw text coming out of a file (one PDF page, or a whole text file)."""
    text: str
    source: str
    page: Optional[int] = None


@dataclass
class Chunk:
    id: str
    text: str
    source: str
    page: Optional[int] = None

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict) -> "Chunk":
        return cls(id=d["id"], text=d["text"], source=d["source"], page=d.get("page"))


@dataclass
class RetrievedChunk:
    chunk: Chunk
    score: float


@dataclass
class Prepared:
    """Everything needed to generate an answer (retrieval already done)."""
    question: str
    standalone_question: str
    retrieved: List[RetrievedChunk]
    messages: List[dict]
    skip_llm_answer: Optional[str] = None   # set when no LLM call is needed (empty index ...)


@dataclass
class RAGAnswer:
    answer: str
    sources: List[RetrievedChunk] = field(default_factory=list)
    standalone_question: str = ""
