"""The complete RAG pipeline: ingest (load -> chunk -> embed -> store) and ask (retrieve -> prompt -> generate)."""
from typing import Dict, Iterator, List, Optional

from .chunker import chunk_documents
from .config import Settings, get_settings
from .embedder import get_embedder
from .llm import LLMClient
from .loader import load_bytes
from .models import Prepared, RAGAnswer, RetrievedChunk
from .prompts import build_condense_messages, build_rag_messages
from .vector_store import VectorStore

NOT_FOUND = "I couldn't find that in the uploaded documents."
EMPTY_INDEX = "No documents are indexed yet. Upload a PDF or text file first."


class RAGPipeline:
    def __init__(self, settings: Optional[Settings] = None, embedder=None, llm=None):
        self.settings = settings or get_settings()
        self.embedder = embedder or get_embedder(self.settings)
        self._llm = llm
        self.store = VectorStore.load(self.settings.index_dir, self.embedder.name)
        self.store.embedder_name = self.embedder.name

    @property
    def llm(self):
        if self._llm is None:                       # created lazily so that indexing works without an API key
            self._llm = LLMClient(self.settings)
        return self._llm

    # ------------------------------------------------------------------ ingest
    def ingest_bytes(self, data: bytes, filename: str) -> int:
        """Index one file. Uploading the same filename again replaces the old version. Returns the number of chunks."""
        docs = load_bytes(data, filename)
        chunks = chunk_documents(docs, self.settings.chunk_size, self.settings.chunk_overlap)
        if not chunks:
            raise ValueError(f"'{filename}' produced no text chunks.")
        vectors = self.embedder.embed_documents([c.text for c in chunks])
        self.store.delete_source(filename)
        self.store.add(chunks, vectors)
        self.store.save(self.settings.index_dir)
        return len(chunks)

    def delete_source(self, source: str) -> int:
        removed = self.store.delete_source(source)
        self.store.save(self.settings.index_dir)
        return removed

    def clear(self) -> None:
        self.store.clear()
        self.store.save(self.settings.index_dir)

    def sources(self) -> Dict[str, int]:
        return self.store.sources()

    # ---------------------------------------------------------------- retrieve
    def retrieve(self, question: str, top_k: Optional[int] = None) -> List[RetrievedChunk]:
        q = self.embedder.embed_query(question)
        return self.store.search(q, top_k or self.settings.top_k, self.settings.min_score)

    # ---------------------------------------------------------------- generate
    def prepare(self, question: str, history: Optional[List[Dict]] = None, top_k: Optional[int] = None,
                mode: str = "qa", rewrite: bool = True) -> Prepared:
        """Retrieval + prompt building (no answer yet). Lets the GUI stream the answer token by token."""
        if len(self.store) == 0:
            return Prepared(question, question, [], [], skip_llm_answer=EMPTY_INDEX)

        standalone = question
        if rewrite and history:                                   # turn "and its limits?" into a full question
            standalone = self.llm.chat(build_condense_messages(question, history), temperature=0.0, max_tokens=120).strip() or question

        retrieved = self.retrieve(standalone, top_k)
        if not retrieved:
            return Prepared(question, standalone, [], [], skip_llm_answer=NOT_FOUND)
        return Prepared(question, standalone, retrieved, build_rag_messages(standalone, retrieved, history, mode))

    def stream_answer(self, prepared: Prepared, temperature: Optional[float] = None) -> Iterator[str]:
        if prepared.skip_llm_answer:
            yield prepared.skip_llm_answer
            return
        yield from self.llm.stream(prepared.messages, temperature)

    def ask(self, question: str, history: Optional[List[Dict]] = None, top_k: Optional[int] = None,
            temperature: Optional[float] = None, mode: str = "qa", rewrite: bool = True) -> RAGAnswer:
        prepared = self.prepare(question, history, top_k, mode, rewrite)
        if prepared.skip_llm_answer:
            return RAGAnswer(prepared.skip_llm_answer, prepared.retrieved, prepared.standalone_question)
        answer = self.llm.chat(prepared.messages, temperature)
        return RAGAnswer(answer, prepared.retrieved, prepared.standalone_question)
