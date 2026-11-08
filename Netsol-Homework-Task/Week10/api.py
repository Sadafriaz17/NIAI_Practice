"""FastAPI service: RAG + Research Assistant endpoints.

    uvicorn api:app --reload --port 8000
    Interactive docs (Swagger):  http://localhost:8000/docs

Optional protection: set API_KEY in .env and send the header  X-API-Key: <value>.
"""
import os
import threading
from typing import List, Literal, Optional

from fastapi import Depends, FastAPI, File, Form, Header, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from rag.config import get_settings
from rag.llm import LLMClient, LLMConfigError
from rag.loader import SUPPORTED, load_bytes
from rag.pipeline import RAGPipeline
from rag.research import summarize

# --------------------------------------------------------------------------- app
def verify_key(x_api_key: Optional[str] = Header(default=None)) -> None:
    expected = os.getenv("API_KEY")
    if expected and x_api_key != expected:
        raise HTTPException(status_code=401, detail="Missing or invalid X-API-Key header.")


app = FastAPI(
    title="RAG + Research Assistant API",
    version="1.0.0",
    description="Upload documents, ask questions with cited answers (RAG) and generate dynamic-prompt paper summaries.",
    dependencies=[Depends(verify_key)],
)
app.add_middleware(CORSMiddleware, allow_origins=os.getenv("CORS_ORIGINS", "*").split(","),
                   allow_methods=["*"], allow_headers=["*"])

_lock = threading.Lock()
_pipeline: Optional[RAGPipeline] = None
_llm: Optional[LLMClient] = None


def get_pipeline() -> RAGPipeline:
    global _pipeline
    with _lock:
        if _pipeline is None:
            _pipeline = RAGPipeline()
        return _pipeline


def get_llm() -> LLMClient:
    global _llm
    with _lock:
        if _llm is None:
            _llm = LLMClient(get_settings())
        return _llm


def guarded(fn, *args, **kwargs):
    """Run fn and translate library errors into proper HTTP errors."""
    try:
        return fn(*args, **kwargs)
    except HTTPException:
        raise
    except LLMConfigError as e:
        raise HTTPException(status_code=503, detail=str(e))
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:                                           # e.g. network / rate-limit error from the LLM provider
        if e.__class__.__module__.startswith("openai"):
            raise HTTPException(status_code=502, detail=f"LLM provider error: {e}")
        raise


# --------------------------------------------------------------------------- schemas
class ChatTurn(BaseModel):
    role: Literal["user", "assistant"]
    content: str


class QueryRequest(BaseModel):
    question: str = Field(..., min_length=1, examples=["What chunk size works well?"])
    top_k: int = Field(4, ge=1, le=20)
    history: List[ChatTurn] = Field(default_factory=list, description="Previous turns (for follow-up questions)")
    rewrite_question: bool = True


class SourceOut(BaseModel):
    source: str
    page: Optional[int] = None
    score: float
    text: str


class QueryResponse(BaseModel):
    answer: str
    standalone_question: str
    sources: List[SourceOut]


class IngestResult(BaseModel):
    filename: str
    chunks: int = 0
    status: Literal["indexed", "error"]
    error: Optional[str] = None


class SummaryResponse(BaseModel):
    title: str
    style: str
    length: str
    format: str
    summary: str


def to_response(ans) -> QueryResponse:
    return QueryResponse(
        answer=ans.answer, standalone_question=ans.standalone_question,
        sources=[SourceOut(source=r.chunk.source, page=r.chunk.page, score=round(r.score, 4), text=r.chunk.text)
                 for r in ans.sources])


# --------------------------------------------------------------------------- endpoints
@app.get("/health", tags=["system"])
def health():
    s = get_settings()
    pipe = get_pipeline()
    return {"status": "ok", "documents": len(pipe.sources()), "chunks": len(pipe.store),
            "llm_provider": s.llm_provider, "llm_model": s.llm_model, "embedder": pipe.embedder.name,
            "supported_files": SUPPORTED}


@app.post("/rag/ingest", response_model=List[IngestResult], tags=["rag"])
def ingest(files: List[UploadFile] = File(..., description="PDF / TXT / MD files")):
    """Index one or more files. Uploading a file with the same name replaces the old version."""
    pipe = get_pipeline()
    results = []
    for f in files:
        try:
            n = pipe.ingest_bytes(f.file.read(), f.filename)
            results.append(IngestResult(filename=f.filename, chunks=n, status="indexed"))
        except ValueError as e:
            results.append(IngestResult(filename=f.filename, status="error", error=str(e)))
    return results


@app.get("/rag/sources", tags=["rag"])
def list_sources():
    return get_pipeline().sources()


@app.delete("/rag/sources/{source}", tags=["rag"])
def delete_source(source: str):
    removed = get_pipeline().delete_source(source)
    if removed == 0:
        raise HTTPException(status_code=404, detail=f"No indexed document named '{source}'.")
    return {"deleted": source, "chunks_removed": removed}


@app.post("/rag/query", response_model=QueryResponse, tags=["rag"])
def rag_query(req: QueryRequest):
    """Ask a question about the indexed documents. The answer contains [n] citations that match `sources`."""
    pipe = get_pipeline()
    history = [t.model_dump() for t in req.history]
    return to_response(guarded(pipe.ask, req.question, history=history, top_k=req.top_k,
                               mode="qa", rewrite=req.rewrite_question))


@app.post("/research/ask", response_model=QueryResponse, tags=["research assistant"])
def research_ask(req: QueryRequest):
    """Research-style answer (direct answer + evidence + caveats) over the indexed papers."""
    pipe = get_pipeline()
    history = [t.model_dump() for t in req.history]
    return to_response(guarded(pipe.ask, req.question, history=history, top_k=req.top_k,
                               mode="research", rewrite=req.rewrite_question))


@app.post("/research/summarize", response_model=SummaryResponse, tags=["research assistant"])
def research_summarize(
    file: Optional[UploadFile] = File(None, description="PDF / TXT / MD paper"),
    text: Optional[str] = Form(None, description="...or paste the paper text here"),
    title: str = Form("paper"),
    style: Literal["beginner", "technical", "executive"] = Form("technical"),
    length: Literal["short", "medium", "long"] = Form("medium"),
    fmt: Literal["paragraph", "bullets", "structured"] = Form("structured", alias="format"),
    focus: Optional[str] = Form(None, description='e.g. "methodology and datasets"'),
):
    """Summarise a paper with a DYNAMIC prompt built from style / length / format / focus."""
    if file is not None:
        docs = guarded(load_bytes, file.file.read(), file.filename)
        body, title = "\n\n".join(d.text for d in docs), file.filename
    elif text and text.strip():
        body = text
    else:
        raise HTTPException(status_code=400, detail="Send either a 'file' or a 'text' field.")
    summary = guarded(lambda: summarize(get_llm(), body, title, style, length, fmt, focus))
    return SummaryResponse(title=title, style=style, length=length, format=fmt, summary=summary)
