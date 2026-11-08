# Week 10 – Mini-Project: complete RAG (GUI + Docker) and FastAPI Research Assistant

```
Week10/
├── rag/                  the RAG library, one file per component
│   ├── loader.py         1. load PDF / TXT / MD
│   ├── chunker.py        2. split into overlapping chunks
│   ├── embedder.py       3. embeddings (fastembed / bge-small, runs locally)
│   ├── vector_store.py   4. vector store (numpy, cosine search, saved to disk)
│   ├── pipeline.py       5. retriever + prompt + generation, follow-up question rewriting
│   ├── prompts.py        prompts (QA, research mode, dynamic summary prompt)
│   ├── llm.py            6. LLM client (Groq / Gemini / OpenAI / Ollama)
│   └── research.py       Research Assistant (dynamic prompt, map-reduce for long papers)
├── app_streamlit.py      GUI  (task 1)
├── api.py                FastAPI endpoints  (task 2)
├── Dockerfile, docker-compose.yml, .dockerignore   (Docker image, task 1)
├── tests/test_core.py    offline test (no API key needed)
└── sample_docs/rag_overview.txt
```

## A. Run locally
```bash
cd Week10
python -m venv .venv
# Windows: .venv\Scripts\activate          macOS/Linux: source .venv/bin/activate
pip install -r requirements.txt
copy .env.example .env        # (macOS/Linux: cp .env.example .env)  -> put your API key in .env
python tests/test_core.py     # optional: checks chunking / retrieval / saving without any API key
```

**GUI**
```bash
streamlit run app_streamlit.py          # http://localhost:8501
```
Upload PDFs in the sidebar → *Index uploaded files* → ask questions (answers stream in with [1], [2] citations and expandable sources).
Other modes: *Research summary* (style / length / format / focus) and *Index inspector* (test retrieval without the LLM).

**API**
```bash
uvicorn api:app --reload --port 8000    # Swagger UI: http://localhost:8000/docs
```
| Method | Path | What it does |
|---|---|---|
| GET | `/health` | status, number of indexed documents / chunks |
| POST | `/rag/ingest` | upload one or more files (multipart `files`) |
| GET | `/rag/sources` · DELETE `/rag/sources/{name}` | list / remove documents |
| POST | `/rag/query` | `{"question": "...", "top_k": 4, "history": []}` → answer + cited sources |
| POST | `/research/ask` | same, research-style answer (answer / evidence / caveats) |
| POST | `/research/summarize` | upload a paper (or `text`) + `style`, `length`, `format`, `focus` → dynamic-prompt summary |

Quick test (the Swagger page at `/docs` is the easiest way; with curl):
```bash
curl -X POST http://localhost:8000/rag/ingest -F "files=@sample_docs/rag_overview.txt"
curl -X POST http://localhost:8000/rag/query -H "Content-Type: application/json" -d "{\"question\": \"What are the components of a RAG system?\"}"
curl -X POST http://localhost:8000/research/summarize -F "file=@your_paper.pdf" -F "style=beginner" -F "length=short" -F "format=bullets"
```
If you set `API_KEY` in `.env`, add `-H "X-API-Key: <your key>"` to every request.

## B. Run with Docker
```bash
cd Week10
copy .env.example .env                 # fill in your key first!
docker build -t rag-mini-project .

# GUI
docker run --rm -p 8501:8501 --env-file .env -v rag_data:/app/data rag-mini-project
# API
docker run --rm -p 8000:8000 --env-file .env -v rag_data:/app/data rag-mini-project uvicorn api:app --host 0.0.0.0 --port 8000

# or both at once (shared index volume):
docker compose up --build
```
GUI: http://localhost:8501 · API docs: http://localhost:8000/docs
The embedding model is downloaded while the image is built, so the first start is fast. The index lives in the `rag_data` volume.
Using Ollama on your own PC from inside Docker: set `LLM_PROVIDER=ollama` and `LLM_BASE_URL=http://host.docker.internal:11434/v1` in `.env`
(on Linux also uncomment `extra_hosts` in `docker-compose.yml`).

## Troubleshooting
* *"No API key …"* → fill `.env` (or use `LLM_PROVIDER=ollama`).
* *Model not found / decommissioned* → set `LLM_MODEL` in `.env` to a model your provider currently offers.
* *"saved index was built with a different embedder"* → delete `data/index` and index again.
* Scanned PDFs have no text layer and cannot be indexed (they need OCR).
