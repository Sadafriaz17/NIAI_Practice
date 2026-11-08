"""Streamlit GUI for the RAG mini project.

    streamlit run app_streamlit.py          ->  http://localhost:8501
"""
from pathlib import Path

import streamlit as st

from rag.config import get_settings
from rag.llm import LLMClient, LLMConfigError
from rag.loader import SUPPORTED, load_bytes
from rag.pipeline import RAGPipeline
from rag.prompts import FORMATS, LENGTHS, STYLES
from rag.research import summarize

st.set_page_config(page_title="RAG Mini Project", page_icon="📚", layout="wide")
settings = get_settings()


@st.cache_resource(show_spinner="Loading the embedding model (the first run downloads it)…")
def get_pipeline() -> RAGPipeline:
    return RAGPipeline()


pipe = get_pipeline()


def show_sources(sources):
    with st.expander(f"📎 Sources ({len(sources)})"):
        for i, s in enumerate(sources, start=1):
            page = f", page {s['page']}" if s.get("page") else ""
            st.markdown(f"**[{i}] {s['source']}{page}** · similarity {s['score']:.2f}")
            st.caption(s["text"])


# ----------------------------------------------------------------------------- sidebar
with st.sidebar:
    st.title("📚 RAG Mini Project")
    st.caption(f"LLM: {settings.llm_provider} / {settings.llm_model}\n\nEmbeddings: {pipe.embedder.name}")
    mode = st.radio("Mode", ["💬 Chat with documents", "📝 Research summary", "🗂️ Index inspector"])

    st.divider()
    st.subheader("Documents")
    uploads = st.file_uploader("Upload PDF / TXT / MD", type=[e.lstrip(".") for e in SUPPORTED],
                               accept_multiple_files=True)
    if st.button("Index uploaded files", disabled=not uploads, use_container_width=True):
        for f in uploads:
            try:
                with st.spinner(f"Indexing {f.name} …"):
                    n = pipe.ingest_bytes(f.getvalue(), f.name)
                st.success(f"{f.name}: {n} chunks")
            except Exception as e:
                st.error(f"{f.name}: {e}")

    for name, n in pipe.sources().items():
        c1, c2 = st.columns([5, 1])
        c1.write(f"📄 {name} ({n} chunks)")
        if c2.button("✕", key=f"del_{name}", help="Remove from the index"):
            pipe.delete_source(name)
            st.rerun()

    st.divider()
    top_k = st.slider("Passages to retrieve (top-k)", 1, 10, settings.top_k)
    temperature = st.slider("Temperature", 0.0, 1.0, settings.temperature, 0.05)
    rewrite = st.checkbox("Rewrite follow-up questions", value=True,
                          help="Uses the chat history to turn 'and its limits?' into a full question before searching.")
    if st.button("Clear chat", use_container_width=True):
        st.session_state.messages = []
        st.rerun()

# ----------------------------------------------------------------------------- chat
if mode.startswith("💬"):
    st.header("💬 Chat with your documents")
    if "messages" not in st.session_state:
        st.session_state.messages = []
    if not pipe.sources():
        st.info("Upload and index at least one document in the sidebar (try `sample_docs/rag_overview.txt`).")

    for m in st.session_state.messages:
        with st.chat_message(m["role"]):
            st.markdown(m["content"])
            if m.get("sources"):
                show_sources(m["sources"])

    question = st.chat_input("Ask a question about your documents…")
    if question:
        history = [{"role": m["role"], "content": m["content"]} for m in st.session_state.messages]
        st.session_state.messages.append({"role": "user", "content": question})
        with st.chat_message("user"):
            st.markdown(question)
        with st.chat_message("assistant"):
            try:
                prepared = pipe.prepare(question, history, top_k, rewrite=rewrite)
                if prepared.standalone_question != question:
                    st.caption(f"🔎 Searched for: *{prepared.standalone_question}*")
                answer = st.write_stream(pipe.stream_answer(prepared, temperature))
                sources = [{"source": r.chunk.source, "page": r.chunk.page, "score": r.score, "text": r.chunk.text}
                           for r in prepared.retrieved]
                if sources:
                    show_sources(sources)
                st.session_state.messages.append({"role": "assistant", "content": answer, "sources": sources})
            except LLMConfigError as e:
                st.error(str(e))
            except Exception as e:
                st.error(f"Something went wrong: {e}")

# ----------------------------------------------------------------------------- research summary
elif mode.startswith("📝"):
    st.header("📝 Research assistant: summary with a dynamic prompt")
    c1, c2, c3 = st.columns(3)
    style = c1.selectbox("Style", list(STYLES), index=1)
    length = c2.selectbox("Length", list(LENGTHS), index=1)
    fmt = c3.selectbox("Format", list(FORMATS), index=2)
    focus = st.text_input("Focus (optional)", placeholder="e.g. methodology and datasets")
    paper = st.file_uploader("Paper (PDF / TXT / MD)", type=["pdf", "txt", "md"], key="paper")
    pasted = st.text_area("…or paste the text", height=150)

    if st.button("Summarise", type="primary"):
        try:
            if paper:
                text = "\n\n".join(d.text for d in load_bytes(paper.getvalue(), paper.name))
                title = paper.name
            elif pasted.strip():
                text, title = pasted, "pasted text"
            else:
                st.warning("Upload a paper or paste some text first.")
                st.stop()
            with st.spinner("Summarising (long papers take a little longer) …"):
                st.session_state.summary = (title, summarize(LLMClient(settings), text, title, style, length, fmt, focus or None))
        except LLMConfigError as e:
            st.error(str(e))
        except Exception as e:
            st.error(f"Something went wrong: {e}")

    if st.session_state.get("summary"):
        title, result = st.session_state.summary
        st.subheader(title)
        st.markdown(result)
        st.download_button("Download as .md", result, file_name=f"{Path(title).stem}_summary.md")

# ----------------------------------------------------------------------------- inspector
else:
    st.header("🗂️ Index inspector")
    a, b = st.columns(2)
    a.metric("Documents", len(pipe.sources()))
    b.metric("Chunks", len(pipe.store))
    st.write(pipe.sources() or "The index is empty.")
    test_q = st.text_input("Try a retrieval query (no LLM is called)")
    if test_q:
        for i, r in enumerate(pipe.retrieve(test_q, top_k), start=1):
            page = f", page {r.chunk.page}" if r.chunk.page else ""
            st.markdown(f"**{i}. {r.chunk.source}{page}** · similarity {r.score:.3f}")
            st.caption(r.chunk.text)
