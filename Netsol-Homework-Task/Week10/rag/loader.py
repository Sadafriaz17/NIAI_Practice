"""Step 1 - LOAD: turn files into Document objects."""
import io
from pathlib import Path
from typing import List

from .models import Document

TEXT_EXTENSIONS = {".txt", ".md", ".markdown", ".rst", ".csv"}
SUPPORTED = sorted(TEXT_EXTENSIONS | {".pdf"})


def load_bytes(data: bytes, filename: str) -> List[Document]:
    ext = Path(filename).suffix.lower()
    if ext == ".pdf":
        docs = _load_pdf(io.BytesIO(data), filename)
    elif ext in TEXT_EXTENSIONS:
        docs = [Document(text=data.decode("utf-8", errors="ignore"), source=filename)]
    else:
        raise ValueError(f"Unsupported file type '{ext}'. Supported: {', '.join(SUPPORTED)}")
    docs = [d for d in docs if d.text.strip()]
    if not docs:
        raise ValueError(f"No extractable text in '{filename}' (is it a scanned PDF?).")
    return docs


def load_path(path) -> List[Document]:
    p = Path(path)
    return load_bytes(p.read_bytes(), p.name)


def _load_pdf(stream, filename: str) -> List[Document]:
    from pypdf import PdfReader
    reader = PdfReader(stream)
    docs = []
    for number, page in enumerate(reader.pages, start=1):
        text = (page.extract_text() or "").strip()
        if text:
            docs.append(Document(text=text, source=filename, page=number))
    return docs
