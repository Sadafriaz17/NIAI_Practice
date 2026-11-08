"""All settings come from environment variables (or a .env file)."""
import os
from dataclasses import dataclass
from pathlib import Path

try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:  # python-dotenv is optional
    pass

BASE_DIR = Path(__file__).resolve().parent.parent

# Model names change over time: if one is rejected, set LLM_MODEL in .env.
PROVIDER_PRESETS = {
    "groq":   {"base_url": "https://api.groq.com/openai/v1", "key_env": "GROQ_API_KEY", "model": "llama-3.3-70b-versatile"},
    "gemini": {"base_url": "https://generativelanguage.googleapis.com/v1beta/openai/", "key_env": "GOOGLE_API_KEY", "model": "gemini-2.0-flash"},
    "openai": {"base_url": "https://api.openai.com/v1", "key_env": "OPENAI_API_KEY", "model": "gpt-4o-mini"},
    "ollama": {"base_url": "http://localhost:11434/v1", "key_env": None, "model": "llama3.2"},
}


@dataclass
class Settings:
    # LLM
    llm_provider: str
    llm_base_url: str
    llm_api_key: str
    llm_model: str
    temperature: float
    # embeddings
    embedder: str            # "fastembed" (real) or "hash" (tiny offline stand-in, used for tests)
    embed_model: str
    # chunking / retrieval
    chunk_size: int
    chunk_overlap: int
    top_k: int
    min_score: float
    # storage
    data_dir: Path
    index_dir: Path
    model_cache_dir: Path


def get_settings() -> Settings:
    provider = os.getenv("LLM_PROVIDER", "groq").lower()
    if provider not in PROVIDER_PRESETS:
        raise ValueError(f"Unknown LLM_PROVIDER '{provider}'. Use one of: {', '.join(PROVIDER_PRESETS)}")
    preset = PROVIDER_PRESETS[provider]
    key_env = preset["key_env"]
    api_key = os.getenv("LLM_API_KEY") or (os.getenv(key_env) if key_env else "") or ""
    if provider == "ollama" and not api_key:
        api_key = "ollama"

    data_dir = Path(os.getenv("DATA_DIR", BASE_DIR / "data"))
    return Settings(
        llm_provider=provider,
        llm_base_url=os.getenv("LLM_BASE_URL") or preset["base_url"],
        llm_api_key=api_key,
        llm_model=os.getenv("LLM_MODEL") or preset["model"],
        temperature=float(os.getenv("LLM_TEMPERATURE", "0.2")),
        embedder=os.getenv("EMBEDDER", "fastembed").lower(),
        embed_model=os.getenv("EMBED_MODEL", "BAAI/bge-small-en-v1.5"),
        chunk_size=int(os.getenv("CHUNK_SIZE", "800")),
        chunk_overlap=int(os.getenv("CHUNK_OVERLAP", "150")),
        top_k=int(os.getenv("TOP_K", "4")),
        min_score=float(os.getenv("MIN_SCORE", "0.0")),
        data_dir=data_dir,
        index_dir=Path(os.getenv("INDEX_DIR", data_dir / "index")),
        model_cache_dir=Path(os.getenv("MODEL_CACHE_DIR", data_dir / "models")),
    )
