"""Create a LangChain chat model for ANY OpenAI-compatible provider.

Choose the provider in the .env file (see .env.example):
    LLM_PROVIDER = groq | gemini | openai | ollama
Optional overrides: LLM_MODEL, LLM_BASE_URL, LLM_API_KEY
"""
import os

from dotenv import load_dotenv
from langchain_openai import ChatOpenAI

load_dotenv()

# base_url, name of the env variable that holds the API key, default model.
# Model names change over time: if one is rejected, set LLM_MODEL to a model
# that your provider currently offers.
PRESETS = {
    "groq":   {"base_url": "https://api.groq.com/openai/v1", "key_env": "GROQ_API_KEY", "model": "llama-3.3-70b-versatile"},
    "gemini": {"base_url": "https://generativelanguage.googleapis.com/v1beta/openai/", "key_env": "GOOGLE_API_KEY", "model": "gemini-2.0-flash"},
    "openai": {"base_url": None, "key_env": "OPENAI_API_KEY", "model": "gpt-4o-mini"},
    "ollama": {"base_url": "http://localhost:11434/v1", "key_env": None, "model": "llama3.2"},  # free, local
}


def get_llm(temperature: float = 0.2, streaming: bool = False) -> ChatOpenAI:
    provider = os.getenv("LLM_PROVIDER", "groq").lower()
    if provider not in PRESETS:
        raise ValueError(f"Unknown LLM_PROVIDER '{provider}'. Use one of: {', '.join(PRESETS)}")
    preset = PRESETS[provider]

    base_url = os.getenv("LLM_BASE_URL") or preset["base_url"]
    model = os.getenv("LLM_MODEL") or preset["model"]
    key_env = preset["key_env"]
    api_key = os.getenv("LLM_API_KEY") or (os.getenv(key_env) if key_env else None)
    if provider == "ollama" and not api_key:
        api_key = "ollama"  # Ollama ignores the key but the client needs a non-empty string
    if not api_key:
        raise RuntimeError(f"No API key found. Put {key_env}=... in your .env file (see .env.example).")

    return ChatOpenAI(model=model, base_url=base_url, api_key=api_key,
                      temperature=temperature, streaming=streaming)
