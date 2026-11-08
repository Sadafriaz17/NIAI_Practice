"""Step 6 - GENERATE: one thin wrapper around any OpenAI-compatible chat API (Groq, Gemini, OpenAI, Ollama ...)."""
from typing import Dict, Iterator, List, Optional


class LLMConfigError(RuntimeError):
    """Raised when the LLM is not configured (e.g. missing API key)."""


class LLMClient:
    def __init__(self, settings):
        if not settings.llm_api_key:
            raise LLMConfigError(
                f"No API key for provider '{settings.llm_provider}'. Put the key in your .env file "
                f"(see .env.example) or choose LLM_PROVIDER=ollama.")
        from openai import OpenAI
        self.model = settings.llm_model
        self.default_temperature = settings.temperature
        self._client = OpenAI(base_url=settings.llm_base_url, api_key=settings.llm_api_key)

    def chat(self, messages: List[Dict], temperature: Optional[float] = None, max_tokens: Optional[int] = None) -> str:
        response = self._client.chat.completions.create(
            model=self.model, messages=messages, max_tokens=max_tokens,
            temperature=self.default_temperature if temperature is None else temperature)
        return response.choices[0].message.content or ""

    def stream(self, messages: List[Dict], temperature: Optional[float] = None) -> Iterator[str]:
        events = self._client.chat.completions.create(
            model=self.model, messages=messages, stream=True,
            temperature=self.default_temperature if temperature is None else temperature)
        for event in events:
            if event.choices and event.choices[0].delta and event.choices[0].delta.content:
                yield event.choices[0].delta.content
