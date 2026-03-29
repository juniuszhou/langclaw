"""Embedding models for RAG (provider/model strings)."""

import os
from typing import Any

from langchain_ollama import OllamaEmbeddings


def _ollama_base_url() -> str:
    """URL for the native Ollama HTTP API (not the OpenAI-compatible ``/v1`` path).

    The ``ollama`` CLI respects ``OLLAMA_HOST``; this project historically used
    ``OLLAMA_BASE_URL``. We honor both so pulls from the CLI and app hit the same server.
    """
    raw = (os.getenv("OLLAMA_BASE_URL") or os.getenv("OLLAMA_HOST") or "").strip()
    if not raw:
        return "http://127.0.0.1:11434"
    base = raw.rstrip("/")
    # Common mistake: ChatOpenAI-style base is .../v1; Ollama's embed API is under /api/* at the root.
    if base.endswith("/v1"):
        base = base[: -len("/v1")].rstrip("/")
    return base or "http://127.0.0.1:11434"


def get_embeddings(embedding_model: str) -> Any:
    """Build a LangChain Embeddings instance from ``provider/model``."""
    parts = embedding_model.split("/", 1)
    if len(parts) != 2 or not parts[0] or not parts[1]:
        raise ValueError(
            f"embedding_model must be 'provider/model', got: {embedding_model!r}"
        )
    provider, model_name = parts[0].lower(), parts[1]

    if provider == "ollama":
        return OllamaEmbeddings(
            model=model_name,
            base_url=_ollama_base_url(),
        )

    raise ValueError(
        f"Unknown embedding provider {provider!r}; use ollama, openai, or google."
    )
