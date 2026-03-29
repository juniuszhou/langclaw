"""Embedding models for RAG (provider/model strings)."""

import importlib
import os
from typing import Any


def get_embeddings(embedding_model: str) -> Any:
    """Build a LangChain Embeddings instance from ``provider/model``."""
    parts = embedding_model.split("/", 1)
    if len(parts) != 2 or not parts[0] or not parts[1]:
        raise ValueError(
            f"embedding_model must be 'provider/model', got: {embedding_model!r}"
        )
    provider, model_name = parts[0].lower(), parts[1]

    if provider == "ollama":
        ollama = importlib.import_module("langchain_community.embeddings.ollama")
        OllamaEmbeddings = ollama.OllamaEmbeddings
        return OllamaEmbeddings(
            model=model_name,
            base_url=os.getenv("OLLAMA_BASE_URL", "http://localhost:11434"),
        )

    raise ValueError(
        f"Unknown embedding provider {provider!r}; use ollama, openai, or google."
    )
