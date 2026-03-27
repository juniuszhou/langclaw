"""Embedding models for RAG (provider/model strings)."""

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
        from langchain_community.embeddings import OllamaEmbeddings

        return OllamaEmbeddings(
            model=model_name,
            base_url=os.getenv("OLLAMA_BASE_URL", "http://localhost:11434"),
        )

    if provider == "openai":
        from langchain_openai import OpenAIEmbeddings

        return OpenAIEmbeddings(
            model=model_name,
            api_key=os.getenv("OPENAI_API_KEY"),
        )

    if provider == "google":
        from langchain_google_genai import GoogleGenerativeAIEmbeddings

        key = os.getenv("GOOGLE_API_KEY")
        return GoogleGenerativeAIEmbeddings(model=model_name, google_api_key=key)

    raise ValueError(
        f"Unknown embedding provider {provider!r}; use ollama, openai, or google."
    )
