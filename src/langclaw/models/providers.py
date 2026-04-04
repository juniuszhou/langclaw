"""Model provider registry: OpenAI, Anthropic, Ollama, Google."""

import os
from typing import Any, Optional

from langchain_core.language_models import BaseChatModel
from pydantic import BaseModel, Field, SecretStr


class ModelConfig(BaseModel):
    """Configuration for an LLM model."""

    model_id: str = Field(
        description="Model identifier, e.g. openai/gpt-4o, ollama/llama3"
    )
    temperature: float = 0.7
    api_key: Optional[str] = None
    base_url: Optional[str] = None


class ModelProvider:
    """Factory for creating chat models from config."""

    @staticmethod
    def get_openai(config: ModelConfig) -> BaseChatModel:
        """Create OpenAI-compatible model (OpenAI, Azure, Ollama)."""
        from langchain_openai import ChatOpenAI

        # Parse model_id: "openai/gpt-4o" or "ollama/llama3"
        parts = config.model_id.split("/", 1)
        provider = parts[0].lower() if len(parts) > 1 else "openai"
        model_name = parts[1] if len(parts) > 1 else config.model_id

        if provider == "ollama":
            ollama_key = config.api_key or "ollama"
            return ChatOpenAI(
                model=model_name,
                temperature=config.temperature,
                base_url=config.base_url or "http://localhost:11434/v1",
                api_key=SecretStr(ollama_key),
            )
        openai_key = config.api_key or os.getenv("OPENAI_API_KEY")
        return ChatOpenAI(
            model=model_name,
            temperature=config.temperature,
            api_key=SecretStr(openai_key) if openai_key else None,
            base_url=config.base_url,
        )

    @staticmethod
    def get_anthropic(config: ModelConfig) -> BaseChatModel:
        """Create Anthropic model."""
        from langchain_anthropic import ChatAnthropic

        model_name = (
            config.model_id.split("/", 1)[-1]
            if "/" in config.model_id
            else config.model_id
        )
        anthropic_key = config.api_key or os.getenv("ANTHROPIC_API_KEY")
        # ChatAnthropic exposes model/timeout/stop under Pydantic aliases model_name, timeout, stop;
        # pyright matches those names. Prefer passing api_key as SecretStr like other LC models.
        if anthropic_key:
            return ChatAnthropic(
                model_name=model_name,
                temperature=config.temperature,
                api_key=SecretStr(anthropic_key),
                timeout=None,
                stop=None,
            )
        return ChatAnthropic(
            model_name=model_name,
            temperature=config.temperature,
            timeout=None,
            stop=None,
        )

    @staticmethod
    def get_google(config: ModelConfig) -> BaseChatModel:
        """Create Google Gemini model."""
        from langchain_google_genai import ChatGoogleGenerativeAI

        model_name = (
            config.model_id.split("/", 1)[-1]
            if "/" in config.model_id
            else config.model_id
        )
        return ChatGoogleGenerativeAI(
            model=model_name,
            temperature=config.temperature,
            api_key=config.api_key or os.getenv("GOOGLE_API_KEY"),
        )

    @classmethod
    def get(cls, config: ModelConfig) -> BaseChatModel:
        """Create model from config. Infers provider from model_id prefix."""
        prefix = config.model_id.split("/", 1)[0].lower()
        if prefix == "anthropic":
            return cls.get_anthropic(config)
        if prefix == "google":
            return cls.get_google(config)
        # openai, ollama, or default
        return cls.get_openai(config)


def get_model(
    model_id: str,
    temperature: float = 0.7,
    api_key: Optional[str] = None,
    base_url: Optional[str] = None,
) -> BaseChatModel:
    """Convenience function to get a chat model."""
    config = ModelConfig(
        model_id=model_id,
        temperature=temperature,
        api_key=api_key,
        base_url=base_url,
    )
    return ModelProvider.get(config)
