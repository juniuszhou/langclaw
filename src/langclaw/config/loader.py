"""Load agent configuration from YAML."""

from pathlib import Path
from typing import Any, Dict, List, Optional

import yaml
from pydantic import BaseModel, Field


class MemoryConfig(BaseModel):
    """Memory backend configuration."""

    backend: str = "memory"
    long_term: Optional[str] = None


class MCPConfig(BaseModel):
    """MCP server configuration for loading tools."""

    server: str = "npx"
    args: List[str] = Field(default_factory=list)
    env: dict = Field(default_factory=dict)


class RAGConfig(BaseModel):
    """Retrieval-augmented generation: local corpus + embeddings."""

    enabled: bool = True
    sources: List[str] = Field(
        default_factory=list,
        description="Paths relative to app dir (files or directories).",
    )
    embedding_model: str = Field(
        default="ollama/nomic-embed-text",
        description="Provider/model, e.g. ollama/nomic-embed-text, openai/text-embedding-3-small",
    )
    chunk_size: int = 800
    chunk_overlap: int = 120
    k: int = 4
    persist_directory: Optional[str] = Field(
        default=None,
        description="If set, load/save FAISS index under this path (relative to app dir).",
    )
    include_pdf: bool = Field(
        default=False,
        description="Load *.pdf from source trees (requires pypdf / rag extra).",
    )


class AgentConfig(BaseModel):
    """Configuration for a single agent."""

    model: str = Field(description="Model ID, e.g. openai/gpt-4o, ollama/llama3")
    system_prompt: str = "You are a helpful AI assistant."
    channels: List[str] = Field(default_factory=lambda: ["terminal"])
    memory: Optional[MemoryConfig] = None
    tools: List[str] = Field(default_factory=list)
    skills: List[str] = Field(default_factory=list)
    mcp: Optional[MCPConfig] = None
    rag: Optional[RAGConfig] = None
    max_turns: int = 50
    temperature: float = 0.7


class Config(BaseModel):
    """Root configuration."""

    agents: Dict[str, AgentConfig] = Field(default_factory=dict)


def load_config(path: Optional[Path] = None) -> Config:
    """Load config from YAML file."""
    path = path or Path("config.yaml")
    if not path.exists():
        return Config()
    with open(path) as f:
        data = yaml.safe_load(f) or {}
    return Config(**data)
