#!/usr/bin/env python3
"""Tests for RAG document loading (no embeddings required)."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent / "src"))

from langclaw.rag.store import load_documents


def test_load_rag_fixture_documents():
    app_dir = Path(__file__).parent.resolve()
    docs = load_documents(app_dir, ["rag_fixtures"], include_pdf=False)
    text = "\n".join(d.page_content for d in docs)
    assert "RAGfixture-7d4c" in text


if __name__ == "__main__":
    test_load_rag_fixture_documents()
    print("ok")
