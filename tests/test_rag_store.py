import importlib
from pathlib import Path
from unittest.mock import patch

from langclaw.config.loader import RAGConfig
from langclaw.rag.store import _collect_paths, build_rag_retriever, load_documents


def test_collect_paths_collects_supported_extensions_and_dedupes(tmp_path: Path):
    docs = tmp_path / "docs"
    docs.mkdir()
    a = docs / "a.md"
    b = docs / "b.txt"
    ignored = docs / "ignore.json"
    a.write_text("a", encoding="utf-8")
    b.write_text("b", encoding="utf-8")
    ignored.write_text("{}", encoding="utf-8")

    # Include the same file via direct path + directory path to verify de-duping.
    sources = [str(docs.relative_to(tmp_path)), str(a.relative_to(tmp_path))]
    paths = _collect_paths(tmp_path, sources, include_pdf=False)

    assert set(paths) == {a.resolve(), b.resolve()}
    assert len(paths) == 2


def test_collect_paths_includes_pdf_only_when_enabled(tmp_path: Path):
    p = tmp_path / "data.pdf"
    p.write_text("fake-pdf", encoding="utf-8")

    without_pdf = _collect_paths(tmp_path, ["data.pdf"], include_pdf=False)
    with_pdf = _collect_paths(tmp_path, ["data.pdf"], include_pdf=True)

    assert without_pdf == [p.resolve()]
    assert with_pdf == [p.resolve()]


def test_build_rag_parse_documents_persist_and_reload_vector_index(tmp_path: Path):
    """Load text from disk, embed with FAISS, persist index files, reload and retrieve."""
    print("start testing =================================================")
    # assert False == True
    unique = "RAG-VECTOR-PERSIST-f8c2"
    corpus = tmp_path / "corpus"
    print("path is ", corpus.absolute())
    corpus.mkdir()
    (corpus / "note.md").write_text(
        f"{unique} stored in markdown for retrieval.", encoding="utf-8"
    )

    docs = load_documents(tmp_path, ["corpus"], include_pdf=False)
    assert docs
    assert any(unique in d.page_content for d in docs)

    cfg = RAGConfig(
        sources=["corpus"],
        persist_directory="faiss_index",
        k=2,
        chunk_size=200,
        chunk_overlap=0,
        embedding_model="nomic-embed-text",
    )
    fake_mod = importlib.import_module("langchain_community.embeddings.fake")
    fake_emb = fake_mod.DeterministicFakeEmbedding(size=32)

    with patch("langclaw.rag.store.get_embeddings", return_value=fake_emb):
        retriever = build_rag_retriever(cfg, tmp_path)

    index_dir = tmp_path / "faiss_index"
    assert (index_dir / "index.faiss").is_file()
    assert (index_dir / "index.pkl").is_file()

    hits = retriever.invoke(unique)
    assert hits and unique in hits[0].page_content

    with patch("langclaw.rag.store.get_embeddings", return_value=fake_emb):
        retriever_reloaded = build_rag_retriever(cfg, tmp_path)
    hits2 = retriever_reloaded.invoke(unique)
    assert hits2 and unique in hits2[0].page_content
