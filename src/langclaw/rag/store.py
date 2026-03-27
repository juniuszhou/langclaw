"""Build FAISS vector store and retriever from local sources."""

from __future__ import annotations

from pathlib import Path
from typing import List

from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

from langclaw.config.loader import RAGConfig
from langclaw.rag.embeddings import get_embeddings

_RAG_INSTALL_HINT = (
    "RAG requires optional dependencies. Install with: pip install -e '.[rag]'"
)


def _require_faiss():
    try:
        from langchain_community.vectorstores import FAISS  # noqa: F401
    except ImportError as e:
        raise ImportError(_RAG_INSTALL_HINT) from e


def _load_file(path: Path, include_pdf: bool) -> List[Document]:
    suffix = path.suffix.lower()
    if suffix in (".md", ".txt"):
        from langchain_community.document_loaders import TextLoader

        loader = TextLoader(str(path), encoding="utf-8")
        return loader.load()
    if suffix == ".pdf" and include_pdf:
        try:
            from langchain_community.document_loaders import PyPDFLoader
        except ImportError as e:
            raise ImportError(
                "PDF ingest needs pypdf. Install with: pip install -e '.[rag]'"
            ) from e
        return PyPDFLoader(str(path)).load()
    return []


def _collect_paths(app_dir: Path, sources: List[str], include_pdf: bool) -> List[Path]:
    paths: List[Path] = []
    exts = {".md", ".txt"}
    if include_pdf:
        exts.add(".pdf")
    for rel in sources:
        root = (app_dir / rel).resolve()
        if not root.exists():
            continue
        if root.is_file():
            paths.append(root)
        else:
            for ext in exts:
                paths.extend(sorted(root.rglob(f"*{ext}")))
    # De-duplicate while preserving order
    seen: set = set()
    out: List[Path] = []
    for p in paths:
        rp = p.resolve()
        if rp not in seen:
            seen.add(rp)
            out.append(rp)
    return out


def load_documents(
    app_dir: Path,
    sources: List[str],
    include_pdf: bool,
) -> List[Document]:
    """Load documents from configured sources (files or directories)."""
    docs: List[Document] = []
    for path in _collect_paths(app_dir, sources, include_pdf):
        docs.extend(_load_file(path, include_pdf))
    return docs


def build_rag_retriever(cfg: RAGConfig, app_dir: Path):
    """Create a retriever over a FAISS index (load or build + optional persist)."""
    _require_faiss()
    from langchain_community.vectorstores import FAISS

    if not cfg.sources:
        raise ValueError("RAG is enabled but rag.sources is empty.")

    embeddings = get_embeddings(cfg.embedding_model)

    persist: Path | None = None
    if cfg.persist_directory:
        persist = (app_dir / cfg.persist_directory).resolve()
        persist.mkdir(parents=True, exist_ok=True)
        index_file = persist / "index.faiss"
        if index_file.exists():
            return FAISS.load_local(
                str(persist),
                embeddings,
                allow_dangerous_deserialization=True,
            ).as_retriever(search_kwargs={"k": cfg.k})

    documents = load_documents(app_dir, cfg.sources, cfg.include_pdf)
    if not documents:
        raise FileNotFoundError(
            f"No documents found under sources {cfg.sources!r} (relative to {app_dir})."
        )

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=cfg.chunk_size,
        chunk_overlap=cfg.chunk_overlap,
    )
    splits = splitter.split_documents(documents)
    store = FAISS.from_documents(splits, embeddings)
    if persist is not None:
        store.save_local(str(persist))
    return store.as_retriever(search_kwargs={"k": cfg.k})
