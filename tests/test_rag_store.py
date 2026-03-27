from pathlib import Path

from langclaw.rag.store import _collect_paths


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
