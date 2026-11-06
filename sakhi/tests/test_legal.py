"""Tests for legal ingestion and retrieval using fake fixtures only."""

from pathlib import Path

import pandas as pd

from sakhi.src.eval_retrieval import evaluate
from sakhi.src.legal_ingest import ingest_sources
from sakhi.src.retrieval import BM25Retriever, search


def test_ingest_parses_sections_and_logs_unparsed(tmp_path: Path) -> None:
    source = tmp_path / "BNS.txt"
    source.write_text(
        "Act: BNS\nSection 78 - Fake title\nFAKE LAW TEXT about stalking.\n"
        "Section 79 -\n",
        encoding="utf-8",
    )
    chunks = ingest_sources(
        tmp_path,
        tmp_path / "chunks" / "chunks.csv",
        tmp_path / "unparsed_sections.txt",
    )
    assert list(chunks["chunk_id"]) == ["BNS-78"]
    assert "BNS.txt: section 79" in (tmp_path / "unparsed_sections.txt").read_text(
        encoding="utf-8"
    )


def test_bm25_keeps_devanagari_and_returns_expected_fields() -> None:
    chunks = pd.DataFrame(
        [
            {
                "chunk_id": "FAKE-1",
                "act": "FAKE LAW TEXT",
                "section_no": "1",
                "section_title": "Fake",
                "text": "FAKE LAW TEXT महिला सुरक्षा",
                "language": "hi",
                "source_file": "fake",
            },
            {
                "chunk_id": "FAKE-2",
                "act": "FAKE LAW TEXT",
                "section_no": "2",
                "section_title": "Fake",
                "text": "FAKE LAW TEXT workplace complaint",
                "language": "en",
                "source_file": "fake",
            },
        ]
    )
    result = BM25Retriever(chunks).search("महिला सुरक्षा", 1)[0]
    assert result["chunk_id"] == "FAKE-1"
    assert set(result) == {"chunk_id", "act", "section_no", "text", "score"}


def test_retrieval_evaluation_reports_metrics() -> None:
    chunks = pd.DataFrame(
        [
            {
                "chunk_id": "FAKE-1",
                "act": "FAKE LAW TEXT",
                "section_no": "1",
                "section_title": "Fake",
                "text": "FAKE LAW TEXT workplace complaint",
                "language": "en",
                "source_file": "fake",
            }
        ]
    )
    testset = pd.DataFrame(
        [{"query": "workplace complaint", "language": "en", "expected_chunk_ids": "FAKE-1"}]
    )
    result = evaluate(testset, BM25Retriever(chunks))
    assert result["queries"] == 1
    assert result["Recall@1"] == 1.0
    assert result["MRR"] == 1.0


def test_empty_configured_corpus_is_safe(tmp_path: Path) -> None:
    path = tmp_path / "chunks.csv"
    pd.DataFrame(
        columns=["chunk_id", "act", "section_no", "section_title", "text", "language", "source_file"]
    ).to_csv(path, index=False)
    assert search("anything", chunks_path=path) == []
