"""Tests for legal ingestion and retrieval using fake fixtures only."""

from pathlib import Path

import pandas as pd

from sakhi.src.eval_retrieval import evaluate
from sakhi.src.legal_ingest import ingest_sources
from sakhi.src.legal_ingest import _parse_file
from sakhi.src.legal_ingest import validate_chunk_openings
from sakhi.src.legal_pdf_to_text import convert_all
from sakhi.src.retrieval import (
    BM25KeywordRetriever,
    BM25Retriever,
    BM25TextCuratedRetriever,
    search,
)


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
        tmp_path / "manual_overrides",
    )
    assert list(chunks["chunk_id"]) == ["BNS-78"]
    assert chunks.iloc[0]["text"].startswith("Section 78 - Fake title")
    assert chunks.iloc[0]["section_title"] == "Fake title"
    assert "BNS.txt: section 79" in (tmp_path / "unparsed_sections.txt").read_text(
        encoding="utf-8"
    )


def test_ingest_rejects_chunk_without_section_number_opening() -> None:
    invalid = [
        {
            "chunk_id": "FAKE-1",
            "section_no": "1",
            "text": "FAKE LAW TEXT",
        }
    ]
    assert validate_chunk_openings(invalid) == ["FAKE-1"]


def test_ingest_applies_fake_manual_override(tmp_path: Path) -> None:
    source = tmp_path / "BNS.txt"
    source.write_text(
        "Section 1 - Original title\nFAKE LAW TEXT original.\n",
        encoding="utf-8",
    )
    overrides = tmp_path / "manual_overrides"
    overrides.mkdir()
    (overrides / "BNS_2.txt").write_text(
        "2. Fake override title\nFAKE LAW TEXT override.\n",
        encoding="utf-8",
    )
    chunks = ingest_sources(
        tmp_path,
        tmp_path / "chunks.csv",
        tmp_path / "unparsed.txt",
        overrides,
    )
    override = chunks.loc[chunks["chunk_id"] == "BNS-2"].iloc[0]
    assert override["text"] == "2. Fake override title\nFAKE LAW TEXT override."
    assert override["source_file"] == "BNS_2.txt"


def test_bm25_keeps_devanagari_and_returns_expected_fields() -> None:
    chunks = pd.DataFrame(
        [
            {
                "chunk_id": "FAKE-1",
                "act": "FAKE LAW TEXT",
                "section_no": "1",
                "section_title": "Fake",
                "text": "1. FAKE LAW TEXT महिला सुरक्षा and reporting process.",
                "language": "hi",
                "source_file": "fake",
            },
            {
                "chunk_id": "FAKE-2",
                "act": "FAKE LAW TEXT",
                "section_no": "2",
                "section_title": "Fake",
                "text": "2. FAKE LAW TEXT workplace complaint and reporting process.",
                "language": "en",
                "source_file": "fake",
            },
        ]
    )
    result = BM25Retriever(chunks).search("महिला सुरक्षा", 1)[0]
    assert result["chunk_id"] == "FAKE-1"
    assert set(result) == {"chunk_id", "act", "section_no", "text", "score"}


def test_keyword_retriever_uses_fake_keywords_without_returning_them(tmp_path: Path) -> None:
    chunks = pd.DataFrame(
        [
            {
                "chunk_id": "FAKE-1",
                "act": "FAKE LAW TEXT",
                "section_no": "1",
                "section_title": "Fake",
                "text": "1. FAKE LAW TEXT about workplace complaints and reporting.",
                "language": "en",
                "source_file": "fake",
            },
            {
                "chunk_id": "FAKE-2",
                "act": "FAKE LAW TEXT",
                "section_no": "2",
                "section_title": "Fake",
                "text": "2. FAKE LAW TEXT about online abuse and reporting.",
                "language": "en",
                "source_file": "fake",
            },
        ]
    )
    keywords = tmp_path / "keywords.csv"
    pd.DataFrame(
        [
            {"chunk_id": "FAKE-1", "keywords": "workplace complaint"},
            {"chunk_id": "FAKE-2", "keywords": "online abuse"},
        ]
    ).to_csv(keywords, index=False)
    provisions = tmp_path / "provisions.yaml"
    provisions.write_text("fake: [FAKE-1, FAKE-2]\n", encoding="utf-8")
    result = BM25KeywordRetriever(chunks, keywords, provisions).search(
        "workplace complaint", 1
    )[0]
    assert result["chunk_id"] == "FAKE-1"
    assert result["text"] == chunks.iloc[0]["text"]


def test_curated_plain_bm25_uses_same_fake_corpus_without_keywords(
    tmp_path: Path,
) -> None:
    chunks = pd.DataFrame(
        [
            {
                "chunk_id": "FAKE-1",
                "act": "FAKE LAW TEXT",
                "section_no": "1",
                "section_title": "Fake",
                "text": "1. FAKE LAW TEXT about workplace complaints and reporting.",
                "language": "en",
                "source_file": "fake",
            },
            {
                "chunk_id": "FAKE-2",
                "act": "FAKE LAW TEXT",
                "section_no": "2",
                "section_title": "Fake",
                "text": "2. FAKE LAW TEXT about online abuse and reporting.",
                "language": "en",
                "source_file": "fake",
            },
        ]
    )
    keywords = tmp_path / "keywords.csv"
    pd.DataFrame(
        [
            {"chunk_id": "FAKE-1", "keywords": "workplace complaint"},
            {"chunk_id": "FAKE-2", "keywords": "online abuse"},
        ]
    ).to_csv(keywords, index=False)
    provisions = tmp_path / "provisions.yaml"
    provisions.write_text("fake: [FAKE-1, FAKE-2]\n", encoding="utf-8")
    result = BM25TextCuratedRetriever(chunks, keywords, provisions).search(
        "workplace complaints", 1
    )[0]
    assert result["chunk_id"] == "FAKE-1"


def test_bm25_text_excludes_out_of_range_fake_chunks() -> None:
    chunks = pd.DataFrame(
        [
            {
                "chunk_id": "FAKE-SHORT",
                "act": "FAKE LAW TEXT",
                "section_no": "1",
                "section_title": "Fake",
                "text": "1. short",
                "language": "en",
                "source_file": "fake",
            },
            {
                "chunk_id": "FAKE-OK",
                "act": "FAKE LAW TEXT",
                "section_no": "2",
                "section_title": "Fake",
                "text": "2. FAKE LAW TEXT with enough content for retrieval testing.",
                "language": "en",
                "source_file": "fake",
            },
        ]
    )
    retriever = BM25Retriever(chunks)
    assert retriever.excluded_ids == ["FAKE-SHORT"]
    assert retriever.search("retrieval testing", 1)[0]["chunk_id"] == "FAKE-OK"


def test_retrieval_evaluation_reports_metrics() -> None:
    chunks = pd.DataFrame(
        [
            {
                "chunk_id": "FAKE-1",
                "act": "FAKE LAW TEXT",
                "section_no": "1",
                "section_title": "Fake",
                "text": "1. FAKE LAW TEXT workplace complaint and reporting process.",
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
    assert result["by_language"]["en"]["Recall@5"] == 1.0
    assert result["by_language"]["en"]["MRR"] == 1.0


def test_empty_configured_corpus_is_safe(tmp_path: Path) -> None:
    path = tmp_path / "chunks.csv"
    pd.DataFrame(
        columns=["chunk_id", "act", "section_no", "section_title", "text", "language", "source_file"]
    ).to_csv(path, index=False)
    assert search("anything", chunks_path=path) == []


def _fake_pdf(texts: list[str]) -> bytes:
    objects = [
        b"<< /Type /Catalog /Pages 2 0 R >>",
        b"<< /Type /Pages /Kids [3 0 R] /Count 1 >>",
        b"<< /Type /Page /Parent 2 0 R /MediaBox [0 0 300 300] /Contents 4 0 R /Resources << /Font << /F1 5 0 R >> >> >>",
        (
            f"<< /Length {len(('BT /F1 12 Tf 20 250 Td (' + texts[0] + ') Tj ET').encode())} >>\n"
            f"stream\nBT /F1 12 Tf 20 250 Td ({texts[0]}) Tj ET\nendstream"
        ).encode(),
        b"<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica >>",
    ]
    output = b"%PDF-1.4\n"
    offsets = [0]
    for index, obj in enumerate(objects, 1):
        offsets.append(len(output))
        output += f"{index} 0 obj\n".encode() + obj + b"\nendobj\n"
    startxref = len(output)
    output += f"xref\n0 {len(objects) + 1}\n0000000000 65535 f \n".encode()
    output += b"".join(f"{offset:010d} 00000 n \n".encode() for offset in offsets[1:])
    output += f"trailer\n<< /Size {len(objects) + 1} /Root 1 0 R >>\nstartxref\n{startxref}\n%%EOF\n".encode()
    return output


def test_pdf_to_text_writes_fake_law_text_and_report(tmp_path: Path) -> None:
    pdf_dir = tmp_path / "source_pdf"
    pdf_dir.mkdir()
    (pdf_dir / "FAKE.pdf").write_bytes(_fake_pdf(["FAKE LAW TEXT"]))
    output_dir = tmp_path / "source"
    report = tmp_path / "legal_pdf_report.md"
    results = convert_all(pdf_dir, output_dir, report)
    assert results[0]["pages"] == 1
    assert (output_dir / "FAKE.txt").read_text(encoding="utf-8").strip() == "FAKE LAW TEXT"
    assert "FAKE LAW TEXT" in report.read_text(encoding="utf-8")


def test_pdfplumber_recovers_spaces_from_positioned_fake_words(tmp_path: Path) -> None:
    pdf_dir = tmp_path / "source_pdf"
    pdf_dir.mkdir()
    content = b"BT /F1 12 Tf 20 250 Td (FAKE) Tj 35 0 Td (LAW) Tj 35 0 Td (TEXT) Tj ET"
    objects = [
        b"<< /Type /Catalog /Pages 2 0 R >>",
        b"<< /Type /Pages /Kids [3 0 R] /Count 1 >>",
        b"<< /Type /Page /Parent 2 0 R /MediaBox [0 0 300 300] /Contents 4 0 R /Resources << /Font << /F1 5 0 R >> >> >>",
        f"<< /Length {len(content)} >>\nstream\n".encode() + content + b"\nendstream",
        b"<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica >>",
    ]
    output = b"%PDF-1.4\n"
    offsets = [0]
    for index, obj in enumerate(objects, 1):
        offsets.append(len(output))
        output += f"{index} 0 obj\n".encode() + obj + b"\nendobj\n"
    startxref = len(output)
    output += f"xref\n0 {len(objects) + 1}\n0000000000 65535 f \n".encode()
    output += b"".join(f"{offset:010d} 00000 n \n".encode() for offset in offsets[1:])
    output += f"trailer\n<< /Size {len(objects) + 1} /Root 1 0 R >>\nstartxref\n{startxref}\n%%EOF\n".encode()
    (pdf_dir / "FAKE_POSITIONED.pdf").write_bytes(output)

    output_dir = tmp_path / "source"
    results = convert_all(pdf_dir, output_dir, tmp_path / "report.md")

    assert results[0]["engine"] == "pdfplumber"
    assert (output_dir / "FAKE_POSITIONED.txt").read_text(encoding="utf-8").strip() == (
        "FAKE LAW TEXT"
    )
