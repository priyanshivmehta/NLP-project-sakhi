"""Tests for legal ingestion and retrieval using fake fixtures only."""

from pathlib import Path

import pandas as pd

from sakhi.src.eval_retrieval import evaluate
from sakhi.src.legal_ingest import ingest_sources
from sakhi.src.legal_ingest import _parse_file
from sakhi.src.legal_pdf_to_text import convert_all
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
