"""Evaluate one or more legal retrieval strategies."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import pandas as pd

from sakhi.src.retrieval import (
    CHUNKS_PATH,
    KEYWORDS_PATH,
    PROVISIONS_PATH,
    BM25Retriever,
    build_retriever,
)

PROJECT_ROOT = Path(__file__).resolve().parents[2]
TESTSET_PATH = PROJECT_ROOT / "sakhi" / "data" / "legal" / "retrieval_testset.csv"
RESULTS_DIR = PROJECT_ROOT / "sakhi" / "reports" / "results"
RETRIEVERS = (
    "bm25_text",
    "bm25_text_curated",
    "bm25_keywords",
    "routed",
    "embedding",
)


def _metrics_for_rows(testset: pd.DataFrame, retriever: Any) -> dict[str, float | int]:
    recalls = {1: [], 3: [], 5: []}
    reciprocal_ranks: list[float] = []
    for row in testset.itertuples():
        expected = set(str(row.expected_chunk_ids).split("|"))
        results = retriever.search(row.query, 5)
        ranked_ids = [result["chunk_id"] for result in results]
        positions = [
            ranked_ids.index(chunk_id) + 1
            for chunk_id in expected
            if chunk_id in ranked_ids
        ]
        reciprocal_ranks.append(1.0 / min(positions) if positions else 0.0)
        for cutoff in recalls:
            recalls[cutoff].append(bool(expected.intersection(ranked_ids[:cutoff])))
    count = len(testset)
    return {
        "queries": count,
        "Recall@1": sum(recalls[1]) / count if count else 0.0,
        "Recall@3": sum(recalls[3]) / count if count else 0.0,
        "Recall@5": sum(recalls[5]) / count if count else 0.0,
        "MRR": sum(reciprocal_ranks) / count if count else 0.0,
    }


def evaluate(testset: pd.DataFrame, retriever: Any) -> dict[str, object]:
    """Evaluate overall and per-language retrieval metrics."""
    required = {"query", "language", "expected_chunk_ids"}
    missing = required - set(testset.columns)
    if missing:
        raise ValueError(f"Retrieval test set is missing columns: {', '.join(sorted(missing))}")
    return {
        "result_label": "retrieval",
        "corpus": getattr(retriever, "corpus_name", "unknown"),
        "corpus_size": len(getattr(retriever, "chunks", [])),
        **_metrics_for_rows(testset, retriever),
        "by_language": {
            str(language): _metrics_for_rows(language_rows, retriever)
            for language, language_rows in testset.groupby("language", sort=True)
        },
    }


def fake_fixture() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    chunks = pd.DataFrame(
        [
            {
                "chunk_id": "FAKE-1",
                "act": "FAKE LAW TEXT",
                "section_no": "1",
                "section_title": "FAKE LAW TEXT",
                "text": "1. FAKE LAW TEXT about workplace complaints and reporting.",
                "language": "en",
                "source_file": "fake",
            },
            {
                "chunk_id": "FAKE-2",
                "act": "FAKE LAW TEXT",
                "section_no": "2",
                "section_title": "FAKE LAW TEXT",
                "text": "2. FAKE LAW TEXT about online abuse and reporting.",
                "language": "en",
                "source_file": "fake",
            },
        ]
    )
    keywords = pd.DataFrame(
        [
            {"chunk_id": "FAKE-1", "keywords": "workplace complaint"},
            {"chunk_id": "FAKE-2", "keywords": "online abuse"},
        ]
    )
    testset = pd.DataFrame(
        [{"query": "workplace complaints", "language": "en", "expected_chunk_ids": "FAKE-1"}]
    )
    return chunks, keywords, testset


def _comparison_rows(
    result: dict[str, object], retriever_name: str, testset_name: str
) -> list[dict[str, object]]:
    rows = []
    scopes = [("overall", result)]
    scopes.extend(
        (str(language), metrics)
        for language, metrics in result["by_language"].items()
    )
    for language, metrics in scopes:
        rows.append(
            {
                "retriever": retriever_name,
                "testset": testset_name,
                "language": language,
                "queries": metrics["queries"],
                "Recall@1": metrics["Recall@1"],
                "Recall@3": metrics["Recall@3"],
                "Recall@5": metrics["Recall@5"],
                "MRR": metrics["MRR"],
                "corpus": result["corpus"],
                "corpus_size": result["corpus_size"],
            }
        )
    return rows


def _write_comparison(rows: list[dict[str, object]], output_dir: Path) -> None:
    frame = pd.DataFrame(rows)
    csv_path = output_dir / "retrieval_comparison.csv"
    md_path = output_dir / "retrieval_comparison.md"
    frame.to_csv(csv_path, index=False)
    headers = list(frame.columns)
    markdown = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join("---" for _ in headers) + " |",
    ]
    markdown.extend(
        "| " + " | ".join(str(row[column]) for column in headers) + " |"
        for _, row in frame.iterrows()
    )
    md_path.write_text("\n".join(markdown) + "\n", encoding="utf-8")
    print(frame.to_string(index=False))
    print(f"Wrote {csv_path}")
    print(f"Wrote {md_path}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--testset", type=Path, action="append")
    parser.add_argument("--chunks", type=Path, default=CHUNKS_PATH)
    parser.add_argument("--keywords", type=Path, default=KEYWORDS_PATH)
    parser.add_argument("--provisions", type=Path, default=PROVISIONS_PATH)
    parser.add_argument("--output-dir", type=Path, default=RESULTS_DIR)
    parser.add_argument(
        "--retriever",
        choices=(*RETRIEVERS, "all"),
        default="bm25_text",
    )
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args()
    if args.smoke:
        chunks, keywords, testset = fake_fixture()
        keywords_path = args.output_dir / "_fake_keywords.csv"
        provisions_path = args.output_dir / "_fake_provisions.yaml"
        keywords.to_csv(keywords_path, index=False)
        provisions_path.write_text("fake: [FAKE-1, FAKE-2]\n", encoding="utf-8")
        retriever = build_retriever("bm25_keywords", chunks, keywords_path, provisions_path)
        result = evaluate(testset, retriever)
        result["result_label"] = "FAKE LAW TEXT, NOT A REAL RESULT (SANITY CHECK ONLY)"
        print(json.dumps(result, indent=2))
        return

    test_paths = args.testset or [TESTSET_PATH]
    chunks = pd.read_csv(args.chunks, encoding="utf-8")
    retriever_names = RETRIEVERS if args.retriever == "all" else (args.retriever,)
    rows: list[dict[str, object]] = []
    for test_path in test_paths:
        testset = pd.read_csv(test_path, encoding="utf-8")
        testset_name = test_path.stem
        for retriever_name in retriever_names:
            retriever = build_retriever(
                retriever_name, chunks, args.keywords, args.provisions
            )
            result = evaluate(testset, retriever)
            rows.extend(_comparison_rows(result, retriever_name, testset_name))
    args.output_dir.mkdir(parents=True, exist_ok=True)
    _write_comparison(rows, args.output_dir)


if __name__ == "__main__":
    main()
