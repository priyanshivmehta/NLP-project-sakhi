"""Evaluate BM25 retrieval against a query-to-section test set."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from sakhi.src.retrieval import BM25Retriever, CHUNKS_PATH

PROJECT_ROOT = Path(__file__).resolve().parents[2]
TESTSET_PATH = PROJECT_ROOT / "sakhi" / "data" / "legal" / "retrieval_testset.csv"
RESULTS_DIR = PROJECT_ROOT / "sakhi" / "reports" / "results"


def evaluate(testset: pd.DataFrame, retriever: BM25Retriever) -> dict[str, float | int | str]:
    recalls = {1: [], 3: [], 5: []}
    reciprocal_ranks: list[float] = []
    for row in testset.itertuples():
        expected = set(str(row.expected_chunk_ids).split("|"))
        results = retriever.search(row.query, 5)
        ranked_ids = [result["chunk_id"] for result in results]
        positions = [ranked_ids.index(chunk_id) + 1 for chunk_id in expected if chunk_id in ranked_ids]
        reciprocal_ranks.append(1.0 / min(positions) if positions else 0.0)
        for cutoff in recalls:
            recalls[cutoff].append(bool(expected.intersection(ranked_ids[:cutoff])))
    count = len(testset)
    return {
        "result_label": "retrieval",
        "queries": count,
        "Recall@1": sum(recalls[1]) / count if count else 0.0,
        "Recall@3": sum(recalls[3]) / count if count else 0.0,
        "Recall@5": sum(recalls[5]) / count if count else 0.0,
        "MRR": sum(reciprocal_ranks) / count if count else 0.0,
    }


def fake_fixture() -> tuple[pd.DataFrame, pd.DataFrame]:
    chunks = pd.DataFrame(
        [
            {
                "chunk_id": "FAKE-1",
                "act": "FAKE LAW TEXT",
                "section_no": "1",
                "section_title": "FAKE LAW TEXT",
                "text": "FAKE LAW TEXT about workplace complaints.",
                "language": "en",
                "source_file": "fake",
            },
            {
                "chunk_id": "FAKE-2",
                "act": "FAKE LAW TEXT",
                "section_no": "2",
                "section_title": "FAKE LAW TEXT",
                "text": "FAKE LAW TEXT about online abuse.",
                "language": "en",
                "source_file": "fake",
            },
        ]
    )
    testset = pd.DataFrame(
        [{"query": "workplace complaints", "language": "en", "expected_chunk_ids": "FAKE-1"}]
    )
    return chunks, testset


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--testset", type=Path, default=TESTSET_PATH)
    parser.add_argument("--chunks", type=Path, default=CHUNKS_PATH)
    parser.add_argument("--output-dir", type=Path, default=RESULTS_DIR)
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args()
    if args.smoke:
        chunks, testset = fake_fixture()
        result = evaluate(testset, BM25Retriever(chunks))
        result["result_label"] = "FAKE LAW TEXT, NOT A REAL RESULT"
    else:
        testset = pd.read_csv(args.testset, encoding="utf-8")
        chunks = pd.read_csv(args.chunks, encoding="utf-8")
        result = evaluate(testset, BM25Retriever(chunks))
    args.output_dir.mkdir(parents=True, exist_ok=True)
    output = args.output_dir / ("retrieval_smoke_metrics.json" if args.smoke else "retrieval_metrics.json")
    output.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
