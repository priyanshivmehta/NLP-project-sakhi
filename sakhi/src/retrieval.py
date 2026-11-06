"""BM25 retrieval over ingested, user-supplied legal chunks."""

from __future__ import annotations

import argparse
import re
from functools import lru_cache
from pathlib import Path
from typing import Any

import pandas as pd
from rank_bm25 import BM25Okapi

PROJECT_ROOT = Path(__file__).resolve().parents[2]
CHUNKS_PATH = PROJECT_ROOT / "sakhi" / "data" / "legal" / "chunks" / "chunks.csv"
TOKEN_RE = re.compile(r"[A-Za-z0-9_\u0900-\u097F]+")
REQUIRED_COLUMNS = {
    "chunk_id",
    "act",
    "section_no",
    "section_title",
    "text",
    "language",
    "source_file",
}


def tokenize(text: str) -> list[str]:
    """Keep Roman and Devanagari word/number tokens."""
    return TOKEN_RE.findall(str(text).casefold())


class BM25Retriever:
    def __init__(self, chunks: pd.DataFrame):
        missing = REQUIRED_COLUMNS - set(chunks.columns)
        if missing:
            raise ValueError(f"Chunks are missing columns: {', '.join(sorted(missing))}")
        self.chunks = chunks.fillna("").reset_index(drop=True)
        if self.chunks.empty:
            self.index = None
            return
        corpus = [
            tokenize(f"{row.section_title} {row.text}")
            for row in self.chunks.itertuples()
        ]
        self.index = BM25Okapi(corpus)

    def search(self, query: str, k: int = 5) -> list[dict[str, Any]]:
        if k <= 0 or self.index is None:
            return []
        scores = self.index.get_scores(tokenize(query))
        ranked = sorted(range(len(scores)), key=lambda index: scores[index], reverse=True)
        return [
            {
                "chunk_id": self.chunks.iloc[index]["chunk_id"],
                "act": self.chunks.iloc[index]["act"],
                "section_no": self.chunks.iloc[index]["section_no"],
                "text": self.chunks.iloc[index]["text"],
                "score": float(scores[index]),
            }
            for index in ranked[:k]
        ]


@lru_cache(maxsize=4)
def load_retriever(chunks_path: str = str(CHUNKS_PATH)) -> BM25Retriever:
    path = Path(chunks_path)
    if not path.is_file():
        return BM25Retriever(pd.DataFrame(columns=sorted(REQUIRED_COLUMNS)))
    return BM25Retriever(pd.read_csv(path, encoding="utf-8"))


def search(query: str, k: int = 5, chunks_path: str | Path = CHUNKS_PATH) -> list[dict[str, Any]]:
    """Search the configured legal chunk CSV."""
    return load_retriever(str(chunks_path)).search(query, k)


def retrieve(text: str, k: int = 3) -> list[dict[str, Any]]:
    """Pipeline-compatible retrieval wrapper."""
    return search(text, k)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("query")
    parser.add_argument("-k", type=int, default=5)
    parser.add_argument("--chunks", type=Path, default=CHUNKS_PATH)
    args = parser.parse_args()
    for result in search(args.query, args.k, args.chunks):
        print(result)


if __name__ == "__main__":
    main()
