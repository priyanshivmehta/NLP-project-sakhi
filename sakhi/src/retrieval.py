"""Retrieval over user-supplied legal chunks and curated provision aids."""

from __future__ import annotations

import argparse
import json
import logging
import re
from functools import lru_cache
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import yaml
from rank_bm25 import BM25Okapi

LOGGER = logging.getLogger(__name__)
PROJECT_ROOT = Path(__file__).resolve().parents[2]
CHUNKS_PATH = PROJECT_ROOT / "sakhi" / "data" / "legal" / "chunks" / "chunks.csv"
KEYWORDS_PATH = PROJECT_ROOT / "sakhi" / "data" / "legal" / "provision_keywords.csv"
PROVISIONS_PATH = PROJECT_ROOT / "sakhi" / "config" / "situations_to_provisions.yaml"
EMBEDDING_CACHE_PATH = PROJECT_ROOT / "sakhi" / "models_embedding" / "legal_chunk_embeddings.npy"
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
MIN_TEXT_LENGTH = 40
MAX_TEXT_LENGTH = 6000


def tokenize(text: str) -> list[str]:
    """Keep Roman and Devanagari word/number tokens."""
    return TOKEN_RE.findall(str(text).casefold())


def _result(row: pd.Series, score: float) -> dict[str, Any]:
    return {
        "chunk_id": row["chunk_id"],
        "act": row["act"],
        "section_no": row["section_no"],
        "text": row["text"],
        "score": float(score),
    }


class BM25Retriever:
    """BM25 over the full corpus after the documented length filter."""

    corpus_name = "bm25_text"

    def __init__(self, chunks: pd.DataFrame, filter_lengths: bool = True):
        missing = REQUIRED_COLUMNS - set(chunks.columns)
        if missing:
            raise ValueError(f"Chunks are missing columns: {', '.join(sorted(missing))}")
        frame = chunks.fillna("").reset_index(drop=True)
        included = (
            frame["text"].astype(str).str.len().between(MIN_TEXT_LENGTH, MAX_TEXT_LENGTH)
            if filter_lengths
            else pd.Series(True, index=frame.index)
        )
        excluded = frame.loc[~included, "chunk_id"].tolist()
        for chunk_id in excluded:
            LOGGER.info("%s: excluded from bm25_text by text length", chunk_id)
        self.chunks = frame.loc[included].reset_index(drop=True)
        self.excluded_ids = excluded
        self.index = (
            BM25Okapi([tokenize(text) for text in self.chunks["text"]])
            if not self.chunks.empty
            else None
        )

    def search(self, query: str, k: int = 5) -> list[dict[str, Any]]:
        if k <= 0 or self.index is None:
            return []
        scores = self.index.get_scores(tokenize(query))
        ranked = sorted(range(len(scores)), key=lambda index: scores[index], reverse=True)
        return [_result(self.chunks.iloc[index], scores[index]) for index in ranked[:k]]


class BM25KeywordRetriever(BM25Retriever):
    """BM25 over curated chunk text plus search-only keyword aids."""

    corpus_name = "bm25_keywords"

    def __init__(
        self,
        chunks: pd.DataFrame,
        keywords_path: Path = KEYWORDS_PATH,
        provisions_path: Path = PROVISIONS_PATH,
    ):
        curated = load_curated_chunks(chunks, keywords_path, provisions_path)
        super().__init__(curated, filter_lengths=False)
        keywords = load_keywords(keywords_path)
        self.index = (
            BM25Okapi(
                [
                    tokenize(f"{row.text} {keywords.get(row.chunk_id, '')}")
                    for row in self.chunks.itertuples()
                ]
            )
            if not self.chunks.empty
            else None
        )


class BM25TextCuratedRetriever(BM25Retriever):
    """Plain BM25 over the curated chunk set without keyword aids."""

    corpus_name = "bm25_text_curated"

    def __init__(
        self,
        chunks: pd.DataFrame,
        keywords_path: Path = KEYWORDS_PATH,
        provisions_path: Path = PROVISIONS_PATH,
    ):
        curated = load_curated_chunks(chunks, keywords_path, provisions_path)
        super().__init__(curated, filter_lengths=False)


def load_keywords(path: Path = KEYWORDS_PATH) -> dict[str, str]:
    frame = pd.read_csv(path, dtype=str, keep_default_na=False)
    required = {"chunk_id", "keywords"}
    missing = required - set(frame.columns)
    if missing:
        raise ValueError(f"Keyword file is missing columns: {', '.join(sorted(missing))}")
    return dict(zip(frame["chunk_id"], frame["keywords"]))


def load_provision_map(path: Path = PROVISIONS_PATH) -> dict[str, list[str]]:
    config = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    if not isinstance(config, dict):
        raise ValueError("Situation provision map must be a YAML mapping.")
    return {str(label): [str(item) for item in values] for label, values in config.items()}


def load_curated_chunks(
    chunks: pd.DataFrame,
    keywords_path: Path = KEYWORDS_PATH,
    provisions_path: Path = PROVISIONS_PATH,
) -> pd.DataFrame:
    ids = set(chunks["chunk_id"].astype(str))
    keywords = load_keywords(keywords_path)
    provision_map = load_provision_map(provisions_path)
    configured = list(keywords)
    configured.extend(chunk_id for values in provision_map.values() for chunk_id in values)
    unknown = sorted(set(configured) - ids)
    if unknown:
        print("Unknown provision chunk IDs:")
        print(", ".join(unknown))
        raise ValueError(f"Provision configuration references unknown chunk IDs: {unknown}")
    return chunks.loc[chunks["chunk_id"].isin(keywords)].copy()


class EmbeddingRetriever:
    corpus_name = "embedding"

    def __init__(
        self,
        chunks: pd.DataFrame,
        keywords_path: Path = KEYWORDS_PATH,
        cache_path: Path = EMBEDDING_CACHE_PATH,
        encoder: Any | None = None,
    ):
        self.chunks = load_curated_chunks(chunks, keywords_path)
        keywords = load_keywords(keywords_path)
        corpus_text = [
            f"{row.text} {keywords.get(row.chunk_id, '')}"
            for row in self.chunks.itertuples()
        ]
        if cache_path.exists():
            embeddings = np.load(cache_path)
            if len(embeddings) != len(corpus_text):
                raise ValueError(f"Embedding cache row count does not match curated corpus: {cache_path}")
        else:
            if encoder is None:
                try:
                    from sentence_transformers import SentenceTransformer
                except ImportError as exc:
                    raise RuntimeError(
                        "sentence-transformers is required for embedding retrieval."
                    ) from exc
                encoder = SentenceTransformer("paraphrase-multilingual-MiniLM-L12-v2")
            embeddings = np.asarray(encoder.encode(corpus_text, show_progress_bar=False))
            cache_path.parent.mkdir(parents=True, exist_ok=True)
            np.save(cache_path, embeddings)
        norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
        self.embeddings = embeddings / np.maximum(norms, 1e-12)
        self.encoder = encoder

    def search(self, query: str, k: int = 5) -> list[dict[str, Any]]:
        if k <= 0 or self.chunks.empty:
            return []
        if self.encoder is None:
            try:
                from sentence_transformers import SentenceTransformer
            except ImportError as exc:
                raise RuntimeError("sentence-transformers is required for embedding retrieval.") from exc
            self.encoder = SentenceTransformer("paraphrase-multilingual-MiniLM-L12-v2")
        query_embedding = np.asarray(self.encoder.encode([query], show_progress_bar=False))[0]
        query_embedding = query_embedding / max(float(np.linalg.norm(query_embedding)), 1e-12)
        scores = self.embeddings @ query_embedding
        ranked = np.argsort(-scores)[:k]
        return [_result(self.chunks.iloc[index], scores[index]) for index in ranked]


class RoutedRetriever:
    corpus_name = "routed"

    def __init__(
        self,
        chunks: pd.DataFrame,
        keywords_path: Path = KEYWORDS_PATH,
        provisions_path: Path = PROVISIONS_PATH,
    ):
        self.keyword_retriever = BM25KeywordRetriever(
            chunks, keywords_path, provisions_path
        )
        self.chunks = self.keyword_retriever.chunks
        self.keywords = load_keywords(keywords_path)
        self.provision_map = load_provision_map(provisions_path)

    def search(self, query: str, k: int = 5) -> list[dict[str, Any]]:
        from sakhi.src.pipeline import detect_language, load_artifacts
        from sakhi.src.preprocess import preprocess_text, _load_spelling_variants

        artifacts = load_artifacts()
        language = detect_language(query)
        model_input = preprocess_text(
            query, language, _load_spelling_variants()
        )["final_model_input"]
        features = artifacts.vectorizer.transform([model_input])
        probabilities = artifacts.situation_model.predict_proba(features)[0]
        labels = [
            label
            for index, label in enumerate(artifacts.situation_labels)
            if probabilities[index] >= artifacts.thresholds[label]
        ]
        candidate_ids = {
            chunk_id for label in labels for chunk_id in self.provision_map.get(label, [])
        }
        if candidate_ids:
            candidate = self.keyword_retriever.chunks[
                self.keyword_retriever.chunks["chunk_id"].isin(candidate_ids)
            ]
            retriever = BM25KeywordRetriever.__new__(BM25KeywordRetriever)
            retriever.chunks = candidate.reset_index(drop=True)
            retriever.index = BM25Okapi(
                [
                    tokenize(f"{row.text} {self.keywords.get(row.chunk_id, '')}")
                    for row in retriever.chunks.itertuples()
                ]
            )
            return retriever.search(query, k)
        return self.keyword_retriever.search(query, k)


def build_retriever(
    retriever: str,
    chunks: pd.DataFrame,
    keywords_path: Path = KEYWORDS_PATH,
    provisions_path: Path = PROVISIONS_PATH,
) -> Any:
    if retriever == "bm25_text":
        return BM25Retriever(chunks)
    if retriever == "bm25_keywords":
        return BM25KeywordRetriever(chunks, keywords_path, provisions_path)
    if retriever == "bm25_text_curated":
        return BM25TextCuratedRetriever(chunks, keywords_path, provisions_path)
    if retriever == "routed":
        return RoutedRetriever(chunks, keywords_path, provisions_path)
    if retriever == "embedding":
        return EmbeddingRetriever(chunks, keywords_path)
    raise ValueError(f"Unknown retriever: {retriever}")


@lru_cache(maxsize=8)
def load_retriever(
    chunks_path: str = str(CHUNKS_PATH), retriever: str = "bm25_text"
) -> Any:
    path = Path(chunks_path)
    chunks = (
        pd.read_csv(path, encoding="utf-8")
        if path.is_file()
        else pd.DataFrame(columns=sorted(REQUIRED_COLUMNS))
    )
    return build_retriever(retriever, chunks)


def search(
    query: str,
    k: int = 5,
    chunks_path: str | Path = CHUNKS_PATH,
    retriever: str = "bm25_text",
) -> list[dict[str, Any]]:
    """Search legal chunks with the selected retrieval strategy."""
    return load_retriever(str(chunks_path), retriever).search(query, k)


def retrieve(
    text: str, k: int = 3, retriever: str = "bm25_text"
) -> list[dict[str, Any]]:
    """Pipeline-compatible retrieval wrapper."""
    return search(text, k, retriever=retriever)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("query")
    parser.add_argument("-k", type=int, default=5)
    parser.add_argument("--chunks", type=Path, default=CHUNKS_PATH)
    parser.add_argument(
        "--retriever",
        choices=("bm25_text", "bm25_keywords", "routed", "embedding"),
        default="bm25_text",
    )
    args = parser.parse_args()
    for result in search(args.query, args.k, args.chunks, args.retriever):
        print(json.dumps(result, ensure_ascii=False))


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    main()
