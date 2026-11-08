"""Train a multilingual sentence-embedding baseline."""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.multiclass import OneVsRestClassifier

from sakhi.src.preprocess import preprocess_dataframe
from sakhi.src.train_baseline import (
    PROCESSED_DIR,
    PROJECT_ROOT,
    RANDOM_SEED,
    _model_text,
    _situation_targets,
    tune_thresholds,
    load_labels,
)

DEFAULT_MODEL_NAME = "paraphrase-multilingual-MiniLM-L12-v2"
MODELS_DIR = PROJECT_ROOT / "sakhi" / "models_embedding"
CACHE_DIR = MODELS_DIR / "embedding_cache"


@dataclass
class EmbeddingArtifacts:
    encoder: Any
    urgency_model: LogisticRegression
    situation_model: OneVsRestClassifier
    situation_labels: list[str]
    thresholds: dict[str, float]


def _load_encoder(model_name: str) -> Any:
    try:
        from sentence_transformers import SentenceTransformer
    except ImportError as exc:
        raise RuntimeError(
            "sentence-transformers is required for embedding training/evaluation. "
            "Install requirements-train.txt."
        ) from exc
    return SentenceTransformer(model_name)


def _encode(
    encoder: Any,
    texts: pd.Series,
    cache_path: Path | None = None,
) -> np.ndarray:
    if cache_path is not None and cache_path.exists():
        return np.load(cache_path)
    embeddings = np.asarray(encoder.encode(texts.tolist(), show_progress_bar=False))
    if cache_path is not None:
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        np.save(cache_path, embeddings)
    return embeddings


def train_embedding_from_frames(
    train_data: pd.DataFrame,
    validation_data: pd.DataFrame,
    output_dir: Path = MODELS_DIR,
    model_name: str = DEFAULT_MODEL_NAME,
    encoder: Any | None = None,
) -> EmbeddingArtifacts:
    """Train classifiers on train embeddings and tune thresholds on validation only."""
    labels = load_labels()
    situation_labels = list(labels["situation_labels"])
    encoder = encoder or _load_encoder(model_name)
    cache_dir = output_dir / "embedding_cache"
    train_embeddings = _encode(encoder, _model_text(train_data), cache_dir / "train.npy")
    validation_embeddings = _encode(encoder, _model_text(validation_data), cache_dir / "val.npy")

    urgency_model = LogisticRegression(
        class_weight="balanced", max_iter=1000, random_state=RANDOM_SEED
    )
    urgency_model.fit(train_embeddings, train_data["urgency"])
    situation_model = OneVsRestClassifier(
        LogisticRegression(class_weight="balanced", max_iter=1000, random_state=RANDOM_SEED)
    )
    situation_model.fit(
        train_embeddings,
        _situation_targets(train_data, situation_labels),
    )
    thresholds = tune_thresholds(
        situation_model,
        validation_embeddings,
        _situation_targets(validation_data, situation_labels),
        situation_labels,
    )

    output_dir.mkdir(parents=True, exist_ok=True)
    joblib.dump(urgency_model, output_dir / "urgency_model.joblib")
    joblib.dump(situation_model, output_dir / "situation_model.joblib")
    (output_dir / "situation_thresholds.json").write_text(
        json.dumps(thresholds, indent=2), encoding="utf-8"
    )
    (output_dir / "model_metadata.json").write_text(
        json.dumps(
            {
                "model_family": "embedding",
                "model_name": model_name,
                "training_row_count": int(len(train_data)),
                "urgency_labels": sorted(train_data["urgency"].astype(str).unique()),
                "situation_labels": situation_labels,
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    print(f"Training file: in-memory frame ({len(train_data)} rows)")
    print(f"Validation file: in-memory frame ({len(validation_data)} rows)")
    print(f"Model directory: {output_dir}")
    print(f"Embedding model: {model_name}")
    return EmbeddingArtifacts(
        encoder=encoder,
        urgency_model=urgency_model,
        situation_model=situation_model,
        situation_labels=situation_labels,
        thresholds=thresholds,
    )


def train_embedding_from_paths(
    train_path: Path = PROCESSED_DIR / "train.csv",
    validation_path: Path = PROCESSED_DIR / "val.csv",
    output_dir: Path = MODELS_DIR,
    model_name: str = DEFAULT_MODEL_NAME,
) -> EmbeddingArtifacts:
    train_data = pd.read_csv(train_path, encoding="utf-8")
    validation_data = pd.read_csv(validation_path, encoding="utf-8")
    print(f"Training file: {train_path} ({len(train_data)} rows)")
    print(f"Validation file: {validation_path} ({len(validation_data)} rows)")
    return train_embedding_from_frames(
        train_data, validation_data, output_dir, model_name
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train", type=Path, default=PROCESSED_DIR / "train.csv")
    parser.add_argument("--val", type=Path, default=PROCESSED_DIR / "val.csv")
    parser.add_argument("--output-dir", type=Path, default=MODELS_DIR)
    parser.add_argument("--model-name", default=DEFAULT_MODEL_NAME)
    args = parser.parse_args()
    train_embedding_from_paths(args.train, args.val, args.output_dir, args.model_name)


if __name__ == "__main__":
    main()
