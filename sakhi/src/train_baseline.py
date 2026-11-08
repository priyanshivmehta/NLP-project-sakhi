"""Train train-only TF-IDF and logistic-regression baseline models."""

from __future__ import annotations

import argparse
import json
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
import yaml
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import f1_score
from sklearn.multiclass import OneVsRestClassifier
from sklearn.pipeline import FeatureUnion
from sklearn.linear_model import LogisticRegression

from sakhi.src.preprocess import preprocess_dataframe

LOGGER = logging.getLogger(__name__)
PROJECT_ROOT = Path(__file__).resolve().parents[2]
PROCESSED_DIR = PROJECT_ROOT / "sakhi" / "data" / "processed"
MODELS_DIR = PROJECT_ROOT / "sakhi" / "models"
LABELS_PATH = PROJECT_ROOT / "sakhi" / "config" / "labels.yaml"
RANDOM_SEED = 459
THRESHOLD_GRID = np.arange(0.10, 0.91, 0.05)


@dataclass
class BaselineArtifacts:
    """Fitted baseline components kept together for evaluation."""

    vectorizer: FeatureUnion
    urgency_model: LogisticRegression
    situation_model: OneVsRestClassifier
    situation_labels: list[str]
    thresholds: dict[str, float]


def load_labels(path: Path = LABELS_PATH) -> dict[str, list[str]]:
    with path.open(encoding="utf-8") as handle:
        config: Any = yaml.safe_load(handle)
    return config


def _model_text(data: pd.DataFrame) -> pd.Series:
    if "final_model_input" in data:
        return data["final_model_input"].fillna("").astype(str)
    if {"text", "language"}.issubset(data.columns):
        return preprocess_dataframe(data)["final_model_input"]
    raise ValueError("Training data must contain final_model_input or text and language.")


def fit_vectorizers(train_text: pd.Series) -> FeatureUnion:
    """Fit word and character TF-IDF features using train text only."""
    vectorizer = FeatureUnion(
        [
            (
                "word",
                TfidfVectorizer(ngram_range=(1, 2), analyzer="word", min_df=1),
            ),
            (
                "char",
                TfidfVectorizer(ngram_range=(2, 5), analyzer="char", min_df=1),
            ),
        ]
    )
    vectorizer.fit(train_text)
    return vectorizer


def _situation_targets(
    data: pd.DataFrame,
    labels: list[str],
) -> np.ndarray:
    label_index = {label: index for index, label in enumerate(labels)}
    targets = np.zeros((len(data), len(labels)), dtype=int)
    for row_index, value in enumerate(data["situation_labels"]):
        for label in str(value).split("|"):
            if label in label_index:
                targets[row_index, label_index[label]] = 1
    return targets


def tune_thresholds(
    model: OneVsRestClassifier,
    validation_features: Any,
    validation_targets: np.ndarray,
    labels: list[str],
) -> dict[str, float]:
    """Tune one probability threshold per label on validation data only."""
    probabilities = model.predict_proba(validation_features)
    thresholds: dict[str, float] = {}
    for label_index, label in enumerate(labels):
        best_threshold = 0.5
        best_score = -1.0
        for threshold in THRESHOLD_GRID:
            predictions = (probabilities[:, label_index] >= threshold).astype(int)
            score = f1_score(
                validation_targets[:, label_index],
                predictions,
                zero_division=0,
            )
            if score > best_score:
                best_score = score
                best_threshold = float(threshold)
        thresholds[label] = best_threshold
    return thresholds


def train_from_frames(
    train_data: pd.DataFrame,
    validation_data: pd.DataFrame,
    output_dir: Path = MODELS_DIR,
) -> BaselineArtifacts:
    """Fit all models using train features and validation thresholds only."""
    labels = load_labels()
    situation_labels = list(labels["situation_labels"])
    train_text = _model_text(train_data)
    validation_text = _model_text(validation_data)
    vectorizer = fit_vectorizers(train_text)
    train_features = vectorizer.transform(train_text)
    validation_features = vectorizer.transform(validation_text)

    urgency_model = LogisticRegression(
        class_weight="balanced",
        max_iter=1000,
        random_state=RANDOM_SEED,
    )
    urgency_model.fit(train_features, train_data["urgency"])

    situation_model = OneVsRestClassifier(
        LogisticRegression(
            class_weight="balanced",
            max_iter=1000,
            random_state=RANDOM_SEED,
        )
    )
    train_targets = _situation_targets(train_data, situation_labels)
    validation_targets = _situation_targets(validation_data, situation_labels)
    situation_model.fit(train_features, train_targets)
    thresholds = tune_thresholds(
        situation_model,
        validation_features,
        validation_targets,
        situation_labels,
    )

    output_dir.mkdir(parents=True, exist_ok=True)
    joblib.dump(vectorizer, output_dir / "tfidf_features.joblib")
    joblib.dump(urgency_model, output_dir / "urgency_model.joblib")
    joblib.dump(situation_model, output_dir / "situation_model.joblib")
    with (output_dir / "situation_thresholds.json").open("w", encoding="utf-8") as handle:
        json.dump(thresholds, handle, indent=2)
    with (output_dir / "model_metadata.json").open("w", encoding="utf-8") as handle:
        json.dump(
            {
                "training_row_count": int(len(train_data)),
                "urgency_labels": sorted(train_data["urgency"].astype(str).unique()),
                "situation_labels": situation_labels,
            },
            handle,
            indent=2,
        )
    print(f"Training file: in-memory frame ({len(train_data)} rows)")
    print(f"Validation file: in-memory frame ({len(validation_data)} rows)")
    print(f"Model directory: {output_dir}")
    print(
        "Model metadata: "
        f"training_row_count={len(train_data)}, "
        f"urgency_labels={sorted(train_data['urgency'].astype(str).unique())}, "
        f"situation_labels={situation_labels}"
    )
    return BaselineArtifacts(
        vectorizer=vectorizer,
        urgency_model=urgency_model,
        situation_model=situation_model,
        situation_labels=situation_labels,
        thresholds=thresholds,
    )


def train_from_paths(
    train_path: Path = PROCESSED_DIR / "train.csv",
    validation_path: Path = PROCESSED_DIR / "val.csv",
    output_dir: Path = MODELS_DIR,
) -> BaselineArtifacts:
    """Load only train and validation CSVs and fit the baseline."""
    train_data = pd.read_csv(train_path, encoding="utf-8")
    validation_data = pd.read_csv(validation_path, encoding="utf-8")
    print(f"Training file: {train_path} ({len(train_data)} rows)")
    print(f"Validation file: {validation_path} ({len(validation_data)} rows)")
    print(f"Model directory: {output_dir}")
    return train_from_frames(train_data, validation_data, output_dir)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train", type=Path, default=PROCESSED_DIR / "train.csv")
    parser.add_argument("--val", type=Path, default=PROCESSED_DIR / "val.csv")
    parser.add_argument("--output-dir", type=Path, default=MODELS_DIR)
    args = parser.parse_args()
    train_from_paths(args.train, args.val, args.output_dir)
    LOGGER.info("Saved baseline models to %s", args.output_dir)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    main()
