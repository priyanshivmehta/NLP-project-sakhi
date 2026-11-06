"""Evaluate baseline models without reading test data unless explicitly requested."""

from __future__ import annotations

import argparse
import json
import logging
import tempfile
from pathlib import Path
from typing import Any

import joblib
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import yaml
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_recall_fscore_support,
)

from sakhi.src.preprocess import preprocess_dataframe
from sakhi.src.train_baseline import (
    BaselineArtifacts,
    MODELS_DIR,
    PROCESSED_DIR,
    PROJECT_ROOT,
    _model_text,
    _situation_targets,
    train_from_frames,
)

LOGGER = logging.getLogger(__name__)
RESULTS_DIR = PROJECT_ROOT / "sakhi" / "reports" / "results"
LABELS_PATH = PROJECT_ROOT / "sakhi" / "config" / "labels.yaml"
VERIFIED_SYNTHETIC_LABEL = (
    "SYNTHETIC BENCHMARK (AI-generated, human-verified), NOT REAL-WORLD ACCURACY"
)


def load_labels() -> dict[str, list[str]]:
    with LABELS_PATH.open(encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def _predict(artifacts: BaselineArtifacts, data: pd.DataFrame) -> tuple[Any, Any]:
    features = artifacts.vectorizer.transform(_model_text(data))
    urgency = artifacts.urgency_model.predict(features)
    probabilities = artifacts.situation_model.predict_proba(features)
    situation = pd.DataFrame(
        {
            label: probabilities[:, index] >= artifacts.thresholds[label]
            for index, label in enumerate(artifacts.situation_labels)
        }
    ).to_numpy(dtype=int)
    return urgency, situation


def evaluate_frames(
    artifacts: BaselineArtifacts,
    data: pd.DataFrame,
    output_dir: Path,
    result_label: str = "validation",
) -> dict[str, Any]:
    """Calculate and save metrics, supports, and the urgency confusion matrix."""
    output_dir.mkdir(parents=True, exist_ok=True)
    labels = load_labels()
    urgency_labels = list(labels["urgency_labels"])
    situation_labels = artifacts.situation_labels
    urgency_predictions, situation_predictions = _predict(artifacts, data)
    urgency_true = data["urgency"].to_numpy()
    situation_true = _situation_targets(data, situation_labels)
    verified_synthetic = (
        not data.empty
        and set(data["source"]) == {"ai_generated"}
        and "human_verified" in data
        and data["human_verified"].astype(bool).all()
    )
    quality = (
        VERIFIED_SYNTHETIC_LABEL
        if verified_synthetic
        else (
            "SYNTHETIC, NOT A REAL RESULT"
            if not data.empty and set(data["source"]) == {"ai_generated"}
            else "HUMAN_OR_MIXED"
        )
    )

    urgency_precision, urgency_recall, urgency_f1, urgency_support = (
        precision_recall_fscore_support(
            urgency_true,
            urgency_predictions,
            labels=urgency_labels,
            zero_division=0,
        )
    )
    rows: list[dict[str, Any]] = []
    for label, precision, recall, f1, support in zip(
        urgency_labels,
        urgency_precision,
        urgency_recall,
        urgency_f1,
        urgency_support,
    ):
        rows.append(
            {
                "task": "urgency",
                "metric_scope": "per_class",
                "label": label,
                "precision": precision,
                "recall": recall,
                "f1": f1,
                "support": int(support),
                "data_quality": quality,
            }
        )
    rows.append(
        {
            "task": "urgency",
            "metric_scope": "macro",
            "label": "macro_avg",
            "f1": f1_score(urgency_true, urgency_predictions, labels=urgency_labels, average="macro", zero_division=0),
            "support": int(len(data)),
            "data_quality": quality,
        }
    )

    situation_precision, situation_recall, situation_f1, situation_support = (
        precision_recall_fscore_support(
            situation_true,
            situation_predictions,
            labels=list(range(len(situation_labels))),
            zero_division=0,
        )
    )
    for label, precision, recall, f1, support in zip(
        situation_labels,
        situation_precision,
        situation_recall,
        situation_f1,
        situation_support,
    ):
        rows.append(
            {
                "task": "situation",
                "metric_scope": "per_label",
                "label": label,
                "precision": precision,
                "recall": recall,
                "f1": f1,
                "support": int(support),
                "data_quality": quality,
            }
        )
    rows.extend(
        [
            {
                "task": "situation",
                "metric_scope": "macro",
                "label": "macro_avg",
                "f1": f1_score(situation_true, situation_predictions, average="macro", zero_division=0),
                "support": int(situation_true.sum()),
                "data_quality": quality,
            },
            {
                "task": "situation",
                "metric_scope": "micro",
                "label": "micro_avg",
                "f1": f1_score(situation_true, situation_predictions, average="micro", zero_division=0),
                "support": int(situation_true.sum()),
                "data_quality": quality,
            },
            {
                "task": "situation",
                "metric_scope": "subset_accuracy",
                "label": "subset_accuracy",
                "f1": accuracy_score(situation_true, situation_predictions),
                "support": int(len(data)),
                "data_quality": quality,
            },
        ]
    )
    for language, language_data in data.groupby("language", sort=True):
        language_urgency, language_situation = _predict(artifacts, language_data)
        language_urgency_true = language_data["urgency"].to_numpy()
        language_situation_true = _situation_targets(language_data, situation_labels)
        language_quality = (
            VERIFIED_SYNTHETIC_LABEL
            if (
                set(language_data["source"]) == {"ai_generated"}
                and "human_verified" in language_data
                and language_data["human_verified"].astype(bool).all()
            )
            else (
                "SYNTHETIC, NOT A REAL RESULT"
                if set(language_data["source"]) == {"ai_generated"}
                else "HUMAN_OR_MIXED"
            )
        )
        rows.append(
            {
                "task": "urgency",
                "metric_scope": "per_language",
                "label": language,
                "f1": f1_score(
                    language_urgency_true,
                    language_urgency,
                    labels=urgency_labels,
                    average="macro",
                    zero_division=0,
                ),
                "support": int(len(language_data)),
                "data_quality": language_quality,
            }
        )
        for index, label in enumerate(situation_labels):
            rows.append(
                {
                    "task": "situation",
                    "metric_scope": "per_language_label",
                    "label": f"{language}:{label}",
                    "f1": f1_score(
                        language_situation_true[:, index],
                        language_situation[:, index],
                        zero_division=0,
                    ),
                    "support": int(language_situation_true[:, index].sum()),
                    "data_quality": language_quality,
                }
            )
    if "noisy" in data:
        for noisy, subset in data.groupby(data["noisy"].astype(bool), sort=True):
            subset_urgency, subset_situation = _predict(artifacts, subset)
            subset_true_urgency = subset["urgency"].to_numpy()
            subset_true_situation = _situation_targets(subset, situation_labels)
            rows.extend(
                [
                    {
                        "task": "combined",
                        "metric_scope": "noisy" if noisy else "clean",
                        "label": "urgency_macro_f1",
                        "f1": f1_score(
                            subset_true_urgency,
                            subset_urgency,
                            labels=urgency_labels,
                            average="macro",
                            zero_division=0,
                        ),
                        "support": int(len(subset)),
                        "data_quality": quality,
                    },
                    {
                        "task": "combined",
                        "metric_scope": "noisy" if noisy else "clean",
                        "label": "situation_macro_f1",
                        "f1": f1_score(
                            subset_true_situation,
                            subset_situation,
                            average="macro",
                            zero_division=0,
                        ),
                        "support": int(len(subset)),
                        "data_quality": quality,
                    },
                ]
            )
    metrics = pd.DataFrame(rows)
    metrics.to_csv(output_dir / f"{result_label}_metrics.csv", index=False)
    confusion = confusion_matrix(urgency_true, urgency_predictions, labels=urgency_labels)
    plt.figure(figsize=(8, 6))
    sns.heatmap(confusion, annot=True, fmt="d", cmap="Blues", xticklabels=urgency_labels, yticklabels=urgency_labels)
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.title(f"{result_label.title()} urgency confusion matrix")
    plt.tight_layout()
    plt.savefig(output_dir / f"{result_label}_urgency_confusion_matrix.png", dpi=150)
    plt.close()
    summary = {
        "result_label": result_label,
        "metrics_csv": str(output_dir / f"{result_label}_metrics.csv"),
        "rows": rows,
    }
    with (output_dir / f"{result_label}_metrics.json").open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2, default=float)
    return summary


def _smoke_frames() -> tuple[pd.DataFrame, pd.DataFrame]:
    sample_path = PROJECT_ROOT / "sakhi" / "data" / "raw" / "sample_fictional.csv"
    sample = preprocess_dataframe(pd.read_csv(sample_path, encoding="utf-8"))
    expanded = pd.concat([sample] * 12, ignore_index=True)
    expanded["id"] = [f"smoke-{index}" for index in range(len(expanded))]
    expanded["split"] = ["train" if index % 3 else "val" for index in range(len(expanded))]
    return expanded.loc[expanded["split"] == "train"], expanded.loc[expanded["split"] == "val"]


def run_smoke(output_dir: Path) -> dict[str, Any]:
    """Run a labelled smoke-only train/evaluation in a temporary model directory."""
    train_data, validation_data = _smoke_frames()
    with tempfile.TemporaryDirectory(prefix="sakhi_smoke_") as model_dir:
        artifacts = train_from_frames(train_data, validation_data, Path(model_dir))
        return evaluate_frames(
            artifacts,
            validation_data,
            output_dir,
            result_label="SMOKE_TEST_NOT_A_REAL_RESULT",
        )


def load_evaluation_data(data_path: Path, final_test: bool = False) -> pd.DataFrame:
    """Load validation data, or test data only with an explicit final flag."""
    if data_path.name == "test.csv" and not final_test:
        raise ValueError("test.csv requires the explicit --final-test flag.")
    return pd.read_csv(data_path, encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path)
    parser.add_argument("--models-dir", type=Path, default=MODELS_DIR)
    parser.add_argument("--output-dir", type=Path, default=RESULTS_DIR)
    parser.add_argument("--final-test", action="store_true")
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args()
    if args.smoke:
        run_smoke(args.output_dir)
        print("SMOKE TEST, NOT A REAL RESULT")
        return
    split_name = "test" if args.final_test else "val"
    data_path = args.data or PROCESSED_DIR / f"{split_name}.csv"
    if args.final_test:
        print("WARNING: final test evaluation should be run once only.")
    data = load_evaluation_data(data_path, final_test=args.final_test)
    artifacts = BaselineArtifacts(
        vectorizer=joblib.load(args.models_dir / "tfidf_features.joblib"),
        urgency_model=joblib.load(args.models_dir / "urgency_model.joblib"),
        situation_model=joblib.load(args.models_dir / "situation_model.joblib"),
        situation_labels=json.loads((args.models_dir / "model_metadata.json").read_text())["situation_labels"],
        thresholds=json.loads((args.models_dir / "situation_thresholds.json").read_text()),
    )
    evaluate_frames(artifacts, data, args.output_dir, split_name)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    main()
