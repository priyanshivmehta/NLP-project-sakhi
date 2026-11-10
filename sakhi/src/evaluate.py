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
from sakhi.src.train_embedding import MODELS_DIR as EMBEDDING_MODELS_DIR
from sakhi.src.train_embedding import _encode

LOGGER = logging.getLogger(__name__)
RESULTS_DIR = PROJECT_ROOT / "sakhi" / "reports" / "results"
EMBEDDING_RESULTS_DIR = PROJECT_ROOT / "sakhi" / "reports" / "results_embedding"
ERRORS_DIR = PROJECT_ROOT / "sakhi" / "reports" / "results_v1"
LABELS_PATH = PROJECT_ROOT / "sakhi" / "config" / "labels.yaml"
VERIFIED_SYNTHETIC_LABEL = (
    "SYNTHETIC BENCHMARK (AI-generated, human-verified), NOT REAL-WORLD ACCURACY"
)


def load_labels() -> dict[str, list[str]]:
    with LABELS_PATH.open(encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def _predict(artifacts: BaselineArtifacts, data: pd.DataFrame) -> tuple[Any, Any]:
    if hasattr(artifacts, "encoder"):
        features = _encode(artifacts.encoder, _model_text(data))
    else:
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
    error_dump: Path | None = None,
) -> dict[str, Any]:
    """Calculate and save metrics, supports, and the urgency confusion matrix."""
    output_dir.mkdir(parents=True, exist_ok=True)
    labels = load_labels()
    urgency_labels = list(labels["urgency_labels"])
    situation_labels = artifacts.situation_labels
    urgency_predictions, situation_predictions = _predict(artifacts, data)
    evaluated_row_count = len(urgency_predictions)
    if evaluated_row_count != len(data) or len(situation_predictions) != len(data):
        raise ValueError(
            "Evaluated row count does not match the number of rows in the evaluation file: "
            f"file={len(data)}, urgency={len(urgency_predictions)}, "
            f"situation={len(situation_predictions)}."
        )
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
    if error_dump is not None:
        error_rows = data.copy()
        error_rows["true_urgency"] = urgency_true
        error_rows["predicted_urgency"] = urgency_predictions
        true_situations = [
            "|".join(
                label for index, label in enumerate(situation_labels) if situation_true[row_index, index]
            )
            for row_index in range(len(data))
        ]
        predicted_situations = [
            "|".join(
                label
                for index, label in enumerate(situation_labels)
                if situation_predictions[row_index, index]
            )
            for row_index in range(len(data))
        ]
        error_rows["true_situation_labels"] = true_situations
        error_rows["predicted_situation_labels"] = predicted_situations
        error_rows = error_rows.loc[
            (error_rows["true_urgency"] != error_rows["predicted_urgency"])
            | (
                error_rows["true_situation_labels"]
                != error_rows["predicted_situation_labels"]
            )
        ]
        columns = [
            column
            for column in [
                "id",
                "text",
                "true_urgency",
                "predicted_urgency",
                "true_situation_labels",
                "predicted_situation_labels",
            ]
            if column in error_rows
        ]
        error_dump.parent.mkdir(parents=True, exist_ok=True)
        error_rows.loc[:, columns].to_csv(error_dump, index=False, encoding="utf-8")
        print(f"Error dump: {error_dump} ({len(error_rows)} rows)")
    print(f"Evaluation file: in-memory frame ({len(data)} rows)")
    print(f"Rows evaluated: {evaluated_row_count}")
    print(f"Files written: {summary['metrics_csv']}, {output_dir / f'{result_label}_metrics.json'}, "
          f"{output_dir / f'{result_label}_urgency_confusion_matrix.png'}")
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


def _load_model_metadata(models_dir: Path) -> dict[str, Any]:
    metadata_path = models_dir / "model_metadata.json"
    try:
        return json.loads(metadata_path.read_text(encoding="utf-8"))
    except FileNotFoundError as exc:
        raise ValueError(f"Model metadata file not found: {metadata_path}") from exc


def _validate_evaluation_labels(
    data: pd.DataFrame,
    metadata: dict[str, Any],
) -> None:
    model_urgency = set(metadata.get("urgency_labels", []))
    model_situation = set(metadata.get("situation_labels", []))
    if not model_urgency or not model_situation:
        raise ValueError(
            "model_metadata.json must contain non-empty urgency_labels and situation_labels."
        )
    observed_urgency = set(data["urgency"].astype(str))
    observed_situation = {
        label
        for value in data["situation_labels"]
        for label in str(value).split("|")
        if label
    }
    unknown_urgency = sorted(observed_urgency - model_urgency)
    unknown_situation = sorted(observed_situation - model_situation)
    if unknown_urgency or unknown_situation:
        details = []
        if unknown_urgency:
            details.append("urgency labels: " + ", ".join(unknown_urgency))
        if unknown_situation:
            details.append("situation labels: " + ", ".join(unknown_situation))
        raise ValueError(
            "Evaluation labels do not match the model training label list; "
            + "; ".join(details)
        )


def _write_clean_noisy_comparison(
    clean_summary: dict[str, Any],
    noisy_summary: dict[str, Any],
    output_dir: Path,
    split_name: str,
) -> tuple[Path, Path]:
    clean = pd.DataFrame(clean_summary["rows"])
    noisy = pd.DataFrame(noisy_summary["rows"])
    keys = ["task", "metric_scope", "label"]
    clean = clean[keys + ["f1", "support"]].rename(
        columns={"f1": "clean_f1", "support": "clean_support"}
    )
    noisy = noisy[keys + ["f1", "support"]].rename(
        columns={"f1": "noisy_f1", "support": "noisy_support"}
    )
    comparison = clean.merge(noisy, on=keys, how="outer")
    csv_path = output_dir / f"{split_name}_clean_noisy_comparison.csv"
    json_path = output_dir / f"{split_name}_clean_noisy_comparison.json"
    comparison.to_csv(csv_path, index=False)
    json_path.write_text(
        json.dumps(
            {
                "result_label": "CLEAN_VS_NOISY_FINAL_TEST",
                "clean_metrics": clean_summary["metrics_csv"],
                "noisy_metrics": noisy_summary["metrics_csv"],
                "rows": comparison.fillna("").to_dict("records"),
            },
            indent=2,
            default=float,
        ),
        encoding="utf-8",
    )
    return csv_path, json_path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path)
    parser.add_argument("--models-dir", type=Path, default=MODELS_DIR)
    parser.add_argument("--output-dir", type=Path, default=RESULTS_DIR)
    parser.add_argument("--model-type", choices=("tfidf", "embedding"), default="tfidf")
    parser.add_argument("--error-dump", action="store_true")
    parser.add_argument("--final-test", action="store_true")
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args()
    if args.smoke:
        run_smoke(args.output_dir)
        print("SMOKE TEST, NOT A REAL RESULT")
        return
    split_name = "test" if args.final_test else "val"
    if args.model_type == "embedding":
        if args.models_dir == MODELS_DIR:
            args.models_dir = EMBEDDING_MODELS_DIR
        if args.output_dir == RESULTS_DIR:
            args.output_dir = EMBEDDING_RESULTS_DIR
    data_path = args.data or PROCESSED_DIR / f"{split_name}.csv"
    if args.final_test:
        print("WARNING: final test evaluation should be run once only.")
    data = load_evaluation_data(data_path, final_test=args.final_test)
    metadata = _load_model_metadata(args.models_dir)
    _validate_evaluation_labels(data, metadata)
    print(f"Evaluation file: {data_path} ({len(data)} rows)")
    print(f"Model directory: {args.models_dir}")
    print(
        "Model metadata: "
        f"training_row_count={metadata.get('training_row_count')}, "
        f"urgency_labels={metadata.get('urgency_labels')}, "
        f"situation_labels={metadata.get('situation_labels')}"
    )
    if args.model_type == "embedding":
        from sakhi.src.train_embedding import EmbeddingArtifacts, _load_encoder

        artifacts = EmbeddingArtifacts(
            encoder=_load_encoder(metadata["model_name"]),
            urgency_model=joblib.load(args.models_dir / "urgency_model.joblib"),
            situation_model=joblib.load(args.models_dir / "situation_model.joblib"),
            situation_labels=metadata["situation_labels"],
            thresholds=json.loads(
                (args.models_dir / "situation_thresholds.json").read_text()
            ),
        )
    else:
        artifacts = BaselineArtifacts(
            vectorizer=joblib.load(args.models_dir / "tfidf_features.joblib"),
            urgency_model=joblib.load(args.models_dir / "urgency_model.joblib"),
            situation_model=joblib.load(args.models_dir / "situation_model.joblib"),
            situation_labels=metadata["situation_labels"],
            thresholds=json.loads(
                (args.models_dir / "situation_thresholds.json").read_text()
            ),
        )
    error_dump = (
        ERRORS_DIR / f"errors_{args.model_type}.csv" if args.error_dump else None
    )
    summary = evaluate_frames(artifacts, data, args.output_dir, split_name, error_dump)
    metrics = pd.DataFrame(summary["rows"])
    print("\nUrgency and situation scores")
    print(
        metrics.loc[
            metrics["metric_scope"].isin({"per_class", "per_label"}),
            ["task", "label", "f1", "support"],
        ].to_string(index=False)
    )
    if args.final_test:
        noisy_path = data_path.with_name("test_noisy.csv")
        if noisy_path.is_file():
            noisy_data = load_evaluation_data(noisy_path, final_test=True)
            _validate_evaluation_labels(noisy_data, metadata)
            print(f"Noisy evaluation file: {noisy_path} ({len(noisy_data)} rows)")
            noisy_summary = evaluate_frames(
                artifacts,
                noisy_data,
                args.output_dir,
                "test_noisy",
            )
            clean_noisy_csv, clean_noisy_json = _write_clean_noisy_comparison(
                summary, noisy_summary, args.output_dir, "test"
            )
            clean_noisy = pd.DataFrame(
                json.loads(clean_noisy_json.read_text(encoding="utf-8"))["rows"]
            )
            print("\nClean vs noisy final-test scores")
            print(clean_noisy.to_string(index=False))
            print(f"Files written: {clean_noisy_csv}, {clean_noisy_json}")
        else:
            print(f"No noisy final-test file found; skipped: {noisy_path}")
    print("\nPer-language scores")
    print(
        metrics.loc[
            metrics["metric_scope"].isin({"per_language", "per_language_label"}),
            ["task", "label", "f1", "support"],
        ].to_string(index=False)
    )


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    main()
