"""Tests for the train-only baseline and guarded evaluator."""

import json
from pathlib import Path
from types import SimpleNamespace

import pandas as pd
import pytest

from sakhi.src.evaluate import (
    _validate_evaluation_labels,
    _write_clean_noisy_comparison,
    evaluate_frames,
    load_evaluation_data,
    run_smoke,
)
from sakhi.src.compare_models import compare_models
from sakhi.src.train_embedding import train_embedding_from_frames
from sakhi.src.train_baseline import fit_vectorizers, train_from_frames


def test_smoke_path_writes_labelled_results(tmp_path: Path) -> None:
    result = run_smoke(tmp_path / "results")
    assert result["result_label"] == "SMOKE_TEST_NOT_A_REAL_RESULT"
    assert (tmp_path / "results" / "SMOKE_TEST_NOT_A_REAL_RESULT_metrics.csv").exists()
    assert (tmp_path / "results" / "SMOKE_TEST_NOT_A_REAL_RESULT_urgency_confusion_matrix.png").exists()


def test_vectorizer_is_fitted_on_train_only() -> None:
    train = pd.Series(["trainonlytoken common"])
    vectorizer = fit_vectorizers(train)
    word_vocabulary = vectorizer.transformer_list[0][1].vocabulary_
    assert "trainonlytoken" in word_vocabulary
    assert "validationonlytoken" not in word_vocabulary


def test_training_does_not_read_test_csv(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    sample = pd.DataFrame(
        {
            "final_model_input": ["safe text", "other safe text"],
            "urgency": ["urgent", "non_urgent"],
            "situation_labels": ["other", "street_harassment"],
        }
    )
    train = tmp_path / "train.csv"
    val = tmp_path / "val.csv"
    test = tmp_path / "test.csv"
    sample.to_csv(train, index=False)
    sample.to_csv(val, index=False)
    test.write_text("must not be read", encoding="utf-8")
    original_read_csv = pd.read_csv

    def guarded_read_csv(path, *args, **kwargs):
        if Path(path) == test:
            raise AssertionError("test.csv was read during training")
        return original_read_csv(path, *args, **kwargs)

    monkeypatch.setattr(pd, "read_csv", guarded_read_csv)
    train_from_frames(original_read_csv(train), original_read_csv(val), tmp_path / "models")


def test_training_metadata_records_rows_and_labels(tmp_path: Path) -> None:
    data = pd.DataFrame(
        {
            "final_model_input": ["safe text", "other safe text"],
            "urgency": ["urgent", "non_urgent"],
            "situation_labels": ["other", "street_harassment"],
        }
    )
    train_from_frames(data, data, tmp_path / "models")
    metadata = json.loads((tmp_path / "models" / "model_metadata.json").read_text())
    assert metadata["training_row_count"] == 2
    assert metadata["urgency_labels"] == ["non_urgent", "urgent"]
    assert "street_harassment" in metadata["situation_labels"]


def test_evaluation_rejects_labels_not_in_model_metadata() -> None:
    data = pd.DataFrame(
        {
            "urgency": ["urgent"],
            "situation_labels": ["unknown_label"],
        }
    )
    with pytest.raises(ValueError, match="do not match the model training label list"):
        _validate_evaluation_labels(
            data,
            {
                "urgency_labels": ["urgent"],
                "situation_labels": ["other"],
            },
        )


def test_evaluation_rejects_prediction_row_count_mismatch(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    data = pd.DataFrame(
        {
            "urgency": ["urgent", "non_urgent"],
            "situation_labels": ["other", "other"],
            "language": ["en", "en"],
            "source": ["team_written", "team_written"],
            "human_verified": [True, True],
        }
    )
    monkeypatch.setattr(
        "sakhi.src.evaluate._predict",
        lambda artifacts, frame: (["urgent"], [[1] * 8]),
    )
    with pytest.raises(ValueError, match="Evaluated row count does not match"):
        evaluate_frames(
            SimpleNamespace(situation_labels=["other"] * 8),
            data,
            tmp_path,
        )


def test_test_csv_requires_final_test_flag(tmp_path: Path) -> None:
    test_path = tmp_path / "test.csv"
    test_path.write_text("final_model_input\ntext\n", encoding="utf-8")
    with pytest.raises(ValueError, match="requires the explicit --final-test"):
        load_evaluation_data(test_path)


def test_clean_noisy_comparison_writes_side_by_side_metrics(tmp_path: Path) -> None:
    clean = {
        "metrics_csv": "clean.csv",
        "rows": [
            {
                "task": "urgency",
                "metric_scope": "per_class",
                "label": "urgent",
                "f1": 0.8,
                "support": 2,
            }
        ],
    }
    noisy = {
        "metrics_csv": "noisy.csv",
        "rows": [
            {
                "task": "urgency",
                "metric_scope": "per_class",
                "label": "urgent",
                "f1": 0.5,
                "support": 2,
            }
        ],
    }
    csv_path, json_path = _write_clean_noisy_comparison(
        clean, noisy, tmp_path, "test"
    )
    result = pd.read_csv(csv_path)
    assert result.loc[0, "clean_f1"] == 0.8
    assert result.loc[0, "noisy_f1"] == 0.5
    assert json_path.exists()


class _FakeEncoder:
    def encode(self, texts: list[str], show_progress_bar: bool = False) -> list[list[float]]:
        return [[float(len(text)), float(index + 1), 1.0] for index, text in enumerate(texts)]


def test_embedding_training_uses_fake_encoder_and_caches_embeddings(tmp_path: Path) -> None:
    data = pd.DataFrame(
        {
            "final_model_input": ["one", "two", "three", "four", "five", "six"],
            "urgency": ["urgent", "non_urgent", "urgent", "non_urgent", "urgent", "non_urgent"],
            "situation_labels": ["other", "street_harassment"] * 3,
        }
    )
    artifacts = train_embedding_from_frames(
        data,
        data,
        tmp_path / "embedding",
        model_name="fake-model",
        encoder=_FakeEncoder(),
    )
    assert artifacts.situation_labels
    assert (tmp_path / "embedding" / "embedding_cache" / "train.npy").exists()
    metadata = json.loads((tmp_path / "embedding" / "model_metadata.json").read_text())
    assert metadata["model_name"] == "fake-model"
    assert metadata["training_row_count"] == 6


def test_compare_models_writes_csv_and_markdown(tmp_path: Path) -> None:
    for folder in (tmp_path / "tfidf", tmp_path / "embedding"):
        folder.mkdir()
        pd.DataFrame(
            [
                {
                    "task": "urgency",
                    "metric_scope": "macro",
                    "label": "macro_avg",
                    "f1": 0.5,
                    "support": 2,
                    "data_quality": "HUMAN_OR_MIXED",
                }
            ]
        ).to_csv(folder / "val_metrics.csv", index=False)
    result = compare_models(
        tmp_path / "tfidf",
        tmp_path / "embedding",
        tmp_path / "comparison",
    )
    assert len(result) == 2
    assert (tmp_path / "comparison.csv").exists()
    assert (tmp_path / "comparison.md").exists()
