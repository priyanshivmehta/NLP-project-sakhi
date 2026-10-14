"""Tests for the train-only baseline and guarded evaluator."""

from pathlib import Path

import pandas as pd
import pytest

from sakhi.src.evaluate import load_evaluation_data, run_smoke
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


def test_test_csv_requires_final_test_flag(tmp_path: Path) -> None:
    test_path = tmp_path / "test.csv"
    test_path.write_text("final_model_input\ntext\n", encoding="utf-8")
    with pytest.raises(ValueError, match="requires the explicit --final-test"):
        load_evaluation_data(test_path)
