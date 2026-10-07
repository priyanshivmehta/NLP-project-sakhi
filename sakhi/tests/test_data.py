"""Tests for config-driven dataset validation and splitting."""

from pathlib import Path

import pandas as pd
import pytest

from sakhi.src.data import (
    DatasetValidationError,
    create_splits,
    load_dataset,
    validate_dataset,
)

SAMPLE_PATH = Path(__file__).parents[1] / "data" / "raw" / "sample_fictional.csv"


def test_sample_dataset_loads_and_has_required_rows() -> None:
    data = load_dataset(SAMPLE_PATH)
    assert len(data) == 10
    assert set(data["language"]) == {"hi", "hinglish", "en"}
    assert set(data["perspective"]) == {"first", "third"}


def test_validation_reports_multiple_errors() -> None:
    data = pd.read_csv(SAMPLE_PATH)
    data.loc[0, "text"] = " "
    data.loc[1, "language"] = "fr"
    data.loc[2, "situation_labels"] = "not_a_label"
    data.loc[3, "id"] = data.loc[4, "id"]

    with pytest.raises(DatasetValidationError) as error:
        validate_dataset(data)

    message = str(error.value)
    assert "empty text" in message
    assert "invalid languages" in message
    assert "invalid situation labels" in message
    assert "duplicate ids" in message


def test_split_is_reproducible_and_saved(tmp_path: Path) -> None:
    data = load_dataset(SAMPLE_PATH)
    first = create_splits(data, tmp_path / "first")
    second = create_splits(data, tmp_path / "second")

    assert {name: len(frame) for name, frame in first.items()} == {
        "train": 7,
        "val": 1,
        "test": 2,
    }
    for split_name in first:
        pd.testing.assert_frame_equal(first[split_name], second[split_name])
        assert (tmp_path / "first" / f"{split_name}.csv").exists()

