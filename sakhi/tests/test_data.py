"""Tests for config-driven dataset validation and splitting."""

from pathlib import Path

import pandas as pd
import pytest

from sakhi.src.data import (
    DatasetValidationError,
    create_splits,
    find_cross_split_text_duplicates,
    load_dataset,
    validate_dataset,
)

SAMPLE_PATH = Path(__file__).parents[1] / "data" / "raw" / "sample_fictional.csv"


def test_sample_dataset_loads_and_has_required_rows() -> None:
    data = load_dataset(SAMPLE_PATH)
    assert len(data) == 10
    assert set(data["language"]) == {"hi", "hinglish", "en"}
    assert set(data["perspective"]) == {"first", "third"}
    assert data["group_id"].is_unique


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
    with pytest.raises(ValueError, match="only for tests"):
        create_splits(data, tmp_path / "first", source_path=SAMPLE_PATH)


def test_group_ids_stay_in_one_split(tmp_path: Path) -> None:
    data = load_dataset(SAMPLE_PATH)
    first = create_splits(
        data.assign(id=[f"production-{i}" for i in range(10)]),
        tmp_path / "first",
        require_test_coverage=False,
    )

    assert {name: len(frame) for name, frame in first.items()} == {
        "train": 7,
        "val": 1,
        "test": 2,
    }
    assert not (tmp_path / "frozen" / "frozen_dataset.csv").exists()
    for split_name, split_data in first.items():
        assert (tmp_path / "first" / f"{split_name}.csv").exists()
        assert split_data["group_id"].is_unique
    train_groups = set(first["train"]["group_id"])
    val_groups = set(first["val"]["group_id"])
    test_groups = set(first["test"]["group_id"])
    assert train_groups.isdisjoint(val_groups)
    assert train_groups.isdisjoint(test_groups)
    assert val_groups.isdisjoint(test_groups)


def test_preview_does_not_create_frozen_dataset(tmp_path: Path) -> None:
    data = load_dataset(SAMPLE_PATH)
    data = data.assign(id=[f"production-{i}" for i in range(10)])
    output_dir = tmp_path / "frozen"
    create_splits(data, output_dir, require_test_coverage=False)
    assert not (output_dir / "frozen_dataset.csv").exists()


def test_freeze_requires_complete_dataset(tmp_path: Path) -> None:
    data = load_dataset(SAMPLE_PATH).assign(id=[f"production-{i}" for i in range(10)])
    with pytest.raises(ValueError, match="Refusing to freeze"):
        create_splits(data, tmp_path / "freeze", require_test_coverage=False, freeze=True)


def test_class_counts_are_saved_for_preview(tmp_path: Path) -> None:
    data = load_dataset(SAMPLE_PATH).assign(id=[f"production-{i}" for i in range(10)])
    create_splits(data, tmp_path / "counts", require_test_coverage=False)
    counts = pd.read_csv(tmp_path / "counts" / "split_class_counts.csv")
    assert {"urgency", "situation"} == set(counts["task"])


def test_no_group_id_crosses_splits() -> None:
    data = load_dataset(SAMPLE_PATH)
    split_groups = {
        split_name: set(data.loc[data["split"] == split_name, "group_id"])
        for split_name in ("train", "val", "test")
    }
    assert split_groups["train"].isdisjoint(split_groups["val"])
    assert split_groups["train"].isdisjoint(split_groups["test"])
    assert split_groups["val"].isdisjoint(split_groups["test"])


def test_cross_split_exact_and_near_duplicates_are_flagged() -> None:
    data = pd.DataFrame(
        {
            "id": ["a", "b", "c"],
            "text": ["Same scenario text.", "same   scenario text", "Completely different"],
            "split": ["train", "test", "val"],
        }
    )
    findings = find_cross_split_text_duplicates(data)
    assert len(findings) == 1
    assert findings[0]["left_id"] == "a"
    assert findings[0]["right_id"] == "b"


def test_test_coverage_is_required_for_real_freeze(tmp_path: Path) -> None:
    data = load_dataset(SAMPLE_PATH).assign(id=[f"production-{i}" for i in range(10)])
    with pytest.raises(DatasetValidationError, match="test split urgency counts below"):
        create_splits(data, tmp_path / "coverage")


def test_freeze_complete_synthetic_dataset_is_group_contained(tmp_path: Path) -> None:
    """Freeze a complete in-memory dataset without writing outside tmp_path."""
    situations = [
        "street_harassment",
        "stalking",
        "domestic_violence",
        "workplace_harassment",
        "dowry_harassment",
        "cyber_abuse",
        "other",
    ]
    urgencies = ["immediate_danger", "urgent", "non_urgent", "info_seeking"]
    rows = []
    for index in range(450):
        rows.append(
            {
                "id": f"scenario-{index:03d}",
                "text": f"Synthetic scenario {index} has a distinct test description.",
                "language": ("en", "hi", "hinglish")[index % 3],
                "perspective": ("first", "third")[index % 2],
                "situation_labels": situations[index % len(situations)],
                "urgency": urgencies[index % len(urgencies)],
                "legal_refs": "",
                "source": "synthetic_test",
                "annotator": "test",
                "split": "unassigned",
                "group_id": f"group-{index:03d}",
            }
        )
    data = pd.DataFrame(rows)
    output_dir = tmp_path / "freeze"

    splits = create_splits(data, output_dir=output_dir, freeze=True)

    for filename in (
        "frozen_dataset.csv",
        "train.csv",
        "val.csv",
        "test.csv",
        "split_class_counts.csv",
    ):
        assert (output_dir / filename).is_file()

    split_groups = {
        split_name: set(split_data["group_id"])
        for split_name, split_data in splits.items()
    }
    assert split_groups["train"].isdisjoint(split_groups["val"])
    assert split_groups["train"].isdisjoint(split_groups["test"])
    assert split_groups["val"].isdisjoint(split_groups["test"])

    with pytest.raises(FileExistsError, match="must not be regenerated"):
        create_splits(data, output_dir=output_dir, freeze=True)
