from pathlib import Path

import pandas as pd
import pytest

from sakhi.src.build_dataset_v1 import build_dataset, write_readiness_report
from sakhi.src.merge_train_extra import merge_train_extra
from sakhi.src.data import (
    MIN_VAL_EXAMPLES_PER_IMMEDIATE_DANGER,
    MIN_VAL_EXAMPLES_PER_LANGUAGE,
    TEST_SHARE_OF_EVAL,
    _allocate_coverage_holdouts,
)


def _row(index: int, source: str, generator: str, verified: bool, text: str) -> dict:
    return {
        "id": f"{source}-{index}",
        "group_id": f"{source}-group-{index}",
        "text": text,
        "language": ("hi", "hinglish", "en")[index % 3],
        "perspective": "first",
        "situation_labels": "other",
        "urgency": ("immediate_danger", "urgent", "non_urgent", "info_seeking")[index % 4],
        "legal_refs": "",
        "source": source,
        "annotator": "test",
        "split": "",
        "generator": generator,
        "human_verified": verified,
    }


def test_build_requires_imported_review(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError, match="review step has not been imported"):
        build_dataset(tmp_path / "train.csv", tmp_path / "eval_verified.csv")


def test_build_removes_training_near_duplicates_and_reports_readiness(tmp_path: Path) -> None:
    train = pd.DataFrame(
        [
            _row(0, "ai_generated", "chatgpt_train", False, "A distinct training message."),
            _row(1, "ai_generated", "chatgpt_train", False, "The same safety message."),
        ]
    )
    evaluation = pd.DataFrame(
        [
            _row(10, "ai_generated", "claude", True, "The same safety message."),
            _row(11, "ai_generated", "claude", True, "A verified evaluation message."),
            _row(12, "ai_generated", "claude", True, "Another verified evaluation message."),
        ]
    )
    train_path = tmp_path / "train.csv"
    eval_path = tmp_path / "eval_verified.csv"
    output_path = tmp_path / "dataset_v1.csv"
    removed_path = tmp_path / "removed.csv"
    train.to_csv(train_path, index=False)
    evaluation.to_csv(eval_path, index=False)

    combined = build_dataset(train_path, eval_path, output_path, removed_path)
    removed = pd.read_csv(removed_path)
    assert len(combined) == 4
    assert removed["id"].tolist() == ["ai_generated-1"]
    assert set(combined["generator"]) == {"chatgpt_train", "claude"}

    report = write_readiness_report(combined, tmp_path / "readiness.md")
    assert "PREVIEW split only" in report
    assert "Train generators: chatgpt_train" in report
    assert "Val/test generators: claude" in report
    assert "no generator overlap" in report


def test_coverage_allocator_targets_fifty_five_percent_test() -> None:
    rows = []
    for index in range(180):
        language = ("hi", "hinglish", "en")[index % 3]
        urgency = "immediate_danger" if index < 20 else "urgent"
        rows.append(
            {
                "group_id": f"group-{index}",
                "language": language,
                "urgency": urgency,
                "situation_labels": "other",
            }
        )
    groups = pd.DataFrame(rows).set_index("group_id")
    val, test = _allocate_coverage_holdouts(groups, len(groups))
    assert len(test) == round(len(groups) * TEST_SHARE_OF_EVAL)
    assert all(
        int((groups.loc[test, "language"] == language).sum()) >= 30
        for language in ("hi", "hinglish", "en")
    )
    assert all(
        int((groups.loc[val, "language"] == language).sum())
        >= MIN_VAL_EXAMPLES_PER_LANGUAGE
        for language in ("hi", "hinglish", "en")
    )
    assert int((groups.loc[val, "urgency"] == "immediate_danger").sum()) >= (
        MIN_VAL_EXAMPLES_PER_IMMEDIATE_DANGER
    )


def test_build_includes_optional_extra_training_rows(tmp_path: Path) -> None:
    train = pd.DataFrame([_row(0, "ai_generated", "chatgpt_train", False, "Base training text.")])
    extra = pd.DataFrame([_row(1, "ai_generated", "chatgpt_train", False, "Extra training text.")])
    evaluation = pd.DataFrame([_row(2, "ai_generated", "claude", True, "Verified evaluation text.")])
    train_path = tmp_path / "train.csv"
    extra_path = tmp_path / "train_extra.csv"
    eval_path = tmp_path / "eval_verified.csv"
    train.to_csv(train_path, index=False)
    extra.to_csv(extra_path, index=False)
    evaluation.to_csv(eval_path, index=False)

    combined = build_dataset(
        train_path, eval_path, tmp_path / "dataset.csv", tmp_path / "removed.csv", extra_path
    )

    assert len(combined) == 3
    assert "Extra training text." in set(combined["text"])


def test_readiness_reports_training_label_shortfall(tmp_path: Path) -> None:
    data = pd.DataFrame(
        [
            _row(0, "ai_generated", "chatgpt_train", False, "Training text."),
            _row(1, "ai_generated", "claude", True, "Verified text."),
            _row(2, "ai_generated", "claude", True, "Another verified text."),
            _row(3, "ai_generated", "claude", True, "Third verified text."),
        ]
    )
    report = write_readiness_report(data, tmp_path / "readiness.md")
    assert "FAIL — train sexual_violence" in report
    assert "shortfall" in report


def test_merge_train_extra_filters_duplicates_and_adds_provenance(tmp_path: Path) -> None:
    input_dir = tmp_path / "train_extra"
    input_dir.mkdir()
    rows = [
        {
            "text": "A distinct extra training message about support options.",
            "language": "en",
            "perspective": "first",
            "situation_labels": "other",
            "urgency": "info_seeking",
        },
        {
            "text": "A distinct extra training message about support options.",
            "language": "en",
            "perspective": "first",
            "situation_labels": "other",
            "urgency": "info_seeking",
        },
        {
            "text": "A training message already present.",
            "language": "en",
            "perspective": "first",
            "situation_labels": "other",
            "urgency": "info_seeking",
        },
    ]
    pd.DataFrame(rows).to_csv(input_dir / "batch.csv", index=False)
    train_path = tmp_path / "train.csv"
    pd.DataFrame({"text": ["A training message already present."]}).to_csv(train_path, index=False)

    merged = merge_train_extra(
        input_dir,
        tmp_path / "train_extra.csv",
        train_path,
        tmp_path / "missing_eval.csv",
        tmp_path / "removed.csv",
    )

    assert len(merged) == 1
    assert merged["id"].tolist() == ["train-extra-0001"]
    assert merged["group_id"].is_unique
    assert set(merged["source"]) == {"ai_generated"}
    assert set(merged["generator"]) == {"chatgpt_train"}
    assert set(merged["human_verified"]) == {False}
    removed = pd.read_csv(tmp_path / "removed.csv")
    assert set(removed["reason"]) == {"exact_duplicate", "near_duplicate_existing"}
