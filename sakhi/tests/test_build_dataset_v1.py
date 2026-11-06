from pathlib import Path

import pandas as pd
import pytest

from sakhi.src.build_dataset_v1 import build_dataset, write_readiness_report


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
