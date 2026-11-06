from pathlib import Path

import pandas as pd

from sakhi.src.augment_noise import augment_verified_rows
from sakhi.src.import_verified_eval import import_reviewed_sheet
from sakhi.src.merge_eval_batches import merge_batches


ROOT = Path(__file__).parents[1]


def _batch_row(text: str, language: str = "en") -> dict[str, str]:
    return {
        "text": text,
        "language": language,
        "perspective": "first",
        "situation_labels": "other",
        "urgency": "info_seeking",
    }


def test_ai_dataset_has_explicit_training_provenance() -> None:
    data = pd.read_csv(ROOT / "data" / "raw" / "dev_dataset_450.csv")
    assert set(data["generator"]) == {"chatgpt_train"}
    assert set(data["human_verified"]) == {False}


def test_merge_import_and_noise_workflow(tmp_path: Path) -> None:
    batch_dir = tmp_path / "batches"
    batch_dir.mkdir()
    pd.DataFrame(
        [
            _batch_row("A distinct evaluation message about process."),
            _batch_row("A second distinct evaluation message about options."),
        ]
    ).to_csv(batch_dir / "batch99_toolname.csv", index=False)
    train_path = tmp_path / "train.csv"
    pd.DataFrame(
        [{"id": "train-1", "text": "Training-only text."}]
    ).to_csv(train_path, index=False)
    candidates_path = tmp_path / "candidates.csv"
    review_path = tmp_path / "review.csv"
    removed_path = tmp_path / "removed.csv"

    merged = merge_batches(
        batch_dir,
        train_path,
        candidates_path,
        removed_path,
        review_path,
    )
    assert len(merged) == 2
    assert set(merged["human_verified"]) == {False}
    review = pd.read_csv(review_path)
    review.loc[0, "verify_status"] = "corrected"
    review.loc[0, "corrected_labels"] = "stalking"
    review.loc[0, "corrected_urgency"] = "urgent"
    review.loc[1, "verify_status"] = "reject"
    review.to_csv(review_path, index=False)

    verified_path = tmp_path / "verified.csv"
    verified = import_reviewed_sheet(review_path, candidates_path, verified_path)
    assert len(verified) == 1
    assert verified.loc[0, "human_verified"]
    assert verified.loc[0, "situation_labels"] == "stalking"
    assert verified.loc[0, "urgency"] == "urgent"

    noisy_path = tmp_path / "noisy.csv"
    noisy = augment_verified_rows(verified_path, noisy_path, seed=459)
    assert noisy["noisy"].tolist() == [True]
    assert noisy["text"].iloc[0] != ""
