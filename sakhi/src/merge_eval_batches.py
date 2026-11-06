"""Merge fictional evaluation batches and create a human-review workbook."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import pandas as pd
import yaml
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from sakhi.src.data import (
    AI_NEAR_DUPLICATE_THRESHOLD,
    CONFIG_PATH,
    DatasetValidationError,
    validate_dataset,
)

PROJECT_ROOT = Path(__file__).resolve().parents[2]
BATCH_DIR = PROJECT_ROOT / "sakhi" / "data" / "raw" / "eval_batches"
TRAIN_PATH = PROJECT_ROOT / "sakhi" / "data" / "raw" / "dev_dataset_450.csv"
CANDIDATE_PATH = PROJECT_ROOT / "sakhi" / "data" / "raw" / "eval_candidates.csv"
REMOVED_PATH = PROJECT_ROOT / "sakhi" / "reports" / "eval_removed_rows.csv"
REVIEW_PATH = PROJECT_ROOT / "sakhi" / "data" / "raw" / "eval_verification_sheet.csv"
REQUIRED_BATCH_COLUMNS = {
    "text",
    "language",
    "perspective",
    "situation_labels",
    "urgency",
}


def _read_labels() -> dict[str, list[str]]:
    with CONFIG_PATH.open(encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def _row_id(batch_name: str, index: int) -> str:
    return f"{Path(batch_name).stem}-{index:03d}"


def _drop_near_duplicates(
    candidates: pd.DataFrame,
    training: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Keep the first candidate in file order and record every removed row."""
    if candidates.empty:
        return candidates, pd.DataFrame(
            columns=["id", "text", "reason", "matched_id", "similarity"]
        )
    kept: list[int] = []
    removed: list[dict[str, Any]] = []
    training_texts = training["text"].astype(str).tolist()
    training_ids = training["id"].astype(str).tolist()
    for index, row in candidates.iterrows():
        text = str(row["text"])
        if any(text.casefold().strip() == existing.casefold().strip() for existing in training_texts):
            matched = next(
                item for item in training.index
                if str(training.loc[item, "text"]).casefold().strip() == text.casefold().strip()
            )
            removed.append(
                {
                    "id": row["id"],
                    "text": text,
                    "reason": "exact_duplicate_training",
                    "matched_id": training_ids[list(training.index).index(matched)],
                    "similarity": 1.0,
                }
            )
            continue
        comparison_texts = training_texts + [str(candidates.loc[item, "text"]) for item in kept]
        if comparison_texts:
            matrix = TfidfVectorizer().fit_transform(comparison_texts + [text])
            scores = cosine_similarity(matrix[-1], matrix[:-1]).ravel()
            if len(scores) and scores.max() > AI_NEAR_DUPLICATE_THRESHOLD:
                match_index = int(scores.argmax())
                matched_id = (
                    training_ids[match_index]
                    if match_index < len(training_ids)
                    else str(candidates.loc[kept[match_index - len(training_ids)], "id"])
                )
                removed.append(
                    {
                        "id": row["id"],
                        "text": text,
                        "reason": "near_duplicate",
                        "matched_id": matched_id,
                        "similarity": round(float(scores[match_index]), 4),
                    }
                )
                continue
        kept.append(index)
    return candidates.loc[kept].copy(), pd.DataFrame(removed)


def merge_batches(
    batch_dir: Path = BATCH_DIR,
    training_path: Path = TRAIN_PATH,
    candidate_path: Path = CANDIDATE_PATH,
    removed_path: Path = REMOVED_PATH,
    review_path: Path = REVIEW_PATH,
) -> pd.DataFrame:
    """Merge batch CSVs, remove leakage, and write candidate/review exports."""
    labels = _read_labels()
    training = pd.read_csv(training_path, encoding="utf-8")
    candidates: list[pd.DataFrame] = []
    for batch_path in sorted(batch_dir.glob("*.csv")):
        frame = pd.read_csv(batch_path, encoding="utf-8")
        missing = REQUIRED_BATCH_COLUMNS - set(frame.columns)
        if missing:
            raise DatasetValidationError(
                f"{batch_path.name} is missing columns: {', '.join(sorted(missing))}"
            )
        frame = frame.copy()
        frame.insert(0, "id", [_row_id(batch_path.name, i) for i in range(len(frame))])
        frame["group_id"] = frame["id"]
        frame["generator"] = batch_path.stem
        frame["source"] = "ai_generated"
        frame["annotator"] = "batch_generator"
        frame["legal_refs"] = ""
        frame["split"] = "unassigned"
        frame["human_verified"] = False
        candidates.append(frame)
    merged = pd.concat(candidates, ignore_index=True) if candidates else pd.DataFrame()
    if not merged.empty:
        validate_dataset(merged, labels)
    kept, removed = _drop_near_duplicates(merged, training)
    candidate_path.parent.mkdir(parents=True, exist_ok=True)
    kept.to_csv(candidate_path, index=False, encoding="utf-8")
    removed_path.parent.mkdir(parents=True, exist_ok=True)
    removed.to_csv(removed_path, index=False, encoding="utf-8")
    review = kept[
        ["id", "text", "language", "perspective", "situation_labels", "urgency"]
    ].copy()
    review["verify_status"] = "pending"
    review["corrected_labels"] = ""
    review["corrected_urgency"] = ""
    review["reviewer"] = ""
    review_path.parent.mkdir(parents=True, exist_ok=True)
    review.to_csv(review_path, index=False, encoding="utf-8")
    return kept


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--batch-dir", type=Path, default=BATCH_DIR)
    parser.add_argument("--training", type=Path, default=TRAIN_PATH)
    parser.add_argument("--output", type=Path, default=CANDIDATE_PATH)
    parser.add_argument("--removed", type=Path, default=REMOVED_PATH)
    parser.add_argument("--review", type=Path, default=REVIEW_PATH)
    args = parser.parse_args()
    merged = merge_batches(
        args.batch_dir, args.training, args.output, args.removed, args.review
    )
    print(f"Wrote {len(merged)} evaluation candidates to {args.output}")


if __name__ == "__main__":
    main()
