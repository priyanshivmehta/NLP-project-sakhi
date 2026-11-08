"""Merge validated extra training CSVs into one provenance-tagged dataset."""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from sakhi.src.data import (
    AI_NEAR_DUPLICATE_THRESHOLD,
    AI_SOURCE,
    CONFIG_PATH,
    DatasetValidationError,
    PROJECT_ROOT,
    load_label_config,
    _parse_pipe_separated,
)

INPUT_DIR = PROJECT_ROOT / "sakhi" / "data" / "raw" / "train_extra"
OUTPUT_PATH = PROJECT_ROOT / "sakhi" / "data" / "raw" / "train_extra.csv"
TRAIN_PATH = PROJECT_ROOT / "sakhi" / "data" / "raw" / "dev_dataset_450.csv"
EVAL_PATH = PROJECT_ROOT / "sakhi" / "data" / "raw" / "eval_verified.csv"
REMOVED_PATH = PROJECT_ROOT / "sakhi" / "reports" / "audit" / "removed_train_extra.csv"
INPUT_COLUMNS = {"text", "language", "perspective", "situation_labels", "urgency"}


def _validate_input(frame: pd.DataFrame, path: Path, config: dict[str, list[str]]) -> None:
    missing = sorted(INPUT_COLUMNS - set(frame.columns))
    if missing:
        raise DatasetValidationError(
            f"{path} is missing required columns: {', '.join(missing)}"
        )
    invalid_languages = sorted(set(frame["language"].astype(str)) - set(config["allowed_languages"]))
    invalid_perspectives = sorted(
        set(frame["perspective"].astype(str)) - set(config["allowed_perspectives"])
    )
    invalid_urgency = sorted(set(frame["urgency"].astype(str)) - set(config["urgency_labels"]))
    invalid_labels = sorted(
        {
            label
            for value in frame["situation_labels"]
            for label in _parse_pipe_separated(value)
            if label not in config["situation_labels"]
        }
    )
    errors = []
    if invalid_languages:
        errors.append("languages: " + ", ".join(invalid_languages))
    if invalid_perspectives:
        errors.append("perspectives: " + ", ".join(invalid_perspectives))
    if invalid_urgency:
        errors.append("urgency: " + ", ".join(invalid_urgency))
    if invalid_labels:
        errors.append("situation labels: " + ", ".join(invalid_labels))
    if frame["text"].isna().any() or frame["text"].astype(str).str.strip().eq("").any():
        errors.append("text contains an empty value")
    if errors:
        raise DatasetValidationError(f"Invalid rows in {path}: " + "; ".join(errors))


def _similarity_scores(left: pd.Series, right: pd.Series) -> list[float]:
    if left.empty or right.empty:
        return [0.0] * len(left)
    matrix = TfidfVectorizer().fit_transform(
        pd.concat([right.astype(str), left.astype(str)], ignore_index=True)
    )
    return cosine_similarity(matrix[len(right) :], matrix[: len(right)]).max(axis=1).tolist()


def merge_train_extra(
    input_dir: Path = INPUT_DIR,
    output_path: Path = OUTPUT_PATH,
    train_path: Path = TRAIN_PATH,
    eval_path: Path = EVAL_PATH,
    removed_path: Path = REMOVED_PATH,
) -> pd.DataFrame:
    config = load_label_config(CONFIG_PATH)
    paths = sorted(input_dir.glob("*.csv"))
    if not paths:
        raise FileNotFoundError(f"No training CSV files found in {input_dir}")
    frames = []
    for path in paths:
        frame = pd.read_csv(path, encoding="utf-8")
        _validate_input(frame, path, config)
        frames.append(frame.loc[:, sorted(INPUT_COLUMNS)].copy())
    candidates = pd.concat(frames, ignore_index=True)
    candidates["_normalized_text"] = (
        candidates["text"].astype(str).str.casefold().str.replace(r"\s+", " ", regex=True).str.strip()
    )
    removed: list[dict[str, object]] = []
    kept_indices: list[int] = []
    kept_texts: list[str] = []
    for index, row in candidates.iterrows():
        text = row["_normalized_text"]
        if text in kept_texts:
            removed.append({"source_row": int(index), "text": row["text"], "reason": "exact_duplicate"})
            continue
        prior = pd.Series(kept_texts, dtype="string")
        score = _similarity_scores(pd.Series([text]), prior)[0] if kept_texts else 0.0
        if score > AI_NEAR_DUPLICATE_THRESHOLD:
            removed.append(
                {"source_row": int(index), "text": row["text"], "reason": "near_duplicate", "similarity": score}
            )
            continue
        kept_indices.append(index)
        kept_texts.append(text)
    candidates = candidates.loc[kept_indices].copy()
    reference_frames = []
    for path in (train_path, eval_path):
        if path.exists():
            reference = pd.read_csv(path, encoding="utf-8")
            if "text" in reference.columns:
                reference_frames.append(reference["text"].astype(str))
    references = pd.concat(reference_frames, ignore_index=True) if reference_frames else pd.Series(dtype="string")
    scores = _similarity_scores(candidates["_normalized_text"], references)
    keep = [score <= AI_NEAR_DUPLICATE_THRESHOLD for score in scores]
    for index, score in zip(candidates.index[~pd.Series(keep, index=candidates.index)], [s for s, ok in zip(scores, keep) if not ok]):
        removed.append(
            {"source_row": int(index), "text": candidates.loc[index, "text"], "reason": "near_duplicate_existing", "similarity": score}
        )
    candidates = candidates.loc[keep].copy()
    candidates = candidates.drop(columns="_normalized_text")
    candidates.insert(0, "id", [f"train-extra-{index:04d}" for index in range(1, len(candidates) + 1)])
    candidates["legal_refs"] = ""
    candidates["source"] = AI_SOURCE
    candidates["annotator"] = "train_extra_merge"
    candidates["split"] = ""
    candidates["group_id"] = [f"train-extra-group-{index:04d}" for index in range(1, len(candidates) + 1)]
    candidates["generator"] = "chatgpt_train"
    candidates["human_verified"] = False
    columns = [
        "id", "text", "language", "perspective", "situation_labels", "urgency",
        "legal_refs", "source", "annotator", "split", "group_id", "generator", "human_verified",
    ]
    output_path.parent.mkdir(parents=True, exist_ok=True)
    removed_path.parent.mkdir(parents=True, exist_ok=True)
    candidates.to_csv(output_path, index=False, encoding="utf-8")
    pd.DataFrame(removed).to_csv(removed_path, index=False, encoding="utf-8")
    return candidates


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", type=Path, default=INPUT_DIR)
    parser.add_argument("--output", type=Path, default=OUTPUT_PATH)
    parser.add_argument("--train", type=Path, default=TRAIN_PATH)
    parser.add_argument("--eval", type=Path, default=EVAL_PATH)
    parser.add_argument("--removed", type=Path, default=REMOVED_PATH)
    args = parser.parse_args()
    result = merge_train_extra(args.input_dir, args.output, args.train, args.eval, args.removed)
    print(f"Wrote {len(result)} extra training rows to {args.output}")


if __name__ == "__main__":
    main()
