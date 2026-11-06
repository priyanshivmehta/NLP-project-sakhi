"""Build dataset_v1 and produce a non-freezing readiness report."""

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
    MIN_ROWS_TO_FREEZE,
    MIN_TEST_EXAMPLES_PER_LANGUAGE,
    MIN_TEST_EXAMPLES_PER_SITUATION,
    MIN_TEST_EXAMPLES_PER_URGENCY,
    MIN_VERIFIED_EVAL_ROWS_TO_FREEZE,
    PROJECT_ROOT,
    _allocate_group_splits,
    _parse_pipe_separated,
    _verified_mask,
    load_label_config,
    validate_dataset,
)

TRAIN_PATH = PROJECT_ROOT / "sakhi" / "data" / "raw" / "dev_dataset_450.csv"
EVAL_PATH = PROJECT_ROOT / "sakhi" / "data" / "raw" / "eval_verified.csv"
OUTPUT_PATH = PROJECT_ROOT / "sakhi" / "data" / "raw" / "dataset_v1.csv"
REMOVED_PATH = PROJECT_ROOT / "sakhi" / "reports" / "audit" / "removed_train_rows.csv"
READINESS_PATH = PROJECT_ROOT / "sakhi" / "reports" / "audit" / "readiness.md"


def _remove_training_near_duplicates(
    training: pd.DataFrame, evaluation: pd.DataFrame
) -> tuple[pd.DataFrame, pd.DataFrame]:
    if training.empty or evaluation.empty:
        return training.copy(), pd.DataFrame(columns=["id", "similarity"])
    matrix = TfidfVectorizer().fit_transform(
        pd.concat([evaluation["text"], training["text"]], ignore_index=True).astype(str)
    )
    scores = cosine_similarity(matrix[len(evaluation) :], matrix[: len(evaluation)])
    max_scores = scores.max(axis=1)
    keep = max_scores <= AI_NEAR_DUPLICATE_THRESHOLD
    removed = pd.DataFrame(
        {
            "id": training.loc[~keep, "id"].astype(str),
            "similarity": max_scores[~keep],
        }
    )
    return training.loc[keep].copy(), removed


def build_dataset(
    train_path: Path = TRAIN_PATH,
    eval_path: Path = EVAL_PATH,
    output_path: Path = OUTPUT_PATH,
    removed_path: Path = REMOVED_PATH,
) -> pd.DataFrame:
    if not eval_path.exists():
        raise FileNotFoundError(
            f"review step has not been imported yet: {eval_path} does not exist. "
            "Import the reviewed verification sheet before building dataset_v1."
        )
    training = pd.read_csv(train_path, encoding="utf-8")
    evaluation = pd.read_csv(eval_path, encoding="utf-8")
    required = {"id", "group_id", "text", "source", "generator", "human_verified"}
    for name, frame in (("training", training), ("verified evaluation", evaluation)):
        missing = sorted(required - set(frame.columns))
        if missing:
            raise ValueError(f"{name} is missing required columns: {', '.join(missing)}")
    if not _verified_mask(evaluation).all():
        raise ValueError("eval_verified.csv contains rows that are not human_verified=true.")
    if (training["source"] != AI_SOURCE).any() or _verified_mask(training).any():
        raise ValueError("Training input must be ai_generated and human_verified=false.")
    if training["id"].duplicated().any() or evaluation["id"].duplicated().any():
        raise ValueError("Input IDs must be unique within each input file.")
    if set(training["id"]) & set(evaluation["id"]):
        raise ValueError("Training and verified evaluation IDs overlap.")
    if set(training["group_id"]) & set(evaluation["group_id"]):
        raise ValueError("Training and verified evaluation group_ids overlap.")
    train_kept, removed = _remove_training_near_duplicates(training, evaluation)
    removed_path.parent.mkdir(parents=True, exist_ok=True)
    removed.to_csv(removed_path, index=False, encoding="utf-8")
    combined = pd.concat([train_kept, evaluation], ignore_index=True)
    if combined["id"].duplicated().any() or combined["group_id"].duplicated().any():
        raise ValueError("dataset_v1 requires globally unique ids and group_ids.")
    validate_dataset(combined, load_label_config(CONFIG_PATH))
    output_path.parent.mkdir(parents=True, exist_ok=True)
    combined.to_csv(output_path, index=False, encoding="utf-8")
    return combined


def _count_situations(data: pd.DataFrame, label: str) -> int:
    return sum(label in _parse_pipe_separated(value) for value in data["situation_labels"])


def _rule_line(name: str, actual: int, required: int) -> str:
    if actual >= required:
        return f"- PASS — {name}: {actual} (required {required})"
    return f"- FAIL — {name}: {actual} of {required} needed (shortfall {required - actual})"


def write_readiness_report(
    data: pd.DataFrame,
    report_path: Path = READINESS_PATH,
) -> str:
    assignments = _allocate_group_splits(data, ensure_test_coverage=True)
    preview = data.copy()
    preview["split"] = preview["group_id"].map(assignments)
    config = load_label_config(CONFIG_PATH)
    test = preview.loc[preview["split"] == "test"]
    train_generators = sorted(
        preview.loc[preview["split"] == "train", "generator"].astype(str).unique()
    )
    holdout_generators = sorted(
        preview.loc[
            preview["split"].isin({"val", "test"}), "generator"
        ].astype(str).unique()
    )
    overlap = sorted(set(train_generators) & set(holdout_generators))
    lines = [
        "# dataset_v1 readiness report",
        "",
        "This is a PREVIEW split only. No frozen dataset was created.",
        "",
        "## Rows by split, source, generator, verification, and language",
        "",
        "```text",
        preview.groupby(
            ["split", "source", "generator", "human_verified", "language"],
            dropna=False,
        ).size().to_string(),
        "```",
        "",
        "## Test urgency counts",
        "",
    ]
    for label in config["urgency_labels"]:
        lines.append(
            _rule_line(
                f"test {label}",
                int((test["urgency"] == label).sum()),
                MIN_TEST_EXAMPLES_PER_URGENCY,
            )
        )
    lines.extend(["", "## Test language counts", ""])
    for language in config["allowed_languages"]:
        lines.append(
            _rule_line(
                f"test {language}",
                int((test["language"] == language).sum()),
                MIN_TEST_EXAMPLES_PER_LANGUAGE,
            )
        )
    lines.extend(["", "## Test situation-label counts", ""])
    for label in config["situation_labels"]:
        lines.append(
            _rule_line(
                f"test {label}",
                _count_situations(test, label),
                MIN_TEST_EXAMPLES_PER_SITUATION,
            )
        )
    verified_count = int(_verified_mask(preview).sum())
    lines.extend(
        [
            "",
            "## Freeze rules",
            "",
            _rule_line("total rows", len(preview), MIN_ROWS_TO_FREEZE),
            _rule_line(
                "human_verified evaluation rows",
                verified_count,
                MIN_VERIFIED_EVAL_ROWS_TO_FREEZE,
            ),
            "",
            "## Generator isolation",
            "",
            f"- Train generators: {', '.join(train_generators) or '(none)'}",
            f"- Val/test generators: {', '.join(holdout_generators) or '(none)'}",
            (
                "- PASS — no generator overlap."
                if not overlap
                else f"- FAIL — overlapping generators: {', '.join(overlap)}"
            ),
            "",
            "## Exact freeze command (not run)",
            "",
            "```powershell",
            "python -m sakhi.src.data sakhi/data/raw/dataset_v1.csv --freeze",
            "```",
        ]
    )
    report = "\n".join(lines) + "\n"
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(report, encoding="utf-8")
    print(report)
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train", type=Path, default=TRAIN_PATH)
    parser.add_argument("--eval", type=Path, default=EVAL_PATH)
    parser.add_argument("--output", type=Path, default=OUTPUT_PATH)
    parser.add_argument("--removed", type=Path, default=REMOVED_PATH)
    parser.add_argument("--report", type=Path, default=READINESS_PATH)
    args = parser.parse_args()
    combined = build_dataset(args.train, args.eval, args.output, args.removed)
    write_readiness_report(combined, args.report)


if __name__ == "__main__":
    main()
