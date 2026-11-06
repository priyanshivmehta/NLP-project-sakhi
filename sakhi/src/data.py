"""Dataset loading, validation, and reproducible splitting utilities."""

from __future__ import annotations

import argparse
import logging
from difflib import SequenceMatcher
from pathlib import Path
from typing import Any

import pandas as pd
import yaml
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.metrics.pairwise import cosine_similarity

LOGGER = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parents[2]
CONFIG_PATH = PROJECT_ROOT / "sakhi" / "config" / "labels.yaml"
PROCESSED_DIR = PROJECT_ROOT / "sakhi" / "data" / "processed"
RANDOM_SEED = 459
TRAIN_FRACTION = 0.70
VALIDATION_FRACTION = 0.15
TEST_FRACTION = 0.15
MIN_ROWS_TO_FREEZE = 400
MIN_VERIFIED_EVAL_ROWS_TO_FREEZE = 150
MIN_TEST_EXAMPLES_PER_URGENCY = 5
MIN_TEST_EXAMPLES_PER_SITUATION = 3
MIN_TEST_EXAMPLES_PER_LANGUAGE = 30
AI_NEAR_DUPLICATE_THRESHOLD = 0.85
AI_SOURCE = "ai_generated"
HUMAN_SOURCES = frozenset({"team_written", "peer_written", "team_authored"})
LEGACY_TEST_SOURCES = frozenset({"team_authored", "synthetic_test"})
HUMAN_HOLDOUT_SOURCES = HUMAN_SOURCES | LEGACY_TEST_SOURCES
REQUIRED_COLUMNS = (
    "id",
    "text",
    "language",
    "perspective",
    "situation_labels",
    "urgency",
    "legal_refs",
    "source",
    "annotator",
    "split",
    "group_id",
    "generator",
    "human_verified",
)
FROZEN_DATASET_NAME = "frozen_dataset.csv"


class DatasetValidationError(ValueError):
    """Raised when one or more dataset rows or columns are invalid."""


def find_cross_split_text_duplicates(
    data: pd.DataFrame,
    near_duplicate_threshold: float = 0.92,
) -> list[dict[str, object]]:
    """Return exact or near-identical text pairs assigned to different splits."""
    normalized = (
        data["text"].astype(str).str.casefold().str.replace(r"\s+", " ", regex=True).str.strip()
    )
    findings: list[dict[str, object]] = []
    for left_index in range(len(data)):
        for right_index in range(left_index + 1, len(data)):
            if data.iloc[left_index]["split"] == data.iloc[right_index]["split"]:
                continue
            score = SequenceMatcher(None, normalized.iloc[left_index], normalized.iloc[right_index]).ratio()
            if score >= near_duplicate_threshold:
                findings.append(
                    {
                        "left_id": data.iloc[left_index]["id"],
                        "right_id": data.iloc[right_index]["id"],
                        "left_split": data.iloc[left_index]["split"],
                        "right_split": data.iloc[right_index]["split"],
                        "similarity": round(score, 4),
                    }
                )
    return findings


def _remove_ai_near_duplicates(
    ai_data: pd.DataFrame,
    human_holdout: pd.DataFrame,
    threshold: float = AI_NEAR_DUPLICATE_THRESHOLD,
) -> pd.DataFrame:
    """Drop synthetic rows too similar to a human validation/test row."""
    if ai_data.empty or human_holdout.empty:
        return ai_data.copy()
    texts = pd.concat(
        [human_holdout["text"].astype(str), ai_data["text"].astype(str)],
        ignore_index=True,
    )
    matrix = TfidfVectorizer().fit_transform(texts)
    similarities = cosine_similarity(matrix[len(human_holdout) :], matrix[: len(human_holdout)])
    keep = similarities.max(axis=1) <= threshold
    return ai_data.loc[keep].copy()


def validate_test_coverage(
    data: pd.DataFrame,
    labels_config: dict[str, list[str]] | None = None,
) -> None:
    """Require minimum configured urgency and situation counts in test."""
    config = labels_config or load_label_config()
    if set(data["source"].dropna()) == {"synthetic_test"}:
        return
    test_data = data.loc[
        (data["split"] == "test") & _verified_mask(data)
    ]
    urgency_counts = test_data["urgency"].value_counts()
    insufficient_urgency = {
        label: int(urgency_counts.get(label, 0))
        for label in config["urgency_labels"]
        if urgency_counts.get(label, 0) < MIN_TEST_EXAMPLES_PER_URGENCY
    }
    situation_counts = {
        label: sum(
            label in _parse_pipe_separated(value)
            for value in test_data["situation_labels"]
        )
        for label in config["situation_labels"]
    }
    insufficient_situations = {
        label: count
        for label, count in situation_counts.items()
        if count < MIN_TEST_EXAMPLES_PER_SITUATION
    }
    language_counts = test_data["language"].value_counts()
    insufficient_languages = {
        language: int(language_counts.get(language, 0))
        for language in config["allowed_languages"]
        if language_counts.get(language, 0) < MIN_TEST_EXAMPLES_PER_LANGUAGE
    }
    errors = []
    if insufficient_urgency:
        errors.append(
            "test split urgency counts below "
            f"{MIN_TEST_EXAMPLES_PER_URGENCY}: "
            + ", ".join(f"{label}={count}" for label, count in sorted(insufficient_urgency.items()))
        )
    if insufficient_situations:
        errors.append(
            "test split situation counts below "
            f"{MIN_TEST_EXAMPLES_PER_SITUATION}: "
            + ", ".join(
                f"{label}={count}" for label, count in sorted(insufficient_situations.items())
            )
        )
    if insufficient_languages:
        errors.append(
            "human test split language counts below "
            f"{MIN_TEST_EXAMPLES_PER_LANGUAGE}: "
            + ", ".join(
                f"{language}={count}"
                for language, count in sorted(insufficient_languages.items())
            )
        )
    if errors:
        raise DatasetValidationError("\n".join(errors))


def build_split_class_counts(
    data: pd.DataFrame,
    labels_config: dict[str, list[str]] | None = None,
) -> pd.DataFrame:
    """Return urgency and situation counts by split and provenance."""
    config = labels_config or load_label_config()
    rows: list[dict[str, object]] = []
    for split_name, split_data in data.groupby("split", sort=True):
        dimensions = [
            (source, generator, human_verified, language, subset)
            for source in sorted(split_data["source"].unique())
            for generator in sorted(split_data["generator"].astype(str).unique())
            for human_verified in sorted(split_data["human_verified"].astype(str).unique())
            for language in sorted(split_data["language"].unique())
            for subset in [
                split_data.loc[
                    (split_data["source"] == source)
                    & (split_data["generator"].astype(str) == generator)
                    & (split_data["human_verified"].astype(str) == human_verified)
                    & (split_data["language"] == language)
                ]
            ]
            if not subset.empty
        ]
        for source, generator, human_verified, language, subset in dimensions:
            urgency_counts = subset["urgency"].value_counts()
            for label in config["urgency_labels"]:
                rows.append(
                    {
                        "split": split_name,
                        "task": "urgency",
                        "label": label,
                        "source": source,
                        "generator": generator,
                        "human_verified": human_verified,
                        "language": language,
                        "count": int(urgency_counts.get(label, 0)),
                    }
                )
            situation_counts = {
                label: sum(
                    label in _parse_pipe_separated(value)
                    for value in subset["situation_labels"]
                )
                for label in config["situation_labels"]
            }
            rows.extend(
                {
                    "split": split_name,
                    "task": "situation",
                    "label": label,
                    "source": source,
                    "generator": generator,
                    "human_verified": human_verified,
                    "language": language,
                    "count": count,
                }
                for label, count in situation_counts.items()
            )
    return pd.DataFrame(rows)


def load_label_config(path: Path = CONFIG_PATH) -> dict[str, list[str]]:
    """Load the label configuration used by every data validation operation."""
    with path.open(encoding="utf-8") as config_file:
        config: Any = yaml.safe_load(config_file)
    if not isinstance(config, dict):
        raise ValueError(f"Label config must be a mapping: {path}")
    return config


def _parse_pipe_separated(value: object) -> list[str]:
    """Return non-empty pipe-separated values while preserving their order."""
    if pd.isna(value):
        return []
    return [part.strip() for part in str(value).split("|") if part.strip()]


def _verified_mask(data: pd.DataFrame) -> pd.Series:
    """Interpret CSV booleans consistently, including string values."""
    return data["human_verified"].map(
        lambda value: value
        if isinstance(value, bool)
        else str(value).strip().lower() == "true"
    )


def validate_dataset(
    data: pd.DataFrame,
    labels_config: dict[str, list[str]] | None = None,
) -> None:
    """Validate all rows and report every discovered error in one exception."""
    config = labels_config or load_label_config()
    errors: list[str] = []
    missing_columns = [column for column in REQUIRED_COLUMNS if column not in data]
    if missing_columns:
        errors.append(f"missing required columns: {', '.join(missing_columns)}")
        raise DatasetValidationError("; ".join(errors))

    invalid_verified = data["human_verified"].dropna().map(
        lambda value: (
            True
            if isinstance(value, bool)
            else str(value).strip().lower() in {"true", "false"}
        )
    )
    if not invalid_verified.all():
        errors.append("human_verified must contain only true or false")
    empty_generators = data.index[
        data["generator"].isna() | data["generator"].astype(str).str.strip().eq("")
    ]
    if len(empty_generators):
        errors.append(
            "empty generator at rows: " + ", ".join(map(str, empty_generators))
        )

    if data["id"].isna().any():
        errors.append("empty id at rows: " + ", ".join(map(str, data.index[data["id"].isna()])))
    duplicate_ids = data.loc[data["id"].duplicated(keep=False), "id"].dropna().unique()
    if len(duplicate_ids):
        errors.append("duplicate ids: " + ", ".join(map(str, duplicate_ids)))

    for column in ("text", "language", "perspective", "urgency", "group_id"):
        empty_rows = data.index[data[column].isna() | data[column].astype(str).str.strip().eq("")]
        if len(empty_rows):
            errors.append(f"empty {column} at rows: " + ", ".join(map(str, empty_rows)))

    allowed_languages = set(config["allowed_languages"])
    invalid_languages = sorted(set(data["language"].dropna()) - allowed_languages)
    if invalid_languages:
        errors.append(f"invalid languages: {', '.join(map(str, invalid_languages))}")

    allowed_perspectives = set(config["allowed_perspectives"])
    invalid_perspectives = sorted(set(data["perspective"].dropna()) - allowed_perspectives)
    if invalid_perspectives:
        errors.append(f"invalid perspectives: {', '.join(map(str, invalid_perspectives))}")

    allowed_situations = set(config["situation_labels"])
    invalid_situations = sorted(
        {
            label
            for value in data["situation_labels"]
            for label in _parse_pipe_separated(value)
            if label not in allowed_situations
        }
    )
    if invalid_situations:
        errors.append(f"invalid situation labels: {', '.join(invalid_situations)}")

    allowed_urgencies = set(config["urgency_labels"])
    invalid_urgencies = sorted(set(data["urgency"].dropna()) - allowed_urgencies)
    if invalid_urgencies:
        errors.append(f"invalid urgency labels: {', '.join(map(str, invalid_urgencies))}")

    allowed_sources = set(config.get("allowed_sources", ()))
    invalid_sources = sorted(
        set(data["source"].dropna()) - allowed_sources - LEGACY_TEST_SOURCES
    )
    if invalid_sources:
        errors.append(f"invalid sources: {', '.join(map(str, invalid_sources))}")

    if errors:
        raise DatasetValidationError("Dataset validation failed:\n- " + "\n- ".join(errors))


def load_dataset(
    csv_path: Path,
    labels_path: Path = CONFIG_PATH,
) -> pd.DataFrame:
    """Load a UTF-8 CSV and validate its schema and controlled vocabulary."""
    data = pd.read_csv(csv_path, encoding="utf-8")
    data = _add_legacy_provenance_defaults(data)
    validate_dataset(data, load_label_config(labels_path))
    return data


def _add_legacy_provenance_defaults(data: pd.DataFrame) -> pd.DataFrame:
    """Make old in-memory/test fixtures explicit without changing their text."""
    result = data.copy()
    if "generator" not in result:
        result["generator"] = [
            f"legacy_{row_id}" for row_id in result.get("id", result.index)
        ]
    if "human_verified" not in result:
        result["human_verified"] = result["source"].isin(HUMAN_HOLDOUT_SOURCES)
    return result


def _stratification_key(row: pd.Series) -> str:
    """Build a stable multi-label/urgency key for stratification."""
    situations = sorted(_parse_pipe_separated(row["situation_labels"]))
    return f"{row['urgency']}::{('|'.join(situations) or 'none')}"


def _allocate_group_splits(
    data: pd.DataFrame,
    ensure_test_coverage: bool = False,
) -> dict[str, str]:
    """Keep verified, generator-isolated groups in holdouts and AI in train."""
    preassigned_bad = data.loc[
        (data["source"] == AI_SOURCE)
        & ~_verified_mask(data)
        & data["split"].astype(str).isin({"val", "test"})
    ]
    if not preassigned_bad.empty:
        raise DatasetValidationError(
            "Unverified ai_generated rows cannot be assigned to val/test: "
            + ", ".join(map(str, preassigned_bad["id"].tolist()))
        )
    source_by_group = data.groupby("group_id")["source"].agg(
        lambda values: set(values)
    )
    mixed_groups = {
        group_id: sorted(sources)
        for group_id, sources in source_by_group.items()
        if AI_SOURCE in sources and any(source in HUMAN_SOURCES for source in sources)
    }
    if mixed_groups:
        raise DatasetValidationError(
            "ai_generated rows share group_id with human rows: "
            + ", ".join(str(group_id) for group_id in sorted(mixed_groups))
        )

    train_generators = set(
        data.loc[
            (data["source"] == AI_SOURCE) & ~_verified_mask(data),
            "generator",
        ].astype(str)
    )
    holdout_candidates = data.loc[
        _verified_mask(data)
        & ~data["generator"].astype(str).isin(train_generators)
    ]
    eligible_generators = set(holdout_candidates["generator"].astype(str))
    human_data = holdout_candidates.loc[
        holdout_candidates["generator"].astype(str).isin(eligible_generators)
    ]
    if human_data.empty:
        raise DatasetValidationError(
            "No human_verified evaluation rows are eligible for val/test: "
            "their generators are used by training rows."
        )
    groups = human_data.groupby("group_id", sort=True)
    group_data = groups.first()
    if len(group_data) < 3:
        raise ValueError("At least 3 rows are required to create train/val/test splits.")

    keys = group_data.apply(_stratification_key, axis=1)
    counts = keys.value_counts()
    if (counts < 2).any():
        LOGGER.warning(
            "Some multi-label strata contain one row; using urgency strata where possible."
        )
        keys = group_data["urgency"]

    minimum_test_size = (
        MIN_TEST_EXAMPLES_PER_LANGUAGE * 3
        if ensure_test_coverage
        else 1
    )
    if ensure_test_coverage:
        holdout_group_ids = group_data.index
        training_group_ids = data.loc[
            ~data["group_id"].isin(holdout_group_ids), "group_id"
        ].unique()
        test_size = max(
            1,
            round(len(holdout_group_ids) * TEST_FRACTION),
            minimum_test_size,
        )
        test_size = min(test_size, max(1, len(holdout_group_ids) - 1))
        try:
            validation, test = train_test_split(
                holdout_group_ids,
                test_size=test_size,
                random_state=RANDOM_SEED,
                stratify=keys if keys.nunique() < len(keys) else None,
            )
        except ValueError as error:
            LOGGER.warning(
                "Strict holdout stratification is not possible; using fallback: %s",
                error,
            )
            validation, test = train_test_split(
                holdout_group_ids,
                test_size=test_size,
                random_state=RANDOM_SEED,
            )
        assignments = {group_id: "train" for group_id in training_group_ids}
        assignments.update({group_id: "val" for group_id in validation})
        assignments.update({group_id: "test" for group_id in test})
        return assignments

    test_size = max(1, round(len(group_data) * TEST_FRACTION), minimum_test_size)
    test_size = min(test_size, max(1, len(group_data) - 2))
    validation_size = max(1, int(len(group_data) * VALIDATION_FRACTION))
    holdout_size = test_size + validation_size
    group_ids = group_data.index

    try:
        train, holdout = train_test_split(
            group_ids,
            test_size=holdout_size,
            random_state=RANDOM_SEED,
            stratify=keys,
        )
        holdout_keys = keys.loc[holdout]
        validation, test = train_test_split(
            holdout,
            test_size=test_size,
            random_state=RANDOM_SEED,
            stratify=holdout_keys if holdout_keys.nunique() < len(holdout_keys) else None,
        )
    except ValueError as error:
        LOGGER.warning(
            "Strict stratification is not possible for this dataset size; "
            "using a deterministic fallback allocation: %s",
            error,
        )
        train, holdout = train_test_split(
            group_ids,
            test_size=holdout_size,
            random_state=RANDOM_SEED,
        )
        validation, test = train_test_split(
            holdout,
            test_size=test_size,
            random_state=RANDOM_SEED,
        )
    assignments = {group_id: "train" for group_id in data["group_id"].unique()}
    assignments.update({group_id: "val" for group_id in validation})
    assignments.update({group_id: "test" for group_id in test})
    holdout_group_ids = set(validation) | set(test)
    holdout_generators = set(
        data.loc[data["group_id"].isin(holdout_group_ids), "generator"].astype(str)
    )
    for generator in holdout_generators:
        generator_groups = set(
            data.loc[data["generator"].astype(str) == generator, "group_id"]
        )
        if generator_groups - set(assignments):
            raise DatasetValidationError(
                f"Generator {generator} has rows without split assignments."
            )
        if any(assignments[group_id] == "train" for group_id in generator_groups):
            raise DatasetValidationError(
                f"Generator {generator} would be present in both train and holdout."
            )
    return assignments


def create_splits(
    data: pd.DataFrame,
    output_dir: Path = PROCESSED_DIR,
    source_path: Path | None = None,
    require_test_coverage: bool = True,
    freeze: bool = False,
) -> dict[str, pd.DataFrame]:
    """Preview or freeze one group-aware assignment and save derived split files.

    Freezing requires an explicit ``freeze=True`` and at least
    ``MIN_ROWS_TO_FREEZE`` rows. Preview runs never create or overwrite the frozen
    source of truth, so they are safe for throwaway copies during development.
    """
    fictional_ids = data["id"].astype(str).str.startswith("fictional-")
    if (source_path and source_path.name == "sample_fictional.csv") or fictional_ids.all():
        raise ValueError("sample_fictional.csv is only for tests, not production splits.")
    data = _add_legacy_provenance_defaults(data)
    validate_dataset(data)
    frozen_path = output_dir / FROZEN_DATASET_NAME
    if freeze and len(data) < MIN_ROWS_TO_FREEZE:
        raise ValueError(
            f"Refusing to freeze {len(data)} rows; provide the complete dataset "
            f"with at least {MIN_ROWS_TO_FREEZE} rows."
        )
    if freeze and frozen_path.exists():
        raise FileExistsError(
            f"Frozen split already exists at {frozen_path}; it must not be regenerated."
        )
    assignments = _allocate_group_splits(
        data,
        ensure_test_coverage=require_test_coverage or freeze,
    )
    frozen_data = data.copy()
    frozen_data["split"] = frozen_data["group_id"].map(assignments)
    if frozen_data["split"].isna().any():
        raise DatasetValidationError("Every group must receive exactly one split.")
    invalid_holdout = frozen_data.loc[
        frozen_data["split"].isin(("val", "test"))
        & (
            ~_verified_mask(frozen_data)
        )
    ]
    if not invalid_holdout.empty:
        raise DatasetValidationError(
            "val/test may contain only human_verified human rows: "
            + ", ".join(map(str, invalid_holdout["id"].tolist()))
        )
    train_generators = set(
        frozen_data.loc[frozen_data["split"] == "train", "generator"].astype(str)
    )
    holdout_generators = set(
        frozen_data.loc[
            frozen_data["split"].isin(("val", "test")), "generator"
        ].astype(str)
    )
    overlapping_generators = sorted(train_generators & holdout_generators)
    if overlapping_generators:
        raise DatasetValidationError(
            "Generators cannot occur in both train and val/test: "
            + ", ".join(overlapping_generators)
        )
    human_holdout = frozen_data.loc[frozen_data["split"].isin(("val", "test"))]
    ai_rows = frozen_data.loc[
        (frozen_data["source"] == AI_SOURCE) & (frozen_data["split"] == "train")
    ]
    kept_ai = _remove_ai_near_duplicates(ai_rows, human_holdout)
    dropped_ai_ids = set(ai_rows["id"]) - set(kept_ai["id"])
    if dropped_ai_ids:
        LOGGER.warning("Dropped %d AI rows near-duplicate to human holdout.", len(dropped_ai_ids))
        frozen_data = frozen_data.loc[~frozen_data["id"].isin(dropped_ai_ids)].copy()

    if freeze:
        human_count = int(_verified_mask(frozen_data).sum())
        if human_count < MIN_VERIFIED_EVAL_ROWS_TO_FREEZE:
            raise ValueError(
                f"Refusing to freeze {human_count} human_verified eval rows; at least "
                f"{MIN_VERIFIED_EVAL_ROWS_TO_FREEZE} are required."
            )

    splits = {
        "train": frozen_data.loc[frozen_data["split"] == "train"].reset_index(drop=True),
        "val": frozen_data.loc[frozen_data["split"] == "val"].reset_index(drop=True),
        "test": frozen_data.loc[frozen_data["split"] == "test"].reset_index(drop=True),
    }
    cross_split_duplicates = find_cross_split_text_duplicates(frozen_data)
    if cross_split_duplicates:
        LOGGER.warning(
            "Found %d exact or near-duplicate text pairs across splits: %s",
            len(cross_split_duplicates),
            cross_split_duplicates,
        )
    if require_test_coverage:
        validate_test_coverage(frozen_data)
    output_dir.mkdir(parents=True, exist_ok=True)
    for split_name, split_data in splits.items():
        split_data.to_csv(output_dir / f"{split_name}.csv", index=False, encoding="utf-8")
    build_split_class_counts(frozen_data).to_csv(
        output_dir / "split_class_counts.csv", index=False, encoding="utf-8"
    )
    if freeze:
        frozen_data.to_csv(frozen_path, index=False, encoding="utf-8")
    return splits


def main() -> None:
    """Load a dataset, validate it, and save reproducible split files."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("csv_path", type=Path)
    parser.add_argument("--output-dir", type=Path, default=PROCESSED_DIR)
    parser.add_argument(
        "--freeze",
        action="store_true",
        help="Create the permanent frozen dataset; only use after dataset and legal review are complete.",
    )
    args = parser.parse_args()
    data = load_dataset(args.csv_path)
    create_splits(data, args.output_dir, source_path=args.csv_path, freeze=args.freeze)
    LOGGER.info(
        "Saved %s split outputs to %s",
        "frozen" if args.freeze else "preview",
        args.output_dir,
    )


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    main()
