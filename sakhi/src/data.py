"""Dataset loading, validation, and reproducible splitting utilities."""

from __future__ import annotations

import argparse
import logging
from pathlib import Path
from typing import Any

import pandas as pd
import yaml
from sklearn.model_selection import train_test_split

LOGGER = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parents[2]
CONFIG_PATH = PROJECT_ROOT / "sakhi" / "config" / "labels.yaml"
PROCESSED_DIR = PROJECT_ROOT / "sakhi" / "data" / "processed"
RANDOM_SEED = 459
TRAIN_FRACTION = 0.70
VALIDATION_FRACTION = 0.15
TEST_FRACTION = 0.15
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
)


class DatasetValidationError(ValueError):
    """Raised when one or more dataset rows or columns are invalid."""


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

    if data["id"].isna().any():
        errors.append("empty id at rows: " + ", ".join(map(str, data.index[data["id"].isna()])))
    duplicate_ids = data.loc[data["id"].duplicated(keep=False), "id"].dropna().unique()
    if len(duplicate_ids):
        errors.append("duplicate ids: " + ", ".join(map(str, duplicate_ids)))

    for column in ("text", "language", "perspective", "urgency"):
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

    if errors:
        raise DatasetValidationError("Dataset validation failed:\n- " + "\n- ".join(errors))


def load_dataset(
    csv_path: Path,
    labels_path: Path = CONFIG_PATH,
) -> pd.DataFrame:
    """Load a UTF-8 CSV and validate its schema and controlled vocabulary."""
    data = pd.read_csv(csv_path, encoding="utf-8")
    validate_dataset(data, load_label_config(labels_path))
    return data


def _stratification_key(row: pd.Series) -> str:
    """Build a stable multi-label/urgency key for stratification."""
    situations = sorted(_parse_pipe_separated(row["situation_labels"]))
    return f"{row['urgency']}::{('|'.join(situations) or 'none')}"


def _allocate_stratified_indices(data: pd.DataFrame) -> tuple[list[int], list[int], list[int]]:
    """Allocate indices deterministically, stratifying by urgency and situations."""
    if len(data) < 3:
        raise ValueError("At least 3 rows are required to create train/val/test splits.")

    keys = data.apply(_stratification_key, axis=1)
    counts = keys.value_counts()
    if (counts < 2).any():
        LOGGER.warning(
            "Some multi-label strata contain one row; using urgency strata where possible."
        )
        keys = data["urgency"]

    test_size = max(1, round(len(data) * TEST_FRACTION))
    validation_size = max(1, int(len(data) * VALIDATION_FRACTION))
    holdout_size = test_size + validation_size

    try:
        train, holdout = train_test_split(
            data.index,
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
            data.index,
            test_size=holdout_size,
            random_state=RANDOM_SEED,
        )
        validation, test = train_test_split(
            holdout,
            test_size=test_size,
            random_state=RANDOM_SEED,
        )
    return list(train), list(validation), list(test)


def create_splits(
    data: pd.DataFrame,
    output_dir: Path = PROCESSED_DIR,
) -> dict[str, pd.DataFrame]:
    """Create and save deterministic 70/15/15 train, validation, and test CSVs."""
    validate_dataset(data)
    train_indices, validation_indices, test_indices = _allocate_stratified_indices(data)
    splits = {
        "train": data.loc[train_indices].sort_index().reset_index(drop=True),
        "val": data.loc[validation_indices].sort_index().reset_index(drop=True),
        "test": data.loc[test_indices].sort_index().reset_index(drop=True),
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    for split_name, split_data in splits.items():
        split_data.to_csv(output_dir / f"{split_name}.csv", index=False, encoding="utf-8")
    return splits


def main() -> None:
    """Load a dataset, validate it, and save reproducible split files."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("csv_path", type=Path)
    parser.add_argument("--output-dir", type=Path, default=PROCESSED_DIR)
    args = parser.parse_args()
    data = load_dataset(args.csv_path)
    create_splits(data, args.output_dir)
    LOGGER.info("Saved train/val/test splits to %s", args.output_dir)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    main()
