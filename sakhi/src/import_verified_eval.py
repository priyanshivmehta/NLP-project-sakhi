"""Import a reviewed evaluation sheet and mark accepted rows as verified."""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from sakhi.src.data import CONFIG_PATH, DatasetValidationError, load_label_config, validate_dataset

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_INPUT = PROJECT_ROOT / "sakhi" / "data" / "raw" / "eval_verification_sheet.csv"
DEFAULT_CANDIDATES = PROJECT_ROOT / "sakhi" / "data" / "raw" / "eval_candidates.csv"
DEFAULT_OUTPUT = PROJECT_ROOT / "sakhi" / "data" / "raw" / "eval_verified.csv"


def import_reviewed_sheet(
    review_path: Path = DEFAULT_INPUT,
    candidates_path: Path = DEFAULT_CANDIDATES,
    output_path: Path = DEFAULT_OUTPUT,
) -> pd.DataFrame:
    review = pd.read_csv(review_path, encoding="utf-8").fillna("")
    candidates = pd.read_csv(candidates_path, encoding="utf-8")
    required = {"id", "verify_status", "corrected_labels", "corrected_urgency", "reviewer"}
    missing = required - set(review.columns)
    if missing:
        raise DatasetValidationError(
            f"Review sheet is missing columns: {', '.join(sorted(missing))}"
        )
    statuses = review["verify_status"].astype(str).str.strip().str.lower()
    accepted = review.loc[statuses.isin({"ok", "corrected"})].copy()
    rejected_ids = set(review.loc[statuses == "reject", "id"])
    unknown = sorted(set(statuses) - {"", "pending", "ok", "corrected", "reject"})
    if unknown:
        raise DatasetValidationError("Unknown verification statuses: " + ", ".join(unknown))
    result = candidates.loc[candidates["id"].isin(accepted["id"])].copy()
    result = result.merge(
        accepted[["id", "corrected_labels", "corrected_urgency"]],
        on="id",
        how="left",
    )
    result["situation_labels"] = result["corrected_labels"].where(
        result["corrected_labels"].astype(str).str.strip().ne(""),
        result["situation_labels"],
    )
    result["urgency"] = result["corrected_urgency"].where(
        result["corrected_urgency"].astype(str).str.strip().ne(""),
        result["urgency"],
    )
    result = result.drop(columns=["corrected_labels", "corrected_urgency"])
    result["human_verified"] = True
    validate_dataset(result, load_label_config(CONFIG_PATH))
    output_path.parent.mkdir(parents=True, exist_ok=True)
    result.to_csv(output_path, index=False, encoding="utf-8")
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--review", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--candidates", type=Path, default=DEFAULT_CANDIDATES)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    result = import_reviewed_sheet(args.review, args.candidates, args.output)
    print(f"Wrote {len(result)} verified rows to {args.output}")


if __name__ == "__main__":
    main()
