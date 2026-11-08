"""Record file-level metadata for a reviewed evaluation sheet."""

from __future__ import annotations

import argparse
import json
from datetime import date
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_INPUT = PROJECT_ROOT / "sakhi" / "data" / "raw" / "eval_verification_reviewed.csv"
DEFAULT_OUTPUT = PROJECT_ROOT / "sakhi" / "reports" / "audit" / "review_summary.json"
EXPECTED_STATUSES = {"ok", "corrected", "reject"}


def record_review_metadata(
    reviewers: str,
    method: str,
    review_path: Path = DEFAULT_INPUT,
    output_path: Path = DEFAULT_OUTPUT,
) -> dict[str, object]:
    """Write review counts and file-level reviewer metadata to JSON."""
    review = pd.read_csv(review_path, encoding="utf-8")
    if "verify_status" not in review.columns:
        raise ValueError(f"Review file is missing required column: verify_status ({review_path})")

    statuses = review["verify_status"].fillna("").astype(str).str.strip().str.lower()
    unknown = sorted(set(statuses) - EXPECTED_STATUSES)
    if unknown:
        raise ValueError("Unknown verification statuses: " + ", ".join(unknown))

    reviewer_list = [name.strip() for name in reviewers.split(",") if name.strip()]
    summary: dict[str, object] = {
        "reviewers": reviewer_list,
        "date": date.today().isoformat(),
        "total_rows": int(len(review)),
        "ok": int((statuses == "ok").sum()),
        "corrected": int((statuses == "corrected").sum()),
        "rejected": int((statuses == "reject").sum()),
        "method": method,
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reviewers", required=True, help='Comma-separated names, e.g. "Name1,Name2"')
    parser.add_argument("--method", required=True, help="Review method text to record exactly")
    parser.add_argument("--review", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    summary = record_review_metadata(args.reviewers, args.method, args.review, args.output)
    print(f"Wrote review summary for {summary['total_rows']} rows to {args.output}")


if __name__ == "__main__":
    main()
