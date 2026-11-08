"""Compare saved TF-IDF and embedding evaluation results."""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from sakhi.src.train_baseline import PROJECT_ROOT

DEFAULT_TFIDF = PROJECT_ROOT / "sakhi" / "reports" / "results"
DEFAULT_EMBEDDING = PROJECT_ROOT / "sakhi" / "reports" / "results_embedding"
DEFAULT_OUTPUT = PROJECT_ROOT / "sakhi" / "reports" / "results_v1" / "model_comparison"


def compare_models(
    tfidf_dir: Path = DEFAULT_TFIDF,
    embedding_dir: Path = DEFAULT_EMBEDDING,
    output_stem: Path = DEFAULT_OUTPUT,
    split_name: str = "val",
) -> pd.DataFrame:
    frames = []
    for model_name, folder in (("tfidf", tfidf_dir), ("embedding", embedding_dir)):
        path = folder / f"{split_name}_metrics.csv"
        if not path.exists():
            raise FileNotFoundError(f"Metrics file not found: {path}")
        frame = pd.read_csv(path)
        frame.insert(0, "model", model_name)
        frames.append(frame)
    result = pd.concat(frames, ignore_index=True)
    result = result.loc[
        result["metric_scope"].isin(
            {"macro", "micro", "per_class", "per_label", "per_language", "per_language_label"}
        ),
        ["model", "task", "metric_scope", "label", "f1", "support", "data_quality"],
    ]
    output_stem.parent.mkdir(parents=True, exist_ok=True)
    result.to_csv(output_stem.with_suffix(".csv"), index=False)
    headers = list(result.columns)
    markdown = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join("---" for _ in headers) + " |",
    ]
    markdown.extend(
        "| " + " | ".join(str(row[column]) for column in headers) + " |"
        for _, row in result.iterrows()
    )
    output_stem.with_suffix(".md").write_text(
        "\n".join(markdown) + "\n", encoding="utf-8"
    )
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tfidf-dir", type=Path, default=DEFAULT_TFIDF)
    parser.add_argument("--embedding-dir", type=Path, default=DEFAULT_EMBEDDING)
    parser.add_argument("--output-stem", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--split", default="val")
    args = parser.parse_args()
    result = compare_models(args.tfidf_dir, args.embedding_dir, args.output_stem, args.split)
    print(f"Wrote {len(result)} comparison rows to {args.output_stem}.csv and .md")


if __name__ == "__main__":
    main()
