"""Audit dataset duplication, phrase patterns, and label-shuffle sanity."""

from __future__ import annotations

import argparse
import logging
import sys
from collections import Counter
from pathlib import Path

import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score
from sklearn.model_selection import train_test_split
from sklearn.metrics.pairwise import cosine_similarity

LOGGER = logging.getLogger(__name__)


def _normalise(text: str) -> str:
    return " ".join(str(text).casefold().split())


def find_duplicate_rows(data: pd.DataFrame, threshold: float = 0.85) -> list[dict[str, object]]:
    """Return exact and TF-IDF cosine near-duplicate pairs."""
    texts = data["text"].map(_normalise)
    matrix = TfidfVectorizer().fit_transform(texts)
    similarities = cosine_similarity(matrix)
    findings: list[dict[str, object]] = []
    for left in range(len(data)):
        for right in range(left + 1, len(data)):
            score = 1.0 if texts.iloc[left] == texts.iloc[right] else float(
                similarities[left, right]
            )
            if score >= threshold:
                findings.append(
                    {
                        "left_id": data.iloc[left]["id"],
                        "right_id": data.iloc[right]["id"],
                        "similarity": round(score, 4),
                    }
                )
    return findings


def common_openings(data: pd.DataFrame, count: int = 20) -> list[tuple[str, int]]:
    """Return the most common first three-token sentence openings."""
    openings = [
        " ".join(_normalise(text).split()[:3])
        for text in data["text"]
        if _normalise(text)
    ]
    return Counter(openings).most_common(count)


def common_fourgrams(data: pd.DataFrame, count: int = 20) -> list[tuple[str, int]]:
    """Return the most common word four-grams."""
    grams: Counter[str] = Counter()
    for text in data["text"]:
        tokens = _normalise(text).split()
        grams.update(" ".join(tokens[index : index + 4]) for index in range(len(tokens) - 3))
    return grams.most_common(count)


def shuffled_label_sanity_check(data: pd.DataFrame, random_state: int = 459) -> float:
    """Train on shuffled urgency labels and warn when macro-F1 is suspiciously high."""
    if len(data) < 10 or data["urgency"].nunique() < 2:
        return 0.0
    train, test = train_test_split(data, test_size=0.25, random_state=random_state)
    shuffled = train["urgency"].sample(frac=1, random_state=random_state).to_numpy()
    vectorizer = TfidfVectorizer(ngram_range=(1, 2), min_df=1)
    train_features = vectorizer.fit_transform(train["text"].astype(str))
    test_features = vectorizer.transform(test["text"].astype(str))
    model = LogisticRegression(max_iter=1000, random_state=random_state)
    model.fit(train_features, shuffled)
    score = float(
        f1_score(
            test["urgency"],
            model.predict(test_features),
            labels=sorted(data["urgency"].unique()),
            average="macro",
            zero_division=0,
        )
    )
    if score > 0.4:
        LOGGER.warning(
            "Shuffled-label sanity macro-F1 %.3f exceeds 0.4; inspect for leakage.",
            score,
        )
    return score


def audit_dataset(csv_path: Path) -> dict[str, object]:
    """Run all audits and return serialisable results."""
    data = pd.read_csv(csv_path, encoding="utf-8")
    return {
        "duplicate_rows": find_duplicate_rows(data),
        "common_openings": common_openings(data),
        "common_fourgrams": common_fourgrams(data),
        "shuffled_label_macro_f1": shuffled_label_sanity_check(data),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("csv_path", type=Path)
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    sys.stdout.reconfigure(encoding="utf-8")
    result = audit_dataset(args.csv_path)
    for key, value in result.items():
        print(f"{key}: {value}")


if __name__ == "__main__":
    main()
