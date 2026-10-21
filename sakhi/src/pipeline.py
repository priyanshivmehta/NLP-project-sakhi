"""Single-message inference for Sakhi: analyze() is what the Streamlit app calls.

Loads the baseline artifacts written by train_baseline.py (from sakhi/models/,
or the folder in the SAKHI_MODELS_DIR environment variable) and returns:

    {
      "situations": [{"label", "prob", "predicted"}, ...],   # sorted by prob
      "urgency":    {"label", "probs": {level: prob}},
      "provisions": [{"act", "section", "text", "score"}, ...],
      "meta":       {"language_detected", "retrieval"},
    }

Privacy: the message is processed in memory only. This module never logs,
prints, or saves user text.

CLI helpers:
    python -m sakhi.src.pipeline "some message"        # try one message
    python -m sakhi.src.pipeline --tune-112            # recall/false-alarm table on val.csv
"""

from __future__ import annotations

import argparse
import importlib
import json
import os
import re
from functools import lru_cache
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd

from sakhi.src.preprocess import _load_spelling_variants, preprocess_text
from sakhi.src.train_baseline import (
    MODELS_DIR,
    PROCESSED_DIR,
    BaselineArtifacts,
    _model_text,
)

DEVANAGARI_RE = re.compile(r"[\u0900-\u097F]")
LATIN_RE = re.compile(r"[A-Za-z]")
TOKEN_RE = re.compile(r"[a-z]+")
# Romanised-Hindi words that are not ordinary English words.
HINGLISH_MARKERS = {
    "hai", "hain", "nahi", "nhi", "nai", "nahin", "mujhe", "mera", "meri", "mere", "uska", "uski",
    "aur", "kya", "kyu", "kyun", "kaise", "raha", "rahi", "rahe", "karta", "karti", "karte", "karna",
    "karu", "wo", "woh", "yeh", "abhi", "roz", "pati", "ghar", "bahut", "bhi", "koi", "kuch",
    "ladka", "aadmi", "peecha", "peeche", "dhamki", "paise", "sasural", "behen", "dost", "bola",
    "bol", "tha", "thi", "hu", "hoon", "ne", "se", "ko", "ka", "ki", "ke", "mat", "kal", "aaj",
}


def models_dir() -> Path:
    return Path(os.environ.get("SAKHI_MODELS_DIR", MODELS_DIR))


def models_available() -> bool:
    needed = ("tfidf_features.joblib", "urgency_model.joblib", "situation_model.joblib",
              "situation_thresholds.json", "model_metadata.json")
    return all((models_dir() / name).is_file() for name in needed)


@lru_cache(maxsize=2)
def load_artifacts(directory: str | None = None) -> BaselineArtifacts:
    folder = Path(directory) if directory else models_dir()
    if not models_available() and not directory:
        raise FileNotFoundError(
            f"Trained models not found in {folder}. Run: python -m sakhi.src.train_baseline"
        )
    return BaselineArtifacts(
        vectorizer=joblib.load(folder / "tfidf_features.joblib"),
        urgency_model=joblib.load(folder / "urgency_model.joblib"),
        situation_model=joblib.load(folder / "situation_model.joblib"),
        situation_labels=json.loads((folder / "model_metadata.json").read_text(encoding="utf-8"))[
            "situation_labels"
        ],
        thresholds=json.loads((folder / "situation_thresholds.json").read_text(encoding="utf-8")),
    )


@lru_cache(maxsize=1)
def _spelling_variants() -> dict[str, str]:
    return _load_spelling_variants()


def detect_language(text: str) -> str:
    """Heuristic: 'hi' (mostly Devanagari), 'hinglish' (Roman Hindi), else 'en'."""
    dev = len(DEVANAGARI_RE.findall(text))
    latin = len(LATIN_RE.findall(text))
    if dev and dev / (dev + latin) >= 0.5:
        return "hi"
    tokens = TOKEN_RE.findall(text.lower())
    hits = sum(token in HINGLISH_MARKERS for token in tokens)
    if hits >= 2 or (tokens and hits / len(tokens) >= 0.2):
        return "hinglish"
    return "en"


def _retrieve(text: str, k: int) -> tuple[list[dict[str, Any]], str]:
    """Use sakhi/src/retrieval.py's retrieve(text, k) if it exists; else return nothing."""
    try:
        module = importlib.import_module("sakhi.src.retrieval")
    except ModuleNotFoundError as error:
        if error.name == "sakhi.src.retrieval":
            return [], "not connected"
        raise
    return list(module.retrieve(text, k=k)), "connected"


def analyze(text: str, top_k_provisions: int = 3) -> dict[str, Any]:
    """Classify one message and return situation, urgency and retrieved provisions."""
    artifacts = load_artifacts()
    language = detect_language(text)
    model_input = preprocess_text(text, language, _spelling_variants())["final_model_input"]
    features = artifacts.vectorizer.transform([model_input])

    urgency_probs = artifacts.urgency_model.predict_proba(features)[0]
    urgency = {str(c): float(p) for c, p in zip(artifacts.urgency_model.classes_, urgency_probs)}
    urgency_label = max(urgency, key=urgency.get)

    situation_probs = artifacts.situation_model.predict_proba(features)[0]
    situations = [
        {
            "label": label,
            "prob": float(situation_probs[i]),
            "predicted": bool(situation_probs[i] >= artifacts.thresholds[label]),
        }
        for i, label in enumerate(artifacts.situation_labels)
    ]
    situations.sort(key=lambda s: -s["prob"])
    if not any(s["predicted"] for s in situations):
        # Every training row has at least one label, so always keep the top one.
        situations[0]["predicted"] = True

    provisions, retrieval_state = _retrieve(text, top_k_provisions)
    return {
        "situations": situations,
        "urgency": {"label": urgency_label, "probs": urgency},
        "provisions": provisions,
        "meta": {"language_detected": language, "retrieval": retrieval_state},
    }


def report_112_thresholds(val_path: Path) -> pd.DataFrame:
    """Show recall and false-alarm rate for 'show Call 112' at P(immediate_danger) >= t."""
    artifacts = load_artifacts()
    data = pd.read_csv(val_path, encoding="utf-8")
    probs = artifacts.urgency_model.predict_proba(artifacts.vectorizer.transform(_model_text(data)))
    column = list(artifacts.urgency_model.classes_).index("immediate_danger")
    p_danger = probs[:, column]
    truth = (data["urgency"] == "immediate_danger").to_numpy()
    rows = []
    for threshold in np.arange(0.05, 0.95, 0.05):
        flagged = p_danger >= threshold
        rows.append(
            {
                "threshold": round(float(threshold), 2),
                "recall_immediate_danger": round(float((flagged & truth).sum() / max(truth.sum(), 1)), 3),
                "false_alarm_rate": round(float((flagged & ~truth).sum() / max((~truth).sum(), 1)), 3),
                "flagged_share": round(float(flagged.mean()), 3),
            }
        )
    return pd.DataFrame(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("message", nargs="?")
    parser.add_argument("--tune-112", action="store_true")
    parser.add_argument("--val", type=Path, default=PROCESSED_DIR / "val.csv")
    args = parser.parse_args()
    if args.tune_112:
        print(report_112_thresholds(args.val).to_string(index=False))
    elif args.message:
        print(json.dumps(analyze(args.message), indent=2, ensure_ascii=False))
    else:
        parser.print_help()


if __name__ == "__main__":
    main()