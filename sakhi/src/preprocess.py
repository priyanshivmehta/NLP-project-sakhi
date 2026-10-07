"""Pure, staged preprocessing for Hindi, Hinglish, and English text."""

from __future__ import annotations

import argparse
import logging
import re
import unicodedata
from pathlib import Path

import pandas as pd
import yaml

LOGGER = logging.getLogger(__name__)
PROJECT_ROOT = Path(__file__).resolve().parents[2]
SPELLING_CONFIG_PATH = PROJECT_ROOT / "sakhi" / "config" / "spelling_variants.yaml"
PROCESSED_DIR = PROJECT_ROOT / "sakhi" / "data" / "processed"
SNAPSHOT_NAME = "preprocessing_snapshot.csv"

URL_RE = re.compile(r"https?://\S+|www\.\S+", re.IGNORECASE)
EMAIL_RE = re.compile(r"\b[\w.+-]+@[\w-]+\.[\w.-]+\b")
PHONE_RE = re.compile(
    r"(?<!\d)(?:\+?91[-.\s]?)?[6-9]\d{4}[-.\s]?\d{5}(?!\d)"
)
LONG_DIGIT_RE = re.compile(r"(?<!\d)\d{6,}(?!\d)")
EMOJI_RE = re.compile(
    r"[\U0001F1E6-\U0001F1FF\U0001F300-\U0001FAFF\u2600-\u27BF]"
)
REPEATED_CHAR_RE = re.compile(r"([A-Za-z])\1{2,}", re.IGNORECASE)
TOKEN_SEPARATOR_RE = re.compile(r"\s+")


def raw_stage(text: str) -> str:
    """Return text unchanged."""
    return text


def clean_text(text: str) -> str:
    """Apply NFC Unicode normalization, URL removal, whitespace, and emoji tokens."""
    normalized = unicodedata.normalize("NFC", text)
    without_urls = URL_RE.sub(" [URL] ", normalized)
    with_emoji_tokens = EMOJI_RE.sub(" [EMOJI] ", without_urls)
    return TOKEN_SEPARATOR_RE.sub(" ", with_emoji_tokens).strip()


def _load_spelling_variants(path: Path = SPELLING_CONFIG_PATH) -> dict[str, str]:
    with path.open(encoding="utf-8") as config_file:
        config = yaml.safe_load(config_file) or {}
    variants = config.get("variants", {})
    if not isinstance(variants, dict):
        raise ValueError(f"Expected a mapping under variants in {path}")
    return {str(key).casefold(): str(value) for key, value in variants.items()}


def normalize_text(
    text: str,
    language: str,
    spelling_variants: dict[str, str] | None = None,
) -> str:
    """Normalize Roman-script text while retaining Devanagari characters."""
    if language not in {"en", "hinglish"}:
        return text
    variants = spelling_variants or _load_spelling_variants()
    lowered = REPEATED_CHAR_RE.sub(r"\1\1", text.lower())
    tokens = lowered.split()
    normalized_tokens = [variants.get(token, token) for token in tokens]
    return " ".join(normalized_tokens)


def mask_entities(text: str) -> str:
    """Mask contact details without altering short emergency helpline numbers."""
    masked = EMAIL_RE.sub(" [EMAIL] ", text)
    masked = PHONE_RE.sub(" [PHONE] ", masked)
    masked = LONG_DIGIT_RE.sub(" [LONG_NUMBER] ", masked)
    return TOKEN_SEPARATOR_RE.sub(" ", masked).strip()


def final_model_input(text: str) -> str:
    """Return the entity-masked stage as the model input."""
    return text


def preprocess_text(
    text: str,
    language: str,
    spelling_variants: dict[str, str] | None = None,
) -> dict[str, str]:
    """Run every stage for one text value."""
    raw = raw_stage(text)
    cleaned = clean_text(raw)
    normalized = normalize_text(cleaned, language, spelling_variants)
    masked = mask_entities(normalized)
    return {
        "raw": raw,
        "cleaning": cleaned,
        "normalisation": normalized,
        "entity_masking": masked,
        "final_model_input": final_model_input(masked),
    }


def preprocess_dataframe(
    data: pd.DataFrame,
    text_column: str = "text",
    language_column: str = "language",
    output_path: Path | None = None,
) -> pd.DataFrame:
    """Add all stage columns and optionally save a CSV snapshot."""
    if text_column not in data or language_column not in data:
        raise ValueError(f"DataFrame must contain {text_column!r} and {language_column!r}.")
    variants = _load_spelling_variants()
    rows = [
        preprocess_text(str(text), str(language), variants)
        for text, language in zip(data[text_column], data[language_column])
    ]
    stages = pd.DataFrame(rows, index=data.index)
    snapshot = data.copy()
    for column in stages.columns:
        snapshot[column] = stages[column]
    if output_path is not None:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        snapshot.to_csv(output_path, index=False, encoding="utf-8")
    return snapshot


def main() -> None:
    """Run preprocessing from a CSV and save a staged snapshot."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("csv_path", type=Path)
    parser.add_argument("--output", type=Path, default=PROCESSED_DIR / SNAPSHOT_NAME)
    args = parser.parse_args()
    data = pd.read_csv(args.csv_path, encoding="utf-8")
    preprocess_dataframe(data, output_path=args.output)
    LOGGER.info("Saved preprocessing snapshot to %s", args.output)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    main()
