"""Exploratory data analysis and staged word-cloud generation."""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import yaml
from wordcloud import WordCloud

from sakhi.src.preprocess import preprocess_dataframe

LOGGER = logging.getLogger(__name__)
PROJECT_ROOT = Path(__file__).resolve().parents[2]
FIGURES_DIR = PROJECT_ROOT / "sakhi" / "reports" / "figures"
PREPROCESS_CONFIG = PROJECT_ROOT / "sakhi" / "config" / "preprocess.yaml"


def _font_path(config_path: Path = PREPROCESS_CONFIG) -> Path:
    with config_path.open(encoding="utf-8") as config_file:
        config = yaml.safe_load(config_file) or {}
    configured = Path(config["font_path"])
    return configured if configured.is_absolute() else PROJECT_ROOT / "sakhi" / configured


def _save_barplot(series: pd.Series, title: str, path: Path) -> None:
    plt.figure(figsize=(10, 5))
    sns.barplot(x=series.index.astype(str), y=series.values)
    plt.title(title)
    plt.xticks(rotation=30, ha="right")
    plt.tight_layout()
    plt.savefig(path, dpi=150)
    plt.close()


def _save_wordcloud(text: str, title: str, path: Path, font_path: Path) -> None:
    if not font_path.is_file():
        raise FileNotFoundError(
            f"Word-cloud font not found at {font_path}. Download the open-licensed "
            "Noto Sans Devanagari font from https://fonts.google.com/specimen/Noto+Sans+Devanagari"
        )
    cloud = WordCloud(
        width=1400,
        height=800,
        background_color="white",
        font_path=str(font_path),
    ).generate(text or "no text")
    plt.figure(figsize=(12, 7))
    plt.imshow(cloud, interpolation="bilinear")
    plt.axis("off")
    plt.title(title)
    plt.tight_layout()
    plt.savefig(path, dpi=150)
    plt.close()


def run_eda(
    csv_path: Path,
    output_dir: Path = FIGURES_DIR,
    font_path: Path | None = None,
) -> None:
    """Generate counts, length plots, and raw/final/category word clouds."""
    data = pd.read_csv(csv_path, encoding="utf-8")
    staged = preprocess_dataframe(data)
    output_dir.mkdir(parents=True, exist_ok=True)

    situation_counts = (
        staged["situation_labels"].fillna("").str.split("|").explode().str.strip().value_counts()
    )
    _save_barplot(situation_counts, "Situation label counts", output_dir / "situation_counts.png")
    _save_barplot(staged["urgency"].value_counts(), "Urgency counts", output_dir / "urgency_counts.png")
    _save_barplot(staged["language"].value_counts(), "Language mix", output_dir / "language_mix.png")
    _save_barplot(staged["perspective"].value_counts(), "Perspective mix", output_dir / "perspective_mix.png")

    lengths = staged["raw"].str.len()
    plt.figure(figsize=(10, 5))
    sns.histplot(lengths, bins=20)
    plt.title("Raw text-length distribution")
    plt.xlabel("Characters")
    plt.tight_layout()
    plt.savefig(output_dir / "text_length_distribution.png", dpi=150)
    plt.close()

    configured_font = font_path or _font_path()
    _save_wordcloud(" ".join(staged["raw"]), "Raw text", output_dir / "wordcloud_raw.png", configured_font)
    _save_wordcloud(
        " ".join(staged["final_model_input"]),
        "Final model input",
        output_dir / "wordcloud_final_model_input.png",
        configured_font,
    )
    for label in situation_counts.index:
        category_text = " ".join(
            staged.loc[
                staged["situation_labels"].fillna("").str.split("|").apply(
                    lambda labels: label in [item.strip() for item in labels]
                ),
                "raw",
            ]
        )
        safe_label = str(label).replace("/", "_")
        _save_wordcloud(
            category_text,
            f"Situation: {label}",
            output_dir / f"wordcloud_situation_{safe_label}.png",
            configured_font,
        )


def main() -> None:
    """Run EDA from a CSV path."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("csv_path", type=Path)
    parser.add_argument("--output-dir", type=Path, default=FIGURES_DIR)
    parser.add_argument("--font-path", type=Path)
    args = parser.parse_args()
    run_eda(args.csv_path, args.output_dir, args.font_path)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    main()
