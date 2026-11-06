"""Create a deterministic noisy copy of verified evaluation rows."""

from __future__ import annotations

import argparse
import random
import re
from pathlib import Path

import pandas as pd
import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_INPUT = PROJECT_ROOT / "sakhi" / "data" / "raw" / "eval_verified.csv"
DEFAULT_OUTPUT = PROJECT_ROOT / "sakhi" / "data" / "raw" / "eval_verified_noisy.csv"
VARIANTS_PATH = PROJECT_ROOT / "sakhi" / "config" / "spelling_variants.yaml"


def _load_variants() -> dict[str, str]:
    with VARIANTS_PATH.open(encoding="utf-8") as handle:
        return yaml.safe_load(handle).get("variants", {})


def _noise_text(text: str, rng: random.Random, variants: dict[str, str]) -> str:
    result = text
    words = result.split()
    if words and rng.random() < 0.6:
        index = rng.randrange(len(words))
        replacement = variants.get(words[index].casefold())
        if replacement:
            words[index] = replacement
        elif len(words[index]) > 3:
            position = rng.randrange(1, len(words[index]) - 1)
            words[index] = words[index][:position] + words[index][position + 1 :]
    result = " ".join(words)
    if rng.random() < 0.45:
        result = re.sub(r"([!?.,])", "", result)
    if result and rng.random() < 0.35:
        index = rng.randrange(len(result))
        result = result[:index] + result[index] + result[index:]
    return result


def augment_verified_rows(
    input_path: Path = DEFAULT_INPUT,
    output_path: Path = DEFAULT_OUTPUT,
    seed: int = 459,
) -> pd.DataFrame:
    data = pd.read_csv(input_path, encoding="utf-8")
    if "human_verified" not in data or not data["human_verified"].astype(bool).all():
        raise ValueError("Noise augmentation requires only human_verified rows.")
    result = data.copy()
    rng = random.Random(seed)
    variants = _load_variants()
    result["text"] = result["text"].astype(str).map(
        lambda value: _noise_text(value, rng, variants)
    )
    result["noisy"] = True
    output_path.parent.mkdir(parents=True, exist_ok=True)
    result.to_csv(output_path, index=False, encoding="utf-8")
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--seed", type=int, default=459)
    args = parser.parse_args()
    result = augment_verified_rows(args.input, args.output, args.seed)
    print(f"Wrote {len(result)} noisy rows to {args.output}")


if __name__ == "__main__":
    main()
