"""Ingest user-supplied legal source files into section-level chunks.

This module never supplies legal text. It only parses files present under the
configured source directory and records sections it cannot parse.
"""

from __future__ import annotations

import argparse
import logging
import re
from pathlib import Path

import pandas as pd

LOGGER = logging.getLogger(__name__)
PROJECT_ROOT = Path(__file__).resolve().parents[2]
SOURCE_DIR = PROJECT_ROOT / "sakhi" / "data" / "legal" / "source"
CHUNKS_PATH = PROJECT_ROOT / "sakhi" / "data" / "legal" / "chunks" / "chunks.csv"
UNPARSED_PATH = PROJECT_ROOT / "sakhi" / "data" / "legal" / "unparsed_sections.txt"
OUTPUT_COLUMNS = [
    "chunk_id",
    "act",
    "section_no",
    "section_title",
    "text",
    "language",
    "source_file",
]

SECTION_RE = re.compile(
    r"(?im)^(?:#{1,6}\s*)?(?:section|धारा)\s+([0-9A-Za-z-]+)"
    r"(?:\s*[-:.]\s*|\s+)([^\r\n]+)?$|"
    r"^(?:#{1,6}\s*)?([0-9]+[A-Za-z]?)\s*[-:.]\s+([^\r\n]+)$"
)
ACT_RE = re.compile(r"(?im)^(?:act|अधिनियम)\s*[:\-]\s*(.+)$")
SUPPORTED = {".txt", ".md", ".markdown", ".csv"}


def _act_from_path(path: Path) -> str:
    stem = path.stem.upper().replace("_", "-").replace(" ", "-")
    aliases = {
        "BNS": "BNS",
        "IT": "IT",
        "PWDVA": "PWDVA-2005",
        "POSH": "POSH-2013",
        "DPA": "DPA-1961",
    }
    for prefix, act in aliases.items():
        if stem.startswith(prefix):
            return act
    return stem


def _read_source(path: Path) -> str:
    if path.suffix.lower() != ".csv":
        return path.read_text(encoding="utf-8-sig")
    frame = pd.read_csv(path, dtype=str, keep_default_na=False)
    if "text" not in frame.columns:
        raise ValueError("CSV source must contain a text column")
    return "\n".join(frame["text"].tolist())


def _parse_file(path: Path) -> tuple[list[dict[str, str]], list[str]]:
    text = _read_source(path)
    act_match = ACT_RE.search(text)
    act = act_match.group(1).strip() if act_match else _act_from_path(path)
    matches = list(SECTION_RE.finditer(text))
    if not matches:
        return [], [f"{path.name}: no section heading found"]
    chunks: list[dict[str, str]] = []
    unparsed: list[str] = []
    for index, match in enumerate(matches):
        section_no = match.group(1).strip()
        title = (match.group(2) or "").strip()
        start = match.end()
        end = matches[index + 1].start() if index + 1 < len(matches) else len(text)
        body = text[start:end].strip()
        if not body:
            unparsed.append(f"{path.name}: section {section_no}: empty section body")
            continue
        prefix = "S" if act in {"PWDVA-2005", "POSH-2013", "DPA-1961"} else ""
        chunk_id = f"{act}-{prefix}{section_no}" if prefix else f"{act}-{section_no}"
        language = "hi" if len(re.findall(r"[\u0900-\u097F]", body)) > len(
            re.findall(r"[A-Za-z]", body)
        ) else "en"
        chunks.append(
            {
                "chunk_id": chunk_id,
                "act": act,
                "section_no": section_no,
                "section_title": title,
                "text": body,
                "language": language,
                "source_file": path.name,
            }
        )
    return chunks, unparsed


def ingest_sources(
    source_dir: Path = SOURCE_DIR,
    chunks_path: Path = CHUNKS_PATH,
    unparsed_path: Path = UNPARSED_PATH,
) -> pd.DataFrame:
    """Parse all supported source files and write chunks and parse failures."""
    chunks: list[dict[str, str]] = []
    unparsed: list[str] = []
    for path in sorted(source_dir.glob("*")):
        if not path.is_file() or path.suffix.lower() not in SUPPORTED:
            continue
        try:
            file_chunks, file_unparsed = _parse_file(path)
        except (OSError, UnicodeError, ValueError) as error:
            file_chunks, file_unparsed = [], [f"{path.name}: {error}"]
        chunks.extend(file_chunks)
        unparsed.extend(file_unparsed)
    chunks_path.parent.mkdir(parents=True, exist_ok=True)
    frame = pd.DataFrame(chunks, columns=OUTPUT_COLUMNS)
    frame.to_csv(chunks_path, index=False, encoding="utf-8")
    unparsed_path.parent.mkdir(parents=True, exist_ok=True)
    unparsed_path.write_text("\n".join(unparsed) + ("\n" if unparsed else ""), encoding="utf-8")
    for act, count in frame["act"].value_counts().items():
        LOGGER.info("Ingested %d chunks for %s", count, act)
    LOGGER.info("Ingested %d legal chunks total", len(frame))
    return frame


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-dir", type=Path, default=SOURCE_DIR)
    parser.add_argument("--chunks", type=Path, default=CHUNKS_PATH)
    parser.add_argument("--unparsed", type=Path, default=UNPARSED_PATH)
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    ingest_sources(args.source_dir, args.chunks, args.unparsed)


if __name__ == "__main__":
    main()
