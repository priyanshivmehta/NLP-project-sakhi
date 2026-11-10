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
MANUAL_OVERRIDES_DIR = PROJECT_ROOT / "sakhi" / "data" / "legal" / "manual_overrides"
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
SECTION_PREFIX_RE = re.compile(
    r"^\s*(?:section\s+|धारा\s+)?([0-9]+[A-Za-z]?)\b", re.IGNORECASE
)


def _act_from_path(path: Path) -> str:
    stem = path.stem.upper().replace("_", "-").replace(" ", "-")
    aliases = {
        "BNS-2023": "BNS",
        "IT-2000": "IT",
        "PWDVA-2005": "PWDVA-2005",
        "POSH-2013": "POSH-2013",
        "DPA-1961": "DPA-1961",
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


def _section_title(opening_line: str, parsed_title: str) -> str:
    title = parsed_title.strip()
    if title and len(title) <= 80:
        return title
    return opening_line[:80]


def _parse_file(path: Path) -> tuple[list[dict[str, str]], list[str]]:
    text = _read_source(path)
    known_act = _act_from_path(path)
    act_match = ACT_RE.search(text)
    act = known_act if known_act in {"BNS", "IT", "PWDVA-2005", "POSH-2013", "DPA-1961"} else (
        act_match.group(1).strip() if act_match else known_act
    )
    matches = list(SECTION_RE.finditer(text))
    if not matches:
        return [], [f"{path.name}: no section heading found"]
    chunks: list[dict[str, str]] = []
    unparsed: list[str] = []
    for index, match in enumerate(matches):
        section_no = (match.group(1) or match.group(3) or "").strip()
        title = (match.group(2) or match.group(4) or "").strip()
        line_end = text.find("\n", match.start())
        if line_end < 0:
            line_end = len(text)
        opening_line = text[match.start():line_end].rstrip("\r")
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
                "section_title": _section_title(opening_line, title),
                "text": opening_line + (f"\n{body}" if body else ""),
                "language": language,
                "source_file": path.name,
            }
        )
    return chunks, unparsed


def validate_chunk_openings(chunks: list[dict[str, str]] | pd.DataFrame) -> list[str]:
    """Return chunk IDs whose text does not begin with their section number."""
    records = chunks.to_dict("records") if isinstance(chunks, pd.DataFrame) else chunks
    invalid: list[str] = []
    for chunk in records:
        match = SECTION_PREFIX_RE.match(str(chunk["text"]))
        if not match or match.group(1).casefold() != str(chunk["section_no"]).casefold():
            invalid.append(str(chunk["chunk_id"]))
    if invalid:
        print("Chunks missing their section-number opening (first 20):")
        print(", ".join(invalid[:20]))
    return invalid


def _parse_override(path: Path) -> dict[str, str]:
    act, separator, section_part = path.stem.rpartition("_")
    if not separator or not act or not section_part:
        raise ValueError(
            f"Override filename must be <act>_<section>.txt: {path.name}"
        )
    section_no = section_part.removeprefix("S")
    if not section_no:
        raise ValueError(f"Override filename has no section number: {path.name}")
    text = _read_source(path).strip()
    prefix = SECTION_PREFIX_RE.match(text)
    if not prefix or prefix.group(1).casefold() != section_no.casefold():
        raise ValueError(
            f"Override {path.name} must begin with section number {section_no}"
        )
    chunk_id = f"{act}-{section_part}"
    opening_line = text.splitlines()[0].rstrip("\r")
    language = "hi" if len(re.findall(r"[\u0900-\u097F]", text)) > len(
        re.findall(r"[A-Za-z]", text)
    ) else "en"
    return {
        "chunk_id": chunk_id,
        "act": act,
        "section_no": section_no,
        "section_title": _section_title(opening_line, ""),
        "text": text,
        "language": language,
        "source_file": path.name,
    }


def ingest_sources(
    source_dir: Path = SOURCE_DIR,
    chunks_path: Path = CHUNKS_PATH,
    unparsed_path: Path = UNPARSED_PATH,
    manual_overrides_dir: Path = MANUAL_OVERRIDES_DIR,
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
    overrides: list[dict[str, str]] = []
    if manual_overrides_dir.is_dir():
        for path in sorted(manual_overrides_dir.glob("*.txt")):
            override = _parse_override(path)
            LOGGER.info("Applying manual legal override %s", path.name)
            overrides.append(override)
    chunks.extend(overrides)
    unique_chunks: dict[tuple[str, str], dict[str, str]] = {}
    for chunk in chunks:
        key = (chunk["act"], chunk["section_no"])
        if key not in unique_chunks or len(chunk["text"]) > len(unique_chunks[key]["text"]):
            unique_chunks[key] = chunk
    chunks = list(unique_chunks.values())
    invalid_openings = validate_chunk_openings(chunks)
    if invalid_openings:
        raise ValueError(
            f"{len(invalid_openings)} chunks do not begin with their own section number"
        )
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
    parser.add_argument("--manual-overrides-dir", type=Path, default=MANUAL_OVERRIDES_DIR)
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    ingest_sources(args.source_dir, args.chunks, args.unparsed, args.manual_overrides_dir)


if __name__ == "__main__":
    main()
