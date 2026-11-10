"""Extract user-supplied legal PDFs without authoring or correcting legal text."""

from __future__ import annotations

import argparse
import logging
import random
import re
import subprocess
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
SOURCE_PDF_DIR = PROJECT_ROOT / "sakhi" / "data" / "legal" / "source_pdf"
SOURCE_DIR = PROJECT_ROOT / "sakhi" / "data" / "legal" / "source"
REPORT_PATH = PROJECT_ROOT / "sakhi" / "reports" / "audit" / "legal_pdf_report.md"
LOGGER = logging.getLogger(__name__)
PAGE_NUMBER_RE = re.compile(r"^\s*(?:page\s*)?\d+\s*$", re.IGNORECASE)
BNS_NAME = "BNS_2023.pdf"
PDFPLUMBER_X_TOLERANCE = 1.5
BNS_GAZETTE_HEADER_RE = re.compile(
    r"^\s*(?:_+\s*\d*\s*|_*\s*(?:THE GAZETTE OF INDIA EXTRAORDINARY|\[?Part II—).*)\s*$",
    re.IGNORECASE,
)
LONG_SINGLE_TOKEN_RE = re.compile(r"^[^\W\d_]+$", re.UNICODE)


def _extract_with_pdftotext(pdf_path: Path) -> list[str]:
    try:
        completed = subprocess.run(
            ["pdftotext", str(pdf_path), "-"],
            check=True,
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
        )
    except FileNotFoundError as exc:
        raise RuntimeError(
            "The pdftotext engine was requested, but Poppler's pdftotext command "
            "was not found on PATH."
        ) from exc
    except subprocess.CalledProcessError as exc:
        raise RuntimeError(f"pdftotext failed for {pdf_path}: {exc.stderr}") from exc
    return completed.stdout.split("\f")


def _extract_with_pdfplumber(
    pdf_path: Path, x_tolerance: float = PDFPLUMBER_X_TOLERANCE
) -> list[str]:
    try:
        import pdfplumber
    except ImportError as exc:
        raise RuntimeError(
            "pdfplumber is required for legal PDF extraction. Install requirements-train.txt."
        ) from exc
    with pdfplumber.open(pdf_path) as pdf:
        return [page.extract_text(x_tolerance=x_tolerance) or "" for page in pdf.pages]


def _extract_pages(
    pdf_path: Path,
    engine: str = "pdfplumber",
    x_tolerance: float = PDFPLUMBER_X_TOLERANCE,
) -> list[str]:
    if engine == "pdftotext":
        return _extract_with_pdftotext(pdf_path)
    if engine != "pdfplumber":
        raise ValueError(f"Unsupported PDF extraction engine: {engine}")
    if pdf_path.name == BNS_NAME:
        pages, _ = _extract_bns_pages(pdf_path, x_tolerance=x_tolerance)
        return pages
    return _extract_with_pdfplumber(pdf_path, x_tolerance=x_tolerance)


def _extract_bns_pages(
    pdf_path: Path, x_tolerance: float = PDFPLUMBER_X_TOLERANCE
) -> tuple[list[str], list[dict[str, object]]]:
    try:
        import pdfplumber
    except ImportError as exc:
        raise RuntimeError(
            "pdfplumber is required for BNS margin-column extraction. "
            "Install requirements-train.txt."
        ) from exc

    extracted: list[str] = []
    margin_pages: list[dict[str, object]] = []
    with pdfplumber.open(pdf_path) as pdf:
        for page_number, page in enumerate(pdf.pages, start=1):
            words = page.extract_words()
            if not words:
                extracted.append("")
                continue
            page_width = float(page.width)
            max_x = max(float(word["x1"]) for word in words)
            left_margin_words = sum(float(word["x1"]) < 118 for word in words)
            right_margin_words = sum(float(word["x0"]) > 478 for word in words)
            if left_margin_words >= 3 and left_margin_words > right_margin_words:
                body_left = 118.0
                body_right = round(max_x, 1) if max_x > 500 else 478.0
            elif right_margin_words >= 3 and right_margin_words > left_margin_words:
                body_left = 58.0
                body_right = 478.0
            elif max_x > 500:
                body_right = round(max_x, 1)
                body_left = round(body_right - 420.0, 1)
            else:
                body_left, body_right = 58.0, 478.0
            kept_words = [
                word
                for word in words
                if float(word["x0"]) >= body_left - 1
                and float(word["x1"]) <= body_right + 1
            ]
            removed_words = [word for word in words if word not in kept_words]
            margin_detected = bool(removed_words)
            if margin_detected:
                side = (
                    "left"
                    if sum(float(word["x1"]) < body_left for word in removed_words)
                    >= sum(float(word["x0"]) > body_right for word in removed_words)
                    else "right"
                )
                details = {
                    "page": page_number,
                    "side": side,
                    "body_left": round(body_left, 2),
                    "body_right": round(body_right, 2),
                    "removed_word_count": len(removed_words),
                }
                margin_pages.append(details)
                LOGGER.info(
                    "%s page %d: removed %d marginal-column words on the %s "
                    "(body x=%.2f..%.2f)",
                    pdf_path.name,
                    page_number,
                    len(removed_words),
                    side,
                    body_left,
                    body_right,
                )
            cropped = page.crop((body_left, 0, body_right, page.height))
            extracted.append(cropped.extract_text(x_tolerance=x_tolerance) or "")
    return extracted, margin_pages


def _repeated_lines(pages: list[str]) -> set[str]:
    counts: dict[str, int] = {}
    for page in pages:
        lines = [line.strip() for line in page.splitlines() if line.strip()]
        candidates = set(lines[:1] + lines[-1:])
        for line in candidates:
            if not PAGE_NUMBER_RE.fullmatch(line):
                counts[line] = counts.get(line, 0) + 1
    minimum = max(2, (len(pages) + 1) // 2)
    return {line for line, count in counts.items() if count >= minimum}


def _clean_pages(
    pages: list[str], remove_bns_headers: bool = False
) -> tuple[list[str], list[str]]:
    repeated = _repeated_lines(pages)
    removed: list[str] = []
    cleaned: list[str] = []
    for page in pages:
        kept: list[str] = []
        for line in page.splitlines():
            stripped = line.strip()
            if remove_bns_headers and BNS_GAZETTE_HEADER_RE.fullmatch(stripped):
                removed.append(stripped)
                continue
            if stripped and (stripped in repeated or PAGE_NUMBER_RE.fullmatch(stripped)):
                removed.append(stripped)
                continue
            kept.append(line)
        cleaned.append("\n".join(kept).strip())
    return cleaned, removed


def _spacing_check(pages: list[str]) -> dict[str, object]:
    tokens = [
        token
        for page in pages
        for line in page.splitlines()
        if len(line.split()) == 1
        for token in line.split()
        if LONG_SINGLE_TOKEN_RE.fullmatch(token) and len(token) > 40
    ]
    all_tokens = [
        token
        for page in pages
        for token in re.findall(r"[^\W\d_]+", page, flags=re.UNICODE)
    ]
    return {
        "long_single_token_lines": len(tokens),
        "longest_tokens": sorted(set(all_tokens), key=lambda token: (-len(token), token))[:5],
    }


def _looks_garbled(text: str) -> bool:
    if not text.strip():
        return False
    replacement_count = text.count("\ufffd")
    non_letter_count = sum(
        1 for char in text if not char.isspace() and not char.isalnum() and char not in ".,;:!?-()[]'\"/"
    )
    devanagari = sum("\u0900" <= char <= "\u097f" for char in text)
    broken_devanagari = devanagari > 0 and ("\u200c" in text or "\u200d" in text)
    return replacement_count > 0 or non_letter_count > max(10, len(text) // 3) or broken_devanagari


def convert_pdf(
    pdf_path: Path,
    output_dir: Path = SOURCE_DIR,
    engine: str = "pdfplumber",
    x_tolerance: float = PDFPLUMBER_X_TOLERANCE,
) -> dict[str, object]:
    margin_pages: list[dict[str, object]] = []
    if engine == "pdfplumber" and pdf_path.name == BNS_NAME:
        pages, margin_pages = _extract_bns_pages(pdf_path, x_tolerance=x_tolerance)
    else:
        pages = _extract_pages(pdf_path, engine=engine, x_tolerance=x_tolerance)
    cleaned_pages, removed = _clean_pages(
        pages, remove_bns_headers=pdf_path.name == BNS_NAME
    )
    spacing = _spacing_check(cleaned_pages)
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / f"{pdf_path.stem}.txt"
    output_path.write_text("\n\n".join(cleaned_pages) + "\n", encoding="utf-8")
    sparse = [index + 1 for index, text in enumerate(cleaned_pages) if len(text.strip()) < 20]
    garbled = [index + 1 for index, text in enumerate(cleaned_pages) if _looks_garbled(text)]
    for line in removed:
        LOGGER.info("%s: removed repeated header/footer/page-number line: %r", pdf_path.name, line)
    return {
        "name": pdf_path.name,
        "output": str(output_path),
        "pages": len(pages),
        "sparse_pages": sparse,
        "garbled_pages": garbled,
        "removed": removed,
        "margin_pages": margin_pages,
        "pages_text": cleaned_pages,
        "engine": engine,
        "x_tolerance": x_tolerance if engine == "pdfplumber" else None,
        "spacing": spacing,
    }


def convert_all(
    source_pdf_dir: Path = SOURCE_PDF_DIR,
    output_dir: Path = SOURCE_DIR,
    report_path: Path = REPORT_PATH,
    engine: str = "pdfplumber",
    engine_overrides: dict[str, str] | None = None,
    x_tolerance_overrides: dict[str, float] | None = None,
) -> list[dict[str, object]]:
    engine_overrides = engine_overrides or {}
    x_tolerance_overrides = x_tolerance_overrides or {}
    results = [
        convert_pdf(
            path,
            output_dir,
            engine=engine_overrides.get(path.name, engine),
            x_tolerance=x_tolerance_overrides.get(path.name, PDFPLUMBER_X_TOLERANCE),
        )
        for path in sorted(source_pdf_dir.glob("*.pdf"))
    ]
    lines = [
        "# Legal PDF extraction report",
        "",
        "Text below is extracted from user-supplied PDFs. No legal wording was authored or corrected.",
        "",
    ]
    for result in results:
        lines.extend(
            [
                f"## {result['name']}",
                "",
                f"- Output: `{result['output']}`",
                f"- Engine: `{result['engine']}`",
                (
                    f"- pdfplumber x_tolerance: `{result['x_tolerance']}`"
                    if result["x_tolerance"] is not None
                    else ""
                ),
                f"- Pages: {result['pages']}",
                f"- Pages with almost no text: {len(result['sparse_pages'])} "
                f"({result['sparse_pages'] or 'none'})",
                f"- Pages flagged as possibly garbled: {result['garbled_pages'] or 'none'}",
                f"- Removed repeated/header/footer/page-number lines: {len(result['removed'])}",
                f"- BNS Gazette header lines removed: "
                f"{sum(1 for line in result['removed'] if BNS_GAZETTE_HEADER_RE.fullmatch(line))}",
                f"- BNS marginal-column pages removed: {len(result['margin_pages'])}",
                f"- Lines with one alphabetic token longer than 40 letters: "
                f"{result['spacing']['long_single_token_lines']}",
                f"- Five longest tokens: {result['spacing']['longest_tokens']}",
            ]
        )
        if result["margin_pages"]:
            lines.append(
                "  - "
                + "; ".join(
                    f"page {item['page']} ({item['side']}, "
                    f"x={item['body_left']}..{item['body_right']}, "
                    f"{item['removed_word_count']} words)"
                    for item in result["margin_pages"]
                )
            )
        if result["removed"]:
            lines.append("  - " + "; ".join(repr(value) for value in result["removed"]))
        page_count = int(result["pages"])
        sample_indices = random.Random(result["name"]).sample(range(page_count), min(3, page_count))
        lines.extend(["", "### Random page samples (first 300 characters)", ""])
        for index in sorted(sample_indices):
            sample = str(result["pages_text"][index])[:300].replace("\n", " ")
            lines.append(f"- Page {index + 1}: `{sample}`")
        lines.append("")
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text("\n".join(lines), encoding="utf-8")
    for result in results:
        print(
            f"{result['name']}: pages={result['pages']}, "
            f"engine={result['engine']}, "
            f"almost_no_text={len(result['sparse_pages'])}, "
            f"garbled={result['garbled_pages'] or 'none'}, "
            f"long_single_tokens={result['spacing']['long_single_token_lines']}, "
            f"longest_tokens={result['spacing']['longest_tokens']}"
        )
        for index in sorted(
            random.Random(result["name"]).sample(
                range(int(result["pages"])), min(3, int(result["pages"]))
            )
        ):
            print(f"  page {index + 1}: {str(result['pages_text'][index])[:300]}")
    print(f"Report: {report_path}")
    bns_result = next((result for result in results if result["name"] == BNS_NAME), None)
    if bns_result and bns_result["spacing"]["long_single_token_lines"] > 0:
        raise RuntimeError(
            "BNS_2023.txt still contains lines with a single alphabetic token "
            "longer than 40 letters; use a better extraction engine/source PDF."
        )
    return results


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-pdf-dir", type=Path, default=SOURCE_PDF_DIR)
    parser.add_argument("--output-dir", type=Path, default=SOURCE_DIR)
    parser.add_argument("--report", type=Path, default=REPORT_PATH)
    parser.add_argument("--engine", choices=("pdfplumber", "pdftotext"), default="pdfplumber")
    parser.add_argument(
        "--file-engine",
        action="append",
        default=[],
        metavar="FILE=ENGINE",
        help="Override the engine for one file, for example BNS_2023.pdf=pdftotext.",
    )
    parser.add_argument(
        "--x-tolerance",
        action="append",
        default=[],
        metavar="FILE=VALUE",
        help="Override pdfplumber x_tolerance for one file.",
    )
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(errors="replace")
    engine_overrides = {}
    for item in args.file_engine:
        name, separator, value = item.partition("=")
        if not separator or value not in {"pdfplumber", "pdftotext"}:
            parser.error(f"Invalid --file-engine value: {item}")
        engine_overrides[name] = value
    x_tolerance_overrides = {}
    for item in args.x_tolerance:
        name, separator, value = item.partition("=")
        if not separator:
            parser.error(f"Invalid --x-tolerance value: {item}")
        try:
            x_tolerance_overrides[name] = float(value)
        except ValueError:
            parser.error(f"Invalid --x-tolerance value: {item}")
    convert_all(
        args.source_pdf_dir,
        args.output_dir,
        args.report,
        engine=args.engine,
        engine_overrides=engine_overrides,
        x_tolerance_overrides=x_tolerance_overrides,
    )


if __name__ == "__main__":
    main()
