# -*- coding: utf-8 -*-
"""Metadata extraction pipeline using local Large Language Models via Ollama.

This module ingests academic and descriptive documents in various formats,
slices them into a head-tail sandwich representation to fit context windows,
and queries a local LLM to extract structured Dublin Core metadata fields
returned as validated JSON files.
"""

import json
import logging
import sys
import traceback
import subprocess
from pathlib import Path
from typing import Optional

import ollama
import pymupdf4llm

# Configuration constants for model execution and context boundaries.
# Changing these impacts token usage, processing speed, and metadata accuracy.
MODEL_NAME = "qwen3.5:4b"
INPUT_DIR = Path("data/input")
OUTPUT_DIR = Path("data/output")

SUPPORTED_FORMATS = {".pdf", ".epub", ".docx", ".odt", ".txt"}

OLLAMA_OPTIONS = {
    "temperature": 0.0,  # Zero temperature ensures deterministic extractions.
    "num_ctx": 9000      # Context window sized safely for a 4B/8B parameter LLM.
}

SYSTEM_PROMPT = """Extract descriptive bibliographic metadata from the provided text and return exactly one JSON object in the exact schema below.
{
  "title": "The primary name given to the document.",
  "alternative_title": "Subtitle or parallel title in another language only if explicitly present.",
  "authors": [
    "Personal names of all primary creators only. Exclude institutions, departments, and affiliations."
  ],
  "contributors": [
    "List of secondary contributors or institutions."
  ],
  "issued": "Year of formal publication (YYYY). Prefer the print or final version year over online-first or epub-ahead dates. Use YYYY if full date is unavailable.",
  "publisher": "The entity responsible for publication.",
  "publication_place": "The geographic location of publication.",
  "resource_type": "Exact type from: article, book, book-chapter, conference-paper, dataset, thesis, editorial, letter, preprint, report, review, standard, other.",
  "language": "ISO 639-1 code of the document's primary language (e.g. en, de, fr, pt, ja, pl).",
  "doi": "Digital Object Identifier of primary document if available. Without prefix: https://doi.org/",
  "isbn": "International Standard Book Number.",
  "issn_print": "International Standard Serial Number for the print edition.",
  "issn_electronic": "International Standard Serial Number for the electronic edition.",
  "persistent_uri": "The canonical uniform resource identifier.",
  "abstract": "A brief summary of the resource content.",
  "subjects": [
    "Keywords, classification codes, or domain topics."
  ],
  "rights_uri": "The URI pointing to license conditions or copyright laws."
}
"""

# Standardized logging configuration routing info and errors directly to stdout.
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)]
)


# ---------------------------------------------------------------------------
# Markdown conversion helper (NEW)
# ---------------------------------------------------------------------------

def convert_to_markdown_pandoc(file_path: Path) -> str:
    """Convert DOCX or ODT file to Markdown using Pandoc."""
    try:
        result = subprocess.run(
            ["pandoc", str(file_path), "-t", "markdown"],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            check=True,
        )
        return result.stdout.decode("utf-8", errors="replace")
    except FileNotFoundError:
        raise RuntimeError("Pandoc is not installed or not available in PATH.")
    except subprocess.CalledProcessError as e:
        raise RuntimeError(f"Pandoc conversion failed: {e.stderr.decode()}")


# ---------------------------------------------------------------------------
# Direct Text Extraction Functions
# ---------------------------------------------------------------------------

def extract_text(file_path: Path, head: int = 7000, tail: int = 1000) -> str:
    """Extract raw text from a supported file and apply character bounds.

    Routes the file path to the designated third-party parsing library based
    on its suffix. Returns a concatenated string combining the front and end
    of the text layer if the total length exceeds the predefined thresholds.

    Args:
        file_path (Path): The target filesystem path pointing to the document.
        head (int): Number of characters captured from the start. Default 7000.
        tail (int): Number of characters captured from the end. Default 1000.

    Returns:
        str: A bounded, continuous text block formatted as a context sandwich.

    Raises:
        ValueError: If the file extension is missing from SUPPORTED_FORMATS.
        RuntimeError: If the extracted content is empty after processing.
    """
    suffix = file_path.suffix.lower()
    text = ""

    # Processing plain text files with safe byte replacement for encoding errors.
    if suffix == ".txt":
        text = file_path.read_text(encoding="utf-8", errors="replace")

    # PDF and EPUB parsing handled via PyMuPDF's specialized Markdown layer.
    elif suffix in {".pdf", ".epub"}:
        text = pymupdf4llm.to_markdown(str(file_path))

    # Microsoft Word and ODT now parsed via Pandoc → Markdown
    elif suffix in {".docx", ".odt"}:
        text = convert_to_markdown_pandoc(file_path)

    else:
        raise ValueError(f"Unsupported file format: '{suffix}'")

    if not text.strip():
        raise RuntimeError("Extracted text is empty after processing")

    # Head-tail sandwich slice applied directly on character arrays.
    # This prevents long document bodies from overloading the LLM context.
    if len(text) <= (head + tail):
        return text

    return text[:head] + "\n\n[...]\n\n" + text[-tail:]


# ---------------------------------------------------------------------------
# Diagnostics Helpers
# ---------------------------------------------------------------------------

def save_raw_response(stem: str, raw_response: str) -> None:
    """Save the unparsed text response from the LLM to a raw file."""

    raw_file = OUTPUT_DIR / f"{stem}_raw.txt"
    with raw_file.open("w", encoding="utf-8") as f:
        f.write(raw_response)
    logging.info(f"[DIAGNOSTIC] Raw response isolated: {raw_file}")


def save_diagnostic_log(stem: str, error_msg: str) -> None:
    """Write traceback details and runtime exceptions to a log file."""

    log_file = OUTPUT_DIR / f"{stem}_error.txt"
    with log_file.open("w", encoding="utf-8") as f:
        f.write(f"=== ERROR DIAGNOSTIC INFO ===\n{error_msg}\n")
    logging.info(f"[DIAGNOSTIC] Complete execution trace written to: {log_file}")


# ---------------------------------------------------------------------------
# Execution Pipeline
# ---------------------------------------------------------------------------

def run_metadata_pipeline() -> None:
    """Execute the core batch-processing loop over the input directory."""

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    target_files = sorted(
        f for f in INPUT_DIR.iterdir()
        if f.is_file() and f.suffix.lower() in SUPPORTED_FORMATS
    )

    logging.info(f"Discovered {len(target_files)} target files in {INPUT_DIR}")

    for file_path in target_files:
        logging.info(f"Processing document: {file_path.name}")
        raw_response = ""

        try:
            extracted_text = extract_text(file_path)

            if not extracted_text.strip():
                logging.warning(
                    f"Skipping {file_path.name} — extracted text payload is empty"
                )
                continue

            response = ollama.generate(
                model=MODEL_NAME,
                system=SYSTEM_PROMPT,
                prompt=extracted_text,
                format="json",
                options=OLLAMA_OPTIONS,
                keep_alive="5m",
            )
            raw_response = response.get("response", "").strip()
            
            parsed_json = json.loads(raw_response)

            output_file = OUTPUT_DIR / f"{file_path.stem}_metadata.json"
            with output_file.open("w", encoding="utf-8") as f:
                json.dump(parsed_json, f, indent=2, ensure_ascii=False)
            logging.info(f"Successfully serialized metadata: {output_file}")

        except json.JSONDecodeError as json_err:
            tb = traceback.format_exc()
            msg = f"JSON decoding failed.\nException: {json_err}\n\nTraceback:\n{tb}"
            logging.error(f"Failed to parse LLM json structure for {file_path.name}")
            save_raw_response(file_path.stem, raw_response)
            save_diagnostic_log(file_path.stem, msg)

        except Exception as general_err:
            tb = traceback.format_exc()
            msg = f"Unexpected execution crash.\nException: {general_err}\n\nTraceback:\n{tb}"
            logging.error(f"Critical pipeline failure on file {file_path.name}: {general_err}")
            if raw_response:
                save_raw_response(file_path.stem, raw_response)
            save_diagnostic_log(file_path.stem, msg)


if __name__ == "__main__":
    run_metadata_pipeline()