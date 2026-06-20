# -*- coding: utf-8 -*-
"""Document text extraction and context compaction pipelines.

This module provides the Scribe class, which ingests various text and binary
document formats, cleans structural conversion artifacts, and extracts
the bibliographically densest regions to construct a minimized context payload.
"""

import re
from pathlib import Path

import pymupdf4llm

# Constants imported from the central configuration module
from config.settings import HEAD_CHARS, MAX_TEXT_CHARS, TAIL_CHARS


class Scribe:
    """Extract and compact document text for LLM metadata analysis.

    Handles multiple file extensions, strips systematic encoding errors or
    academic markers such as ORCID IDs, and chunks long documents into a
    head-tail sandwich context representation.
    """

    @staticmethod
    def clean_artifacts(text: str) -> str:
        """Remove common PDF conversion artifacts before context slicing.

        Strips ORCID strings, isolates affiliation markers, and resolves
        unwanted spacing artifacts produced by the extraction layer.

        Args:
            text (str): Raw string slice obtained from the document parser.

        Returns:
            str: Sanitized text payload safe for LLM parsing.
        """
        if not text:
            return ""

        # Normalize broken ORCID separators produced by PDF-to-Markdown.
        text = re.sub(r"\s*_\[(?:-|\u2212)\]_\s*", "-", text)

        # Remove ORCID identifiers before the LLM sees the context. They are
        # not target metadata here and often fragment author lines.
        text = re.sub(
            r"\(?.*?orcid.*?\)?",
            "",
            text,
            flags=re.IGNORECASE,
        )
        text = re.sub(r"\b\d{4}-\d{4}-\d{4}-\d{3}[\dX]\b", "", text)
        text = re.sub(r"\[(\d{4})\]", r"\1", text)

        # Drop isolated affiliation icons while preserving surrounding words.
        text = re.sub(r"_\[[,\s\*\u2020\u2021\u00a7]+\]_", "", text)
        text = re.sub(r"[ \t]+", " ", text)

        # Remove Unicode replacement characters.
        text = re.sub(r"\ufffd", "", text)
        text = re.sub(r"(?<=\b[a-zA-Z]) (?=[a-zA-Z]\b)", "", text)

        return text

    @staticmethod
    def extract(file_path: str) -> str:
        """Read a supported document and return a head-tail text sandwich.

        Supports raw text, PDF, EPUB, DOCX, and ODT files. Long inputs
        are clipped into a condensed block focusing on the boundary sections
        defined by the configuration constants.

        Args:
            file_path (str): Absolute or relative filesystem path to target.

        Returns:
            str: Combined context boundary string within MAX_TEXT_CHARS.

        Raises:
            ValueError: If the file extension is not explicitly supported.
            RuntimeError: If document processing fails or returns empty data.
        """
        try:
            suffix = Path(file_path).suffix.lower()
            text = ""

            if suffix == ".txt":
                text = Path(file_path).read_text(
                    encoding="utf-8",
                    errors="replace",
                )
            elif suffix in {".pdf", ".epub"}:
                text = pymupdf4llm.to_markdown(file_path)
            elif suffix == ".docx":
                import docx

                doc = docx.Document(file_path)
                text = "\n".join(
                    paragraph.text
                    for paragraph in doc.paragraphs
                    if paragraph.text.strip()
                )
            elif suffix == ".odt":
                from odf import teletype
                from odf import text as odf_text
                from odf.opendocument import load

                doc = load(file_path)
                elements = doc.getElementsByType(odf_text.P)
                text = "\n".join(
                    teletype.extractText(element)
                    for element in elements
                    if teletype.extractText(element).strip()
                )
            else:
                raise ValueError(f"Scribe: Unsupported file format {suffix}")

            # Clean structural artifacts before performing context segmentation
            text = Scribe.clean_artifacts(text)

            # Apply character-based sandwich sampling using dynamic boundaries
            if len(text) <= HEAD_CHARS + TAIL_CHARS:
                combined = text
            else:
                introduction = text[:HEAD_CHARS]
                conclusion = text[-TAIL_CHARS:]

                # Structural tags are preserved for consistency across
                # configurations
                combined = (
                    f"This is the introduction of the document:\n{introduction}\n"  # noqa: E501
                    f"This is the conclusion of the document:\n{conclusion}"
                )

            if not combined.strip():
                raise RuntimeError("Scribe: extracted text is empty.")

            return combined[:MAX_TEXT_CHARS]
        except Exception as exc:
            raise RuntimeError(f"Scribe Error: {exc}") from exc
