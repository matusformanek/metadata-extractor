# -*- coding: utf-8 -*-
"""Command-line interface entry point for the metadata extraction pipeline.

Configuration: metadata-extractor-configuration D (Entry Point)

Note:
    This script serves as the main entry point for Configuration D.
    It is not a standalone tool and strictly requires other scripts and
    modules within the overall architecture to function properly. It depends
    on external architectural components for configuration, pipeline execution,
    vector store management, and logging setup.

    Configuration D activates the full pipeline including the dual-phase
    Pass 1 / Pass 2 inference with advanced prompting, Pydantic schema
    validation, the agentic repair loop (Inspector), and importantly,
    the RAG subsystem (Librarian/ChromaDB).

Ablation role:
    Represents the complete, fully-featured system. When compared with
    Configuration C, it measures the baseline effectiveness of semantic
    knowledge base (RAG) augmentation.
"""

import glob
import logging
import sys
from pathlib import Path

from config.settings import INPUT_GLOB, LOG_FILE, SUPPORTED_EXTENSIONS
from core.pipeline import process_file
from rag.librarian import Librarian
from utils.logging_setup import setup_logging


def run() -> None:
    """Scan the input directory tree and execute the metadata extraction batch.

    Discovers target objects using pre-configured search globs, screens out
    previously generated output structures to avoid extraction feedback loops,
    verifies vector store connectivity, and evaluates pipeline execution health.  # noqa: E501
    """
    setup_logging(LOG_FILE)

    # Filter out transient data artifacts or pre-generated metadata outputs
    files = [
        file_path
        for file_path in glob.glob(INPUT_GLOB, recursive=True)
        if Path(file_path).suffix.lower() in SUPPORTED_EXTENSIONS
        and "_metadata" not in Path(file_path).stem
    ]

    if not files:
        print("[ERROR] No valid files found in data/input/ directory.")
        sys.exit(1)

    print(f"[INFO] Found {len(files)} file(s) to process.")
    logging.info(f"Starting pipeline. Files: {len(files)}")

    librarian = Librarian()
    if librarian.collection is not None:
        print(f"[INFO] ChromaDB (RAG) is {librarian.status()}.")
    else:
        print(
            "[WARNING] ChromaDB unavailable. "
            f"Proceeding without RAG context ({librarian.status()})."
        )

    ok_count = 0
    for file_path in files:
        if process_file(file_path, librarian):
            ok_count += 1

    summary = f"Done. Successfully processed: {ok_count}/{len(files)}"
    logging.info(summary)
    print(f"\n[INFO] {summary}")


if __name__ == "__main__":
    run()
