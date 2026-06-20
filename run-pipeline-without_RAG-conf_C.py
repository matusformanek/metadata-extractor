# -*- coding: utf-8 -*-
"""Orchestrated metadata extraction pipeline using local Large Language Models via Ollama.  # noqa: E501

Configuration: metadata-extractor-configuration C (Entry Point)

Note:
    This script serves as the main entry point for Configuration C.
    It is not a standalone tool and strictly requires other scripts and
    modules within the overall architecture to function properly. It depends
    on external architectural components for configuration, pipeline execution,
    and logging setup.

    Configuration C activates the full dual-phase Pass 1 / Pass 2 inference
    pipeline with advanced prompting, Pydantic schema validation, and the
    agentic repair loop (Inspector). The RAG subsystem (Librarian/ChromaDB)
    is intentionally excluded. Passing None as the librarian argument to
    process_file disables the RAG lookup between Pass 1 and Pass 2.

    This configuration directly isolates the contribution of the RAG layer
    when compared with Configuration D, which is architecturally identical
    except for the active Librarian module.

Ablation role:
    Measures the isolated contribution of advanced prompting, dual-phase
    inference, and agentic validation feedback — without any external
    semantic knowledge base.
"""

import glob
import logging
import sys
from pathlib import Path

from config.settings import INPUT_GLOB, LOG_FILE, SUPPORTED_EXTENSIONS
from core.pipeline import process_file
from utils.logging_setup import setup_logging


def run() -> None:
    """Scan the input directory tree and execute the metadata extraction batch.

    Discovers target objects using pre-configured search globs, screens out
    previously generated output structures to avoid extraction feedback loops,
    and evaluates pipeline execution health.

    The Librarian module is not instantiated in this configuration. The
    process_file function receives None as the librarian argument, which
    disables the RAG lookup step between Pass 1 and Pass 2 inference phases.
    All other pipeline components (Scribe, Inspector, post-processing) remain
    fully active and identical to Configuration D.
    """
    setup_logging(LOG_FILE)

    # Filter out transient data artifacts or pre-generated metadata outputs.
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

    # RAG subsystem is explicitly disabled for Configuration C.
    # Librarian is not instantiated; process_file receives None instead.
    print(
        "[INFO] RAG subsystem disabled (Configuration C — no Librarian/ChromaDB)."  # noqa: E501
    )
    logging.info("Configuration C: RAG subsystem disabled.")

    ok_count = 0
    for file_path in files:
        # None signals process_file to skip the RAG lookup between Pass 1 and
        # Pass 2.
        if process_file(file_path, None):
            ok_count += 1

    summary = f"Done. Successfully processed: {ok_count}/{len(files)}"
    logging.info(summary)
    print(f"\n[INFO] {summary}")


if __name__ == "__main__":
    run()
