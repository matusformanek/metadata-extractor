# -*- coding: utf-8 -*-
"""Orchestration logic for one-document metadata extraction."""

import logging
from pathlib import Path

from config.settings import (
    LLM_MODEL,
    MAX_ATTEMPTS,
    OLLAMA_OPTIONS,
    OUTPUT_DIR,
    OUTPUT_SUFFIX,
    RAW_SUFFIX,
    SYSTEM_PROMPT,
)
from core.inspector import Inspector
from extraction.candidates import extract_candidates
from extraction.scribe import Scribe
from llm.ollama_wrapper import OllamaWrapper
from rag.librarian import Librarian
from utils.system import release_vram


def process_file(file_path: str, librarian: Librarian) -> bool:
    """Process one source document and write its extracted metadata JSON.

    Extracts text context via Scribe, executes Pass 1 candidate extraction
    for methodological consistency, optionally queries ChromaDB if a librarian
    is available, and runs the final LLM validation loop.
    """
    doc_name = Path(file_path).name
    logging.info(f"Starting: {doc_name}")
    print(f"\n[INFO] Processing: {doc_name}")

    try:
        document_context = Scribe.extract(file_path)
    except Exception as exc:
        logging.error(f"[{doc_name}] Scribe failed: {exc}")
        print(f"  [ERROR] Scribe failed: {exc}")
        return False

    # Pass 1 is preserved for methodological evaluation across all
    # configurations
    print("  [INFO] Pass 1: Extracting candidates for RAG.")
    candidates = extract_candidates(document_context, LLM_MODEL)

    rag_context = ""

    # ChromaDB interaction is isolated to prevent errors when running without
    # RAG
    if librarian is not None and librarian.collection is not None:
        print("  [INFO] Pass 1.5: Searching ChromaDB with candidates.")
        rag_context = librarian.build_rag_context(candidates)
        librarian.release_embeddings()
    else:
        # Safe string fallback to prevent AttributeError on NoneType
        status_msg = (
            librarian.status()
            if librarian is not None
            else "Disabled (No RAG)"
        )
        logging.warning(
            f"[{doc_name}] ChromaDB unavailable, skipping RAG: {status_msg}"
        )
        print(f"  [WARNING] RAG skipped: {status_msg}")

    attempt = 1
    errors = ""
    raw_response_text = ""
    wrapper = OllamaWrapper(LLM_MODEL)

    while attempt <= MAX_ATTEMPTS:
        print(
            "  [INFO] Pass 2 "
            f"(attempt {attempt}/{MAX_ATTEMPTS}): Analyst extraction."
        )
        logging.info(f"[{doc_name}] Attempt {attempt}/{MAX_ATTEMPTS}")

        prompt = f"### DOCUMENT CONTEXT\n{document_context}"
        if rag_context:
            prompt += f"\n\n{rag_context}"
        if errors:
            prompt += (
                "\n\n### CRITICAL ERRORS FROM PREVIOUS ATTEMPT. "
                f"YOU MUST FIX THESE:\n{errors}"
            )

        try:
            raw_response_text = wrapper.generate(
                prompt=prompt,
                options=OLLAMA_OPTIONS,
                system=SYSTEM_PROMPT,
                keep_alive="5m",
            )

            if not raw_response_text:
                errors = "LLM returned empty response. Try again."
                logging.warning(f"[{doc_name}] {errors}")
                print(f"  [WARNING] {errors}")
                attempt += 1
                continue

            success, final_metadata, error_msg = Inspector.validate(
                raw_response_text,
                attempt,
            )

            if success:
                OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
                out_path = OUTPUT_DIR / (Path(file_path).stem + OUTPUT_SUFFIX)
                out_path.write_text(
                    final_metadata.model_dump_json(
                        indent=2,
                        ensure_ascii=False,
                    ),
                    encoding="utf-8",
                )
                logging.info(f"[{doc_name}] Success -> {out_path.name}")
                print(f"  [INFO] Saved: {out_path.name}")
                release_vram(LLM_MODEL)
                return True

            logging.warning(
                f"[{doc_name}] Inspector error (attempt {attempt}): "
                f"{error_msg}"
            )
            print(f"  [WARNING] Inspector: {error_msg}")
            errors = error_msg
            attempt += 1

        except Exception as exc:
            logging.error(f"[{doc_name}] LLM generation failed: {exc}")
            print(f"  [ERROR] LLM generation failed: {exc}")
            break

    if raw_response_text:
        OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
        raw_path = OUTPUT_DIR / (Path(file_path).stem + RAW_SUFFIX)
        raw_path.write_text(raw_response_text, encoding="utf-8")
        print(
            "  [WARNING] Raw LLM output saved for debugging: "
            f"{raw_path.name}"
        )

    logging.error(f"[{doc_name}] Failed after {MAX_ATTEMPTS} attempts.")
    print(f"  [ERROR] Failed after {MAX_ATTEMPTS} attempts.")
    release_vram(LLM_MODEL)
    return False
