# -*- coding: utf-8 -*-
"""Fast first-pass extraction of RAG lookup candidates.

This module implements the primary, lightweight inference layer designed to
slice raw document streams and isolate metadata structural fragments. The output
candidates are subsequently validated against controlled vocabularies via RAG.
"""

import json
import logging
from typing import Any, Dict

from config.settings import CANDIDATES_PROMPT, PASS1_OPTIONS
from llm.ollama_wrapper import OllamaWrapper


def extract_candidates(text: str, model: str) -> Dict[str, Any]:
    """Run the fast first LLM pass and return RAG search candidates.

    Slices the input text into specific boundary segments to minimize token
    overhead during the preliminary screening phase before structural schema
    enforcement.

    Args:
        text (str): The cleaned text payload delivered by the document parser.
        model (str): Name of the local quantized LLM used for initial inference.

    Returns:
        Dict[str, Any]: Deserialized dictionary containing raw, unverified
            metadata fields, or an empty dictionary upon processing failure.
    """
    head_chunk = text[:4000]
    tail_chunk = text[-1000:] if len(text) > 5000 else ""

    text_chunk = (
        f"### HEAD CHUNK (Document Start):\n{head_chunk}\n\n"
        f"### TAIL CHUNK (Document End/Colophon):\n{tail_chunk}"
    )
    prompt = CANDIDATES_PROMPT.replace("{text_chunk}", text_chunk)

    try:
        wrapper = OllamaWrapper(model)
        raw = wrapper.generate(
            prompt=prompt,
            options=PASS1_OPTIONS,
            system=None,
            keep_alive="5m",
        )
        
        if not raw:
            return {}
            
        return json.loads(raw)

    except json.JSONDecodeError as exc:
        logging.warning(
            f"extract_candidates failed to parse LLM JSON response: {exc}"
        )
        return {}
    except Exception as exc:
        logging.warning(
            f"extract_candidates encountered an unexpected pipeline failure: {exc}"
        )
        return {}