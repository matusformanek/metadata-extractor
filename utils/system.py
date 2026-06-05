# -*- coding: utf-8 -*-
"""System level utility helpers for local model resource management.

This module provides helper functions designed to manage hardware resources
and execution contexts when interacting with local Large Language Model
backends such as Ollama or LM Studio.
"""

import logging

import requests

from config.settings import OLLAMA_API_URL


def release_vram(model_name: str) -> None:
    """Instruct the local Ollama service to unload a model from GPU memory.

    By transmitting a standard generation payload with the 'keep_alive'
    parameter explicitly set to 0, Ollama immediately evicts the model layers
    from Video RAM, freeing memory space for subsequent local operations.

    Args:
        model_name (str): The exact identifier string of the local LLM
            that needs to be evicted from VRAM.
    """
    # Construct the target endpoint URL specifically for generation tasks
    endpoint = f"{OLLAMA_API_URL}/api/generate"

    # Define the payload instructing the service to release the model layers
    payload = {
        "model": model_name,
        "keep_alive": 0
    }

    try:
        # Dispatch a synchronous HTTP POST request with a strict timeout safety
        response = requests.post(
            endpoint,
            json=payload,
            timeout=5
        )

        # Trigger an exception automatically if the server returns an error code
        response.raise_for_status()

    except Exception as exc:
        # Log network failures or timeouts without interrupting application flow
        # Using lazy formatting for logging strings to optimize processing time
        logging.warning(
            "VRAM release sequence failed for model %s: %s",
            model_name,
            exc
        )