# -*- coding: utf-8 -*-
"""Ollama API adapter layer for pipeline text generation.

This module provides a unified wrapper interface around the official Ollama
Python library to streamline token generation passes across multiple pipeline
stages.
"""

from typing import Any, Dict, Optional

import ollama


class OllamaWrapper:
    """Low-level adapter encapsulating the Ollama service API endpoints.

    Handles context persistence configuration, system prompt injection, and
    structured response layout enforcement for extraction sub-agents.
    """

    def __init__(self, model: str) -> None:
        """Initialize the wrapper with a target local model configuration.

        Args:
            model (str): Name of the quantized open-source LLM instance
                registered within the Ollama daemon environment.
        """
        self.model = model

    def generate(
        self,
        prompt: str,
        options: Dict[str, Any],
        system: Optional[str] = None,
        keep_alive: str = "5m",
        response_format: str = "json",
    ) -> str:
        """Execute a blocking inference pass against the local model service.

        Args:
            prompt (str): Core execution instruction block passed to the model.
            options (Dict[str, Any]): Dictionary of model hyper-parameters
                such as temperature, top_p, or num_ctx.
            system (Optional[str]): System guidance instructions defining
                the operational persona of the agent. Defaults to None.
            keep_alive (str): VRAM persistence window directive for model
                weights inside memory. Defaults to "5m".
            response_format (str): Expected structural syntax constraint,
                typically set to "json". Defaults to "json".

        Returns:
            str: Stripped raw textual payload delivered by the generation endpoint.
        """
        kwargs: Dict[str, Any] = {
            "model": self.model,
            "prompt": prompt,
            "format": response_format,
            "options": options,
            "keep_alive": keep_alive,
        }
        if system is not None:
            kwargs["system"] = system

        response = ollama.generate(**kwargs)
        return response.get("response", "").strip()