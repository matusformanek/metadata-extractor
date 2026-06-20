# -*- coding: utf-8 -*-
"""Prompt File Loading Utilities.

This module provides helper functions responsible for securely reading,
decoding, and sanitizing prompt instruction templates stored on the local
file system before injecting them into the LLM context window.
"""

from pathlib import Path


def load_system_prompt(path: Path) -> str:
    """Load, decode, and sanitize a prompt instruction file from disk.

    This utility opens a file path using explicit UTF-8 encoding strings,
    extracts the underlying character data stream, and trims any redundant
    leading or trailing whitespaces and newline blocks.

    Args:
        path (Path): The absolute or relative pathlib Path reference pointing
            to the targeted prompt text file.

    Returns:
        str: Cleaned, whitespace-trimmed prompt string ready for runtime use.

    Raises:
        RuntimeError: Wrapped file system exception containing root failure
            diagnostics if the target file cannot be accessed or parsed.
    """
    try:
        # Open the file context safely using explicit encoding to avoid OS
        # collisions
        raw_text = path.read_text(encoding="utf-8")

        # Strip any excessive whitespace characters or trailing blank line
        # blocks
        return raw_text.strip()

    except Exception as exc:
        # Wrap the core operating system error into a clean runtime exception
        # layer
        raise RuntimeError(
            f"Failed to load system prompt from {path}: {exc}"
        ) from exc
