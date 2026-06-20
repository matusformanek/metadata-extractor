# -*- coding: utf-8 -*-
"""Logging Configuration Utility Helpers.

This module provides initialization routines to establish uniform logging
mechanisms across the application runtime environment. It automatically
handles file system dependencies and sets up safe stream configurations.
"""

import logging
from pathlib import Path


def setup_logging(log_file: str) -> None:
    """Configure file-based application logging diagnostics.

    This function initializes the global root logger state by targeting
    a specific output path on the disk, setting the baseline filtering
    threshold level, and defining the chronological format of all emitted
    system records.

    Args:
        log_file (str): The relative or absolute file system path string
            where the logging entries will be persistently stored.
    """
    # Convert string path to a Path object to leverage robust file system
    # operations
    target_path = Path(log_file)

    # Automatically create the entire parent directory hierarchy if it does
    # not exist
    target_path.parent.mkdir(parents=True, exist_ok=True)

    # Establish the global base configuration with explicit UTF-8 encoding
    # strings
    logging.basicConfig(
        filename=str(target_path),
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
        encoding="utf-8",
    )
