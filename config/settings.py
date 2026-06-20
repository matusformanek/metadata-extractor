# -*- coding: utf-8 -*-
"""Central Configuration for the Metadata Extraction Pipeline.

This module defines all runtime constants, file system paths, model options,
and retrieval-augmented generation (RAG) field boundaries utilized across
the automated Dublin Core metadata extraction pipeline.
"""


from pathlib import Path
from typing import Any, Dict

from config.prompt_loader import load_system_prompt

# Uncomment the following lines to enable verbose debug logging if required
# logging.basicConfig(level=logging.DEBUG)


# ==============================================================================  # noqa: E501
# LLM RUNTIME ENVIRONMENT SETTINGS
# ==============================================================================  # noqa: E501

# Active local Large Language Model identifiers hosted via Ollama
LLM_MODEL = "hf.co/unsloth/gemma-4-E4B-it-GGUF:Q4_K_M"

# Text embedding model utilized for vector database operations
EMBED_MODEL = "nomic-embed-text"

# Base network address for the local Ollama API service instance
OLLAMA_API_URL = "http://localhost:11434"


# ==============================================================================  # noqa: E501
# FILE SYSTEM ROUTING AND PATHS
# ==============================================================================  # noqa: E501

# Absolute paths pointing to the core data and prompt directories
BASE_DIR = Path(__file__).resolve().parents[1]
DATA_DIR = BASE_DIR / "data"

SYSTEM_PROMPT_PATH = BASE_DIR / "prompts" / "system_prompt.txt"
CANDIDATES_PROMPT_PATH = BASE_DIR / "prompts" / "pass1_candidates_prompt.txt"

INPUT_GLOB = str(DATA_DIR / "input" / "**" / "*")
OUTPUT_DIR = DATA_DIR / "output"

OUTPUT_SUFFIX = "_metadata.json"
RAW_SUFFIX = "_raw.txt"
LOG_FILE = str(DATA_DIR / "pipeline.log")


# ==============================================================================  # noqa: E501
# CONTEXT WINDOW AND RETRY CONSTRAINTS
# ==============================================================================  # noqa: E501

# Maximum characters allowed from document stream to fit model context bounds
MAX_TEXT_CHARS = 9000

# Number of initial characters extracted from document header sections
HEAD_CHARS = 7000

# Number of trailing characters extracted from document reference sections
TAIL_CHARS = 1000

# Maximum execution attempts for a single pipeline block upon failure
MAX_ATTEMPTS = 2


# ==============================================================================  # noqa: E501
# CHROMADB VECTOR STORAGE CONFIGURATION
# ==============================================================================  # noqa: E501

# Directory mapping for the local vector index deployment
CHROMA_PATH = str(DATA_DIR / "vector_db")

# Collection identity token storing vocabulary term variations
CHROMA_COLLECTION = "metadata_lookup_vocabularies"

# Semantic distance threshold for vector similarity matching (Cosine distance metric).  # noqa: E501
# This value defines the strictness of the domain-filtered vocabulary lookups.
# Empirical testing with the current collection content demonstrated that lowering  # noqa: E501
# this threshold below 0.40 (e.g., to 0.30) caused ChromaDB to return zero hits
# for valid entries due to text density drift in language and publisher fields.
DISTANCE_THRESHOLD = 0.40

# ==============================================================================  # noqa: E501
# PIPELINE INPUT FILTER SPECIFICATIONS
# ==============================================================================  # noqa: E501

# Explicit collection of file extensions supported by document parsing layers
SUPPORTED_EXTENSIONS = {".pdf", ".epub", ".docx", ".odt", ".txt"}


# ==============================================================================  # noqa: E501
# GENERATION INFERENCE SETTINGS
# ==============================================================================  # noqa: E501

# Pass 2 options: Parameters fine-tuned for final metadata field extraction
OLLAMA_OPTIONS: Dict[str, Any] = {
    "temperature": 0.0,
    "top_p": 1.0,
    "repeat_penalty": 1.05,
    "num_predict": 2560,
    "num_ctx": 9000,
}

# Pass 1 options: Lightweight generation configuration for initial RAG terms
PASS1_OPTIONS: Dict[str, Any] = {
    "temperature": 0.0,
    "top_k": 10,
    "top_p": 0.5,
    "num_predict": 200,
}


# ==============================================================================  # noqa: E501
# RAG RETRIEVAL FIELD LAYOUTS
# ==============================================================================  # noqa: E501

# Vector lookup limits mapped to restricted Dublin Core controlled fields
RAG_FIELD_CONFIG: Dict[str, Dict[str, Any]] = {
    "language": {
        "lookup_domain": "language",
        "k": 1,
    },
    "rights_uri": {
        "lookup_domain": "license",
        "k": 1,
    },
    "resource_type": {
        "lookup_domain": "resource_type",
        "k": 1,
    },
    "publisher": {
        "lookup_domain": "publisher",
        "k": 1,
    },
}


# ==============================================================================  # noqa: E501
# PROMPT TEMPLATE INITIALIZATION
# ==============================================================================  # noqa: E501

# Pre-loaded prompt instruction blocks populated at startup runtime
SYSTEM_PROMPT = load_system_prompt(SYSTEM_PROMPT_PATH)
CANDIDATES_PROMPT = load_system_prompt(CANDIDATES_PROMPT_PATH)
