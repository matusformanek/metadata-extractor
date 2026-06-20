# Local LLM Pipeline for Bibliographic Metadata Extraction

A robust software system designed for the automated extraction of descriptive bibliographic metadata from unstructured documents in compliance with the international RDA (Resource Description and Access) standard. The architecture utilizes the orchestration of two local Large Language Model (LLM) inference phases combined with semantic search (RAG) over controlled vocabularies.

## Key System Features

* **Two Phase Extraction (Pass 1 and Pass 2):** The first phase rapidly identifies coarse semantic indicators while the second phase executes the final precise extraction of metadata.
* **Context Augmentation (RAG):** Integration with a local ChromaDB vector database allows the system to validate and prioritize preferred domain terms and controlled vocabularies.
* **Semantic Citation Isolation:** Strict rules embedded within the system prompt prevent the confusion of primary document entities with elements present in cited literature lists.
* **Deterministic Validation and Auditing:** The Inspector module enforces type safety using Pydantic schemas and audits the presence of literal textual evidence for each extracted field before the validation log is discarded.
* **Proactive VRAM Management:** Immediate eviction of the model from GPU memory is executed via platform API calls to Ollama upon completing the inference cycle of each document.

## Architecture and Execution Flow

The processing of each document occurs within a fully deterministic and isolated cycle managed by a central orchestrator. This process consists of the following sequential phases:

1. **Preprocessing and Text Extraction (Scribe):** The input document is transformed into Markdown format and stripped of formatting noise. If the text length exceeds the context limit, a head and tail compression strategy is applied to preserve sections where the density of bibliographic data is historically highest.
2. **Candidate Generation (Pass 1):** A smaller model performs a rapid preliminary analysis of the truncated context to identify clues regarding the publisher, language, or resource type.
3. **Semantic Search (Librarian):** The extracted clues serve as search queries directed to the local ChromaDB vector database. Relevant authority records and preferred domain terms are retrieved and compiled into a control context block.
4. **Final Extraction (Pass 2):** The primary model processes the complete context enriched with the control data from the RAG subsystem and generates a structured JSON object.
5. **Validation and Repair (Inspector):** The output undergoes syntactic and type checking against Pydantic definitions. During this phase, the system verifies literal textual evidence against the source text. If critical fields are missing, the system initiates a repair policy and repeats the inference cycle with a precise error description up to a configured retry limit. Once validation succeeds, the temporary evidence log is discarded to optimize storage, leaving only the clean metadata structure.

## Pipeline Configurations

The system is configured to run in one of four specialized execution modes, providing flexibility for ablation testing, baseline analysis, and full-featured metadata extraction:

### Configuration A (Baseline Simple)
* **Runner Script:** `run-baseline-conf_A.py`
* **Description:** A purely baseline and minimalistic pipeline. It extracts text from PDFs using `pymupdf4llm`, slices it (head/tail technique), and feeds it to an LLM via Ollama to generate a JSON response.
* **Important Note:** This is a **standalone script**, entirely independent of the overall pipeline architecture (it does not use `core`, `rag`, `inspector`, etc.).

### Configuration B (Baseline Advanced)
* **Runner Script:** `run-baseline-conf_B.py`
* **Description:** An advanced baseline that improves upon Configuration A by supporting multiple input formats (PDF, EPUB, DOCX, ODT, TXT). It uses more sophisticated prompting, detailed logging, and deterministic model settings (temperature=0.0).
* **Important Note:** Like Configuration A, this is also a **standalone script** designed to function independently of the main architecture.

### Configuration C (Pipeline without RAG)
* **Runner Script:** `run-pipeline-without_RAG-conf_C.py`
* **Description:** This configuration acts as an ablation study to measure the system's performance *without* external semantic knowledge. It activates the full dual-phase (Pass 1 / Pass 2) inference pipeline, advanced Pydantic schema validation, and the agentic repair loop (Inspector). The RAG subsystem (Librarian/ChromaDB) is intentionally disabled.

### Configuration D (Full Orchestrated Pipeline)
* **Runner Script:** `run-full pipeline-conf_D.py`
* **Description:** The flagship configuration representing the complete, fully-featured system. It is architecturally identical to Configuration C but actively uses the Librarian module to inject verified RAG context from ChromaDB before executing Pass 2 inference.

## Code Quality and Standards

This project is developed with a strict emphasis on high academic integrity, sustainability, and research reproducibility. The source code strictly adheres to the following conventions:

* **PEP 8:** The official style guide for Python code formatting and structure. All scripts have been automatically formatted and linted (`flake8`).
* **PEP 257:** Conventions for Python docstrings. All key classes, methods, and functions are documented following the strict *Google Style Docstrings* format (`pydocstyle`).

Due to consistent compliance with PEP 257, complete technical API documentation can be automatically generated at any time as HTML pages or PDF documents using tools such as `pdoc` or `Sphinx`.

## Project Structure

project/
├── run-baseline-conf_A.py          # Standalone baseline pipeline (PDF only)
├── run-baseline-conf_B.py          # Standalone baseline pipeline (Multi-format)
├── run-pipeline-without_RAG-conf_C.py  # Orchestrated pipeline excluding RAG
├── run-full pipeline-conf_D.py     # Main execution script of the complete pipeline
├── README.md                       # Comprehensive documentation and installation guide
├── prompts/                        # Repository of system instructions for the LLM
│   ├── system_prompt.txt           # Main instructional prompt for structured output (Pass 2)
│   └── pass1_candidates_prompt.txt # Prompt for the preparatory candidate selection phase (Pass 1)
├── config/                         # Global settings and loaders
│   ├── settings.py                 # Environment configuration and OLLAMA_OPTIONS directives
│   └── prompt_loader.py            # Encapsulated logic for dynamic prompt loading
├── core/                           # Core orchestration and repair mechanisms
│   ├── pipeline.py                 # Control logic for batch processing and retry policies
│   └── inspector.py                # State analysis, syntactic repair, and evidence validation
├── extraction/                     # Layer responsible for text acquisition
│   ├── scribe.py                   # Text extractor from PDF documents with OCR routine support
│   └── candidates.py               # Algorithm for identification and isolation of key passages
├── llm/                            # Communication interface with language models
│   └── ollama_wrapper.py           # Abstraction of API calls for the local Ollama environment
├── postprocessing/                 # Data cleaning and linguistic unification
│   └── cleaning.py                 # Deterministic cleaning functions after inference
├── rag/                            # Module for semantic search and vector operations
│   ├── librarian.py                # Context manager for working with external knowledge
│   └── ingest.py                   # Script for text transformation and Chroma DB population
├── schemas/                        # Database and metadata templates
│   └── metadata.py                 # Strict schema definition for Dublin Core elements
├── utils/                          # Auxiliary system tools, data acquisition, and evaluation
│   ├── openalex_dataset_download.py # Asynchronous utility for dataset acquisition and randomization
│   ├── f1_evaluate.py              # Evaluation utility for F1 score and telemetry calculation
│   ├── system.py                   # Routines for operating system interactions and VRAM release
│   └── logging_setup.py            # Central configuration for detailed system logging
└── data/                           # Persistent data layer of the experiment
    ├── input/                      # Input documents intended for processing (you populate this)
    ├── ground/                     # Reference JSON metadata used as Ground Truth (you populate this)
    ├── jsonl_docs/                 # Source JSONL controlled vocabularies consumed by rag/ingest.py
    ├── output/                     # Generated metadata JSON files and raw text extractions
    ├── vector_db/                  # Persistent storage of the Chroma DB vector database
    ├── pdf_openalex/                # Raw PDFs downloaded by utils/openalex_dataset_download.py
    ├── metadata_openalex/           # Raw ground truth JSON downloaded by utils/openalex_dataset_download.py
    └── pipeline.log                # Comprehensive system log of the execution flow


## System Requirements

* Python 3.10 or newer
* A running instance of the local Ollama platform
* Downloaded local models (default configuration):
  * `hf.co/unsloth/gemma-4-E4B-it-GGUF:Q4_K_M` (default `LLM_MODEL` for orchestration/inference, see `config/settings.py`)
  * `nomic-embed-text` (for generating embeddings with a fixed vector dimension of 768)
* `pandoc` installed and available on `PATH` (required by `run-baseline-conf_B.py` to convert `.docx`/`.odt` files to Markdown; PDF/EPUB/TXT do not need it)

## Installation and Setup

1. Clone the project repository and navigate to its root directory.
2. Create and activate a Python virtual environment:

```bash
python -m venv venv
source venv/bin/activate  # On Windows systems: venv\Scripts\activate
```

3. Install the required dependencies:

```bash
pip install -r requirements.txt
```

### Data Transformation and Vector Population (Ingest)

Before executing the primary extraction pipeline (Configurations C and D), it is essential that the local ChromaDB vector database contains the generated index of controlled vocabularies. The system requires the creation of a collection named exactly `metadata_lookup_vocabularies` with a fixed vector dimension of 768.

To transform the source JSONL data (language codes, licensing terms, COAR vocabularies, institutional registries) and perform bulk ingestion into the database, run the dedicated script:

```bash
python rag/ingest.py
```

The script automatically verifies embedding dimension integrity, eliminates syntactic noise, performs semantic atomization, applies SHA 256 deduplication, and indexes structured metadata under the `lookup_domain` control attribute.

### Test Corpus Preparation (Optional)

If your research requires the automated acquisition of a reference dataset from the OpenAlex repository, including downloading open access PDF files and compiling comparative ground truth JSON structures, execute the asynchronous utility:

```bash
python utils/openalex_dataset_download.py
```

By default the script targets `TARGET_PDF_COUNT = 200` documents (adjust the constant in the
script if you need the N = 300 corpus size used in the accompanying study) and writes its
results to `data/pdf_openalex/` (downloaded PDFs) and `data/metadata_openalex/` (per-document
ground truth JSON, plus a consolidated `katalog_ground_truth_openalex.json`). These are **not**
the same folders the pipeline and evaluator read from, so before running the pipeline you must
move the downloaded PDFs into `data/input/` and the corresponding ground truth JSON files into
`data/ground/`.

### Running the Extraction Pipeline

1. Place the documents intended for processing (supported formats include `.pdf`, `.epub`, `.docx`, `.odt`, `.txt`) into the `data/input/` directory.
2. Execute the batch processing via the main orchestrator (e.g., Configuration D):

```bash
python "run-full pipeline-conf_D.py"
```

The resulting processed metadata will be automatically saved into the `data/output/` directory as JSON files encoded in UTF-8.

### Performance Evaluation and F1 Score Calculation

Following the completion of a batch run, you can execute the independent evaluation utility. The script loads the generated data from `data/output/`, performs linguistic normalization, and benchmarks the values against the reference ground truth files located in `data/ground/`:

```bash
python utils/f1_evaluate.py
```

The utility calculates precision, recall, the final Macro $F_1$ score, and displays the telemetry throughput of the pipeline.

## Output Format Example

For each processed document, the system generates a strictly validated file with a `_metadata.json` suffix. The temporary `evidence_log` utilized during the verification stage is discarded upon successful validation, so the final output file contains only the flat, clean metadata structure (the fields below, with no `metadata` wrapper key):

```json
{
  "title": "Algoritmy automatizovanej extrakcie bibliografických dát",
  "alternative_title": null,
  "authors": [
    "Ján Kováč",
    "Peter Malý"
  ],
  "contributors": [],
  "issued": "2025",
  "publisher": "Žilinská univerzita v Žiline",
  "publication_place": "Žilina",
  "resource_type": "article",
  "language": "sk",
  "doi": "10.1234/uniza.2025.1",
  "isbn": null,
  "issn_print": "1335-4205",
  "issn_electronic": null,
  "persistent_uri": null,
  "abstract": "Tento článok analyzuje možnosti integrácie lokálnych jazykových modelov...",
  "subjects": [
    "digitálne knižnice",
    "spracovanie prirodzeného jazyka"
  ],
  "rights_uri": "https://creativecommons.org/licenses/by/4.0/"
}
```

## API Documentation Generation

By adhering strictly to the PEP 257 standard, you can generate the technical documentation directly from the source code with a single command:

```bash
pip install pdoc
python -m pdoc -o docs core extraction rag schemas utils
```

This creates an organized HTML documentation tree (written to `docs/`) of all modules, classes,
and methods deployed within this architecture. Note: the modern `pdoc` package (installed above)
no longer supports a `--html` flag — use `-o <directory>` instead. Run it as `python -m pdoc`
rather than the bare `pdoc` command, since `core/`, `extraction/`, `rag/`, `schemas/`, and `utils/`
have no `__init__.py` and are only resolved correctly as namespace packages when the current
directory is added to `sys.path`, which `python -m` does automatically.
