"""Pipeline for automated bibliographic metadata extraction from PDF documents.

Configuration: metadata-extractor-configuration A

Note:
    This script can be used completely independently as a standalone tool
    without the need for any other scripts or components from the overall
    architecture. It does not invoke or rely on any other architectural scripts.  # noqa: E501

This script extracts text from PDF files using pymupdf4llm, applies a sandwich
sampling technique to focus on the introduction and conclusion parts, and uses
a local LLM via Ollama to generate structured Dublin Core metadata in JSON format.  # noqa: E501
"""

import json
from pathlib import Path
import ollama
import pymupdf4llm

# Configuration constants aligned with PEP 8 naming conventions
MODEL_NAME = "hf.co/unsloth/gemma-4-E4B-it-GGUF:Q4_K_M"
INPUT_DIR = Path("data/input")
OUTPUT_DIR = Path("data/output")

SYSTEM_PROMPT = """Extract bibliographic information from text.

Target JSON Schema Structure:
{
  "title": "",
  "alternative_title": "",
  "authors": [],
  "contributors": [],
  "issued": "",
  "publisher": "",
  "publication_place": "",
  "resource_type": "",
  "language": "",
  "doi": "",
  "isbn": "",
  "issn_print": "",
  "issn_electronic": "",
  "persistent_uri": "",
  "abstract": "",
  "subjects": [],
  "rights_uri": ""
}
"""


def run_baseline_pipeline() -> None:
    """Process all PDF files in the input directory and extract metadata.

    The function reads each PDF file, extracts its content, creates a sandwich
    view of the document (first 7000 and last 1000 characters), sends it to
    the Ollama model, and saves the parsed JSON response.
    """
    # Ensure the output directory exists before processing
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # Process all PDF documents in alphabetical order
    for file_path in sorted(INPUT_DIR.glob("*.pdf")):
        print(f"[PROCESSING] {file_path.name}")
        try:
            # Convert PDF pages to a single Markdown string
            full_text = pymupdf4llm.to_markdown(str(file_path))

            # Apply sandwich sampling if the document exceeds the 8000
            # character limit
            if len(full_text) > 8000:
                introduction = full_text[:7000]
                conclusion = full_text[-1000:]

                # Combine sections with direct introduction and conclusion tags
                extracted_text = (
                    f"This is the introduction of the document:\n{introduction}\n"  # noqa: E501
                    f"This is the conclusion of the document:\n{conclusion}"
                )
            else:
                # Keep the original text intact if it is short enough
                extracted_text = full_text

            # Skip processing if the extracted text block is empty
            if not extracted_text.strip():
                print(f"[SKIP] {file_path.name} - No text extracted.")
                continue

            # Query the local LLM using the Ollama API
            response = ollama.generate(
                model=MODEL_NAME,
                system=SYSTEM_PROMPT,
                format="json",
                prompt=extracted_text,
            )

            # Extract and parse the JSON string from the response object
            raw_response = response.get("response", "").strip()
            parsed_json = json.loads(raw_response)

            # Construct the output path and save the structured metadata
            output_file = OUTPUT_DIR / f"{file_path.stem}_metadata.json"
            with output_file.open("w", encoding="utf-8") as f:
                json.dump(parsed_json, f, indent=2, ensure_ascii=False)

            print(f"[SUCCESS] Saved metadata to {output_file}")

        except json.JSONDecodeError as e:
            print(f"[ERROR] JSON parsing failed for {file_path.name}: {e}")
        except Exception as e:
            print(f"[FAILURE] Unexpected error processing {file_path.name}: {e}")


if __name__ == "__main__":
    run_baseline_pipeline()
