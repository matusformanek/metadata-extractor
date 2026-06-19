import json
from pathlib import Path
import ollama
import pymupdf4llm

MODEL_NAME = "hf.co/mradermacher/MeXtract-3B-GGUF:Q8_0"
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
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    for file_path in sorted(INPUT_DIR.glob("*.pdf")):
        print(f"[PROCESSING] {file_path.name}")
        try:
            # Extrakcia textu vo formáte Markdown pomocou pymupdf4llm
            # Konvertujeme Path objekt na string, ktorý funkcia vyžaduje
            extracted_text = pymupdf4llm.to_markdown(str(file_path))[:8000]

            if not extracted_text.strip():
                print(f"[SKIP] {file_path.name}")
                continue

            response = ollama.generate(
                model=MODEL_NAME,
                system=SYSTEM_PROMPT,
                format="json",
                prompt=extracted_text
            )

            parsed_json = json.loads(response.get("response", "").strip())

            output_file = OUTPUT_DIR / f"{file_path.stem}_metadata.json"
            with output_file.open("w", encoding="utf-8") as f:
                json.dump(parsed_json, f, indent=2, ensure_ascii=False)

            print(f"[SUCCESS] {output_file}")

        except json.JSONDecodeError as e:
            print(f"[ERROR] JSON parse failed for {file_path.name}: {e}")
        except Exception as e:
            print(f"[FAILURE] {file_path.name}: {e}")

if __name__ == "__main__":
    run_baseline_pipeline()