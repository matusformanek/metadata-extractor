# -*- coding: utf-8 -*-
"""OpenAlex Asynchronous Dataset Ingestion Tool.

This module automates the acquisition of open-access scientific publications
and their respective metadata fields from the OpenAlex API. It features
asynchronous download pipelines, query topic rotation to ensure dataset
diversity, abstract reconstruction from inverted indexes, and strict mapping
to a localized Dublin Core compliant schema.
"""

import asyncio
import json
import os
from pathlib import Path
import random

import aiohttp

# ==============================================================================  # noqa: E501
# API AND BOT CONFIGURATION
# ==============================================================================  # noqa: E501

# Target OpenAlex API endpoint for academic works
OPENALEX_API_URL = "https://api.openalex.org/works"

# User email required to gain access to the OpenAlex 'polite pool'
USER_EMAIL = "your@email-address"

# Professional User-Agent identifier string for API identification
USER_AGENT = f"OpenScienceResearchBot/4.2 (mailto:{USER_EMAIL})"

# Maximum number of concurrent network connections permitted
MAX_CONCURRENT_DOWNLOADS = 10

# Total number of successfully downloaded PDFs required to complete execution
TARGET_PDF_COUNT = 200

# Directory routing configuration
BASE_DIR = Path(".")
METADATA_DIR = BASE_DIR / "data/metadata_openalex"
PDF_DIR = BASE_DIR / "data/pdf_openalex"

# Ensure output structures exist on the file system before initialization
METADATA_DIR.mkdir(exist_ok=True)
PDF_DIR.mkdir(exist_ok=True)

# Comprehensive list of diverse scientific disciplines used for target sampling
SCIENTIFIC_FIELDS = [
    "computer science",
    "artificial intelligence",
    "cybernetics",
    "information science",
    "library science",
    "data science",
    "machine learning",
    "deep learning",
    "natural language processing",
    "robotics",
    "automation",
    "internet of things",
    "physics",
    "quantum physics",
    "astrophysics",
    "chemistry",
    "biology",
    "molecular biology",
    "genetics",
    "neuroscience",
    "medicine",
    "gastroenterology",
    "clinical medicine",
    "public health",
    "sociology",
    "economics",
    "education",
    "psychology",
    "history",
    "philosophy",
    "political science",
    "law",
    "mathematics",
    "statistics",
    "engineering",
    "electrical engineering",
    "civil engineering",
    "mechanical engineering",
    "biomedical engineering",
    "environmental science",
    "ecology",
    "geology",
    "geography",
    "agriculture",
    "forestry",
    "veterinary medicine",
    "pharmacy",
    "nursing",
    "dentistry",
    "anthropology",
    "archaeology",
    "linguistics",
    "literature",
    "music",
    "art",
    "architecture",
    "business",
    "management",
    "accounting",
    "finance",
    "marketing",
    "healthcare",
    "epidemiology",
    "bioinformatics",
    "computational biology",
    "biochemistry",
    "materials science",
    "nanotechnology",
    "energy",
    "climate science",
    "oceanography",
    "meteorology",
    "cognitive science",
    "cognitive neuroscience",
    "behavioral science",
    "communications",
    "journalism",
    "media studies",
    "urban planning",
    "transportation",
    "logistics",
    "food science",
    "nutrition",
    "toxicology",
    "immunology",
    "oncology",
    "cardiology",
    "pediatrics",
    "geriatrics",
    "psychiatry",
    "dermatology",
    "orthopedics",
    "sports science",
    "recreation",
    "tourism",
    "criminal justice",
    "criminology",
    "security studies",
    "international relations",
    "development studies",
    "area studies",
]


# ==============================================================================  # noqa: E501
# HELPER DATA PROCESSING FUNCTIONS
# ==============================================================================  # noqa: E501


def reconstruct_abstract(inverted_index: dict) -> str:
    """Convert an OpenAlex abstract inverted index back into plain text.

    This reverse-engineering process reads the token positions map and builds
    a sequentially correct, human-readable abstract string.

    Args:
        inverted_index (dict): The position mapping dictionary from the API.

    Returns:
        str: Reconstructed linear text string or empty if map is missing.
    """
    if not inverted_index:
        return ""

    word_index = {}

    # Map each word token back to its numerical index locations
    for word, pos_list in inverted_index.items():
        for pos in pos_list:
            word_index[pos] = word

    # Sort the dictionary keys to stitch the words together chronologically
    return " ".join(word_index[pos] for pos in sorted(word_index.keys()))


def map_to_custom_schema(item: dict, pdf_url: str) -> dict:
    """Transform complex OpenAlex API records into a streamlined schema format.

    Extracts core entities, purges administrative prefixes from DOIs, and
    restructures authorship arrays and keywords for simpler processing.

    Args:
        item (dict): The original raw JSON record delivered by OpenAlex.
        pdf_url (str): The valid target location resolved for the PDF asset.

    Returns:
        dict: Normalized flat metadata record layout.
    """
    # Extract structural authorship representations
    authors = []
    for authorship in item.get("authorships", []):
        name = authorship.get("author", {}).get("display_name")
        if name:
            authors.append(name)

    # Gather user-defined author keywords instead of high-level concepts
    author_keywords = [
        kw.get("display_name") for kw in item.get("keywords", [])
    ]

    # Clean the DOI string to store only the strict identifier suffix
    raw_doi = item.get("doi", "")
    clean_doi = raw_doi.replace("https://doi.org/", "") if raw_doi else ""

    return {
        "doi": clean_doi,
        "title": item.get("display_name", ""),
        "authors": authors,
        "abstract": reconstruct_abstract(item.get("abstract_inverted_index")),
        "issued": str(item.get("publication_year", "N/A")),
        "type": item.get("type", "N/A"),
        "subject": author_keywords,
        "language": item.get("language", "N/A"),
        "uri": pdf_url,
        "rights.uri": item.get("primary_location", {}).get("license", "N/A"),
    }


# ==============================================================================  # noqa: E501
# ASYNCHRONOUS NETWORK OPERATIONS
# ==============================================================================  # noqa: E501


async def download_pdf(
    session: aiohttp.ClientSession,
    pdf_url: str,
    output_path: Path,
    semaphore: asyncio.Semaphore,
) -> tuple[bool, str]:
    """Asynchronously download a single PDF file with integrated safety rules.

    Employs connection locks via a semaphore, verifies MIME headers, and
    deletes corrupt or abnormally tiny files to protect against paywalls.

    Args:
        session (aiohttp.ClientSession): The underlying active aiohttp pool.
        pdf_url (str): Remote address location of the target document.
        output_path (Path): File system destination where data will be written.
        semaphore (asyncio.Semaphore): Concurrency controller instance.

    Returns:
        tuple[bool, str]: Success boolean combined with an analytical message.
    """
    async with semaphore:
        try:
            # Query the target server allowing automatic redirect tracking
            async with session.get(
                pdf_url, timeout=30, allow_redirects=True
            ) as response:
                if response.status != 200:
                    return False, f"HTTP {response.status}"

                # Ensure that the resolved Content-Type matches a binary PDF
                content_type = response.headers.get("Content-Type", "").lower()
                if "pdf" not in content_type and not pdf_url.lower().endswith(
                    ".pdf"
                ):
                    return False, "Content type is not PDF"

                # Read raw stream data into system memory buffer
                data = await response.read()
                with open(output_path, "wb") as f:
                    f.write(data)

                # Penalize and clean up files smaller than 10 Kilobytes
                if output_path.stat().st_size < 10_000:
                    output_path.unlink(missing_ok=True)
                    return False, "PDF too small or invalid structure"

                return True, "OK"

        except asyncio.TimeoutError:
            return False, "Timeout"
        except Exception as e:
            return False, str(e)


# ==============================================================================  # noqa: E501
# MAIN EXECUTION ENGINE ORCHESTRATION
# ==============================================================================  # noqa: E501


async def process_records(target_count: int) -> None:
    """Manage the primary metadata parsing and downloading workflow loop.

    Iterates through randomized topics using OpenAlex sampling filters, parses
    returned arrays, submits async download tasks, and coordinates file storage.  # noqa: E501

    Args:
        target_count (int): Maximum unique documents to load into the dataset.
    """
    semaphore = asyncio.Semaphore(MAX_CONCURRENT_DOWNLOADS)
    headers = {"User-Agent": USER_AGENT}

    successful_count = 0
    catalog = []
    cycle_count = 1

    print(
        f"Goal: Secure exactly {target_count} OA publications from OpenAlex."
    )
    print(
        "Topic rotation is activated (sampling 3 items per field iteration).\n"
    )

    async with aiohttp.ClientSession(headers=headers) as session:
        while successful_count < target_count:
            # Randomly select a scientific discipline to diversify the database
            selected_topic = random.choice(SCIENTIFIC_FIELDS)
            print(f"--- Batch #{cycle_count} | Topic: '{selected_topic}' ---")

            # Configure endpoint criteria requesting specific metadata fields
            params = {
                "search": selected_topic,
                "filter": "has_oa_accepted_or_published_version:true,has_doi:true",  # noqa: E501
                "sample": 3,
                "mailto": USER_EMAIL,
            }

            async with session.get(
                OPENALEX_API_URL, params=params
            ) as response:
                if response.status != 200:
                    print(
                        f"API Error ({response.status}). Moving to next batch..."
                    )
                    cycle_count += 1
                    continue
                data = await response.json()

            items = data.get("results", [])
            tasks = []

            # 1. Collate data structures and initialize tasks
            for item in items:
                raw_doi = item.get("doi", "")
                clean_doi = (
                    raw_doi.replace("https://doi.org/", "") if raw_doi else ""
                )

                if not clean_doi or successful_count >= target_count:
                    continue

                # Locate the most appropriate public access URL
                best_oa = item.get("best_oa_location") or {}
                pdf_url = best_oa.get("pdf_url")

                if not pdf_url:
                    pdf_url = item.get("open_access", {}).get("oa_url")

                if not pdf_url:
                    continue

                # Generate a safe filename format replacing folder delimiters
                safe_name = clean_doi.replace("/", "_")
                pdf_path = PDF_DIR / f"{safe_name}.pdf"

                metadata_record = map_to_custom_schema(item, pdf_url)

                # Create an asynchronous task execution context
                task = asyncio.create_task(
                    download_pdf(session, pdf_url, pdf_path, semaphore)
                )
                tasks.append((task, metadata_record, pdf_path, safe_name))

            # 2. Sequential wait processing loop
            for task, metadata_record, pdf_path, safe_name in tasks:
                if successful_count >= target_count:
                    break

                success, status = await task

                if success:
                    successful_count += 1
                    # Append file indicators to trace system records locally
                    metadata_record["_local.pdf_path"] = str(pdf_path)

                    metadata_path = METADATA_DIR / f"{safe_name}.json"
                    with open(metadata_path, "w", encoding="utf-8") as f:
                        json.dump(
                            metadata_record, f, indent=4, ensure_ascii=False
                        )

                    catalog.append(metadata_record)
                    print(
                        f"[{successful_count}/{target_count}] ✓ PDF Saved: {metadata_record['doi']}"  # noqa: E501
                    )

            cycle_count += 1

    # Consolidate complete execution dataset records into a master log
    with open(
        "katalog_ground_truth_openalex.json", "w", encoding="utf-8"
    ) as f:
        json.dump(catalog, f, indent=4, ensure_ascii=False)

    print(
        f"\nCOMPLETE: Dataset built successfully with {successful_count} documents."  # noqa: E501
    )


if __name__ == "__main__":
    # Ensure optimal event loop tracking adjustments for Windows environments
    if os.name == "nt":
        asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())
    asyncio.run(process_records(TARGET_PDF_COUNT))
