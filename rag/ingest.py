# -*- coding: utf-8 -*-
"""Ingestion pipeline for structured JSONL lookup vocabularies into ChromaDB.

This module flattens multi-level JSON objects, computes deterministic hashes
for deduplication, maps valid scalar variables to vector store metadata,
and batches records into a local ChromaDB collection using Ollama embeddings.
"""

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Set, Tuple

from config.settings import CHROMA_COLLECTION, CHROMA_PATH, EMBED_MODEL

DEFAULT_DATA_DIR = Path("data/jsonl_docs")
DEFAULT_BATCH_SIZE = 128
MIN_CONTENT_LENGTH = 10
MAX_TEXT_FIELD_CHARS = 4000

PREFERRED_TEXT_FIELDS = [
    "canonical_value",
    "label",
    "name",
    "title",
    "text",
    "content",
    "definition",
    "description",
    "scope_note",
    "usage_note",
    "aliases",
    "keywords",
    "source",
    "type",
]

TECHNICAL_FIELDS = {
    "id",
    "record_id",
    "source_file",
    "line_number",
    "content_hash",
    "lookup_domain",
}


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments for the vocabulary ingestion execution.

    Returns:
        argparse.Namespace: Validated argument namespace containing data paths,
            collection targets, chunk sizing, and operational modes.
    """
    parser = argparse.ArgumentParser(
        description="Ingest JSONL lookup vocabularies into ChromaDB."
    )
    parser.add_argument(
        "--data-dir",
        default=str(DEFAULT_DATA_DIR),
        help="Directory containing .jsonl files.",
    )
    parser.add_argument(
        "--db-dir",
        default=CHROMA_PATH,
        help="ChromaDB persistent directory.",
    )
    parser.add_argument(
        "--collection",
        default=CHROMA_COLLECTION,
        help="ChromaDB collection name.",
    )
    parser.add_argument(
        "--mode",
        choices=["rebuild", "upsert", "append"],
        default="upsert",
        help="rebuild deletes the collection first; upsert reuses IDs.",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=DEFAULT_BATCH_SIZE,
        help="Number of documents inserted per batch.",
    )
    parser.add_argument(
        "--allow-missing-domain",
        action="store_true",
        help="Load records without lookup_domain instead of skipping them.",
    )
    return parser.parse_args()


def normalize_text(value: str) -> str:
    """Normalize whitespace characters and strip carriage returns within text.

    Args:
        value (str): Raw target string containing irregular whitespace.

    Returns:
        str: Cleansed string with single space padding.
    """
    return " ".join(value.replace("\r", "\n").split())


def serialize_value(value: Any) -> str:
    """Serialize scalar types, arrays, or dictionaries into clean text markers.

    Args:
        value (Any): Input structure derived from JSON row tracking.

    Returns:
        str: String representation appropriate for embedding generation.
    """
    if value is None:
        return ""
    if isinstance(value, str):
        return normalize_text(value)
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, (int, float)):
        return str(value)
    if isinstance(value, list):
        return "; ".join(
            item for item in (serialize_value(v) for v in value) if item
        )
    if isinstance(value, dict):
        return json.dumps(value, ensure_ascii=False, sort_keys=True)
    return normalize_text(str(value))


def flatten_record(value: Any, prefix: str = "") -> Dict[str, Any]:
    """Recursively flatten nested dictionary schemas using dotted key notation.

    Args:
        value (Any): The nested object or scalar leaf node to trace.
        prefix (str): Accumulated dotted path tracking parent keys. Defaults to "".  # noqa: E501

    Returns:
        Dict[str, Any]: Single layer map containing primitive components.
    """
    flat: Dict[str, Any] = {}
    if isinstance(value, dict):
        for key, nested in value.items():
            next_key = f"{prefix}.{key}" if prefix else str(key)
            flat.update(flatten_record(nested, next_key))
    else:
        flat[prefix] = value
    return flat


def safe_metadata_value(value: Any) -> Optional[Any]:
    """Filter out non-primitive types prohibited by Chroma metadata constraints.  # noqa: E501

    Args:
        value (Any): Arbitrary object layer evaluation target.

    Returns:
        Optional[Any]: Valid primitive scalar instance, otherwise None.
    """
    if isinstance(value, (str, int, float, bool)):
        return value
    return None


def compute_content_hash(record: Dict[str, Any]) -> str:
    """Generate a stable SHA-256 fingerprint from serialized object layers.

    Args:
        record (Dict[str, Any]): Dictionary schema representing a single record.  # noqa: E501

    Returns:
        str: Hexadecimal string digest used for idempotent data checks.
    """
    canonical = json.dumps(record, ensure_ascii=False, sort_keys=True)
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def get_lookup_domain(record: Dict[str, Any]) -> str:
    """Retrieve the domain classification from root attributes or sub-elements.

    Args:
        record (Dict[str, Any]): Input data record.

    Returns:
        str: Standardized context scope string.
    """
    metadata = record.get("metadata")
    value = record.get("lookup_domain")
    if not value and isinstance(metadata, dict):
        value = metadata.get("lookup_domain")
    return serialize_value(value)


def build_record_id(
    record: Dict[str, Any],
    source_file: str,
    line_number: int,
) -> str:
    """Establish a deterministic identity string using explicit or derived keys.  # noqa: E501

    Args:
        record (Dict[str, Any]): Target record definition block.
        source_file (str): The origin file context name.
        line_number (int): The absolute parsing line location index.

    Returns:
        str: Unique identifier string for vector collection persistence.
    """
    raw_id = serialize_value(record.get("id"))
    if raw_id:
        return raw_id
    digest = compute_content_hash(record)[:24]
    return f"{Path(source_file).stem}:{line_number}:{digest}"


def prettify_field_name(field_name: str) -> str:
    """Transform internal dotted dictionary keys into legible title frames.

    Args:
        field_name (str): Original structural parameter text segment.

    Returns:
        str: Clean human-readable line header phrase.
    """
    return field_name.split(".")[-1].replace("_", " ").capitalize()


def build_page_content(record: Dict[str, Any], lookup_domain: str) -> str:
    """Construct the main textual block to be embedded by the vector database.

    Sorts fields using priority arrangements defined in preferred lists, drops
    pure technical artifacts, and enforces character bounds.

    Args:
        record (Dict[str, Any]): The raw source schema element.
        lookup_domain (str): Domain context information.

    Returns:
        str: Formatted context block text ready for inference embeddings.
    """
    flat = flatten_record(record)
    lines: List[str] = []
    seen: Set[str] = set()

    if lookup_domain:
        lines.append(f"Lookup domain: {lookup_domain}")

    ordered_fields: List[str] = []
    for preferred in PREFERRED_TEXT_FIELDS:
        for field in flat:
            if (
                field.split(".")[-1] == preferred
                and field not in ordered_fields
            ):
                ordered_fields.append(field)

    for field in sorted(flat):
        if field not in ordered_fields:
            ordered_fields.append(field)

    for field in ordered_fields:
        short = field.split(".")[-1]
        if short in TECHNICAL_FIELDS:
            continue

        text = serialize_value(flat[field])
        if not text:
            continue
        if len(text) > MAX_TEXT_FIELD_CHARS:
            text = text[:MAX_TEXT_FIELD_CHARS].rstrip() + " ..."

        dedupe_key = text.casefold()
        if dedupe_key in seen:
            continue
        seen.add(dedupe_key)

        lines.append(f"{prettify_field_name(field)}: {text}")

    return "\n".join(lines).strip()


def build_metadata(
    record: Dict[str, Any],
    source_file: str,
    line_number: int,
    record_id: str,
    content_hash: str,
    lookup_domain: str,
) -> Dict[str, Any]:
    """Map unstructured payload definitions into flat vector storage metadata.

    Args:
        record (Dict[str, Any]): Base source structure map.
        source_file (str): Name of file origin.
        line_number (int): Row number counter index.
        record_id (str): Derived primary collection key tracking tag.
        content_hash (str): Deduplication hash value.
        lookup_domain (str): Vocabulary scope identification string.

    Returns:
        Dict[str, Any]: Validated tracking object mapping primitives.
    """
    metadata: Dict[str, Any] = {
        "record_id": record_id,
        "source_file": source_file,
        "line_number": line_number,
        "content_hash": content_hash,
    }
    if lookup_domain:
        metadata["lookup_domain"] = lookup_domain

    for field, raw in flatten_record(record).items():
        if field in metadata:
            continue
        val = safe_metadata_value(raw)
        if val is not None:
            metadata[field] = val

    return metadata


def iter_jsonl_records(
    data_dir: Path,
) -> Iterable[Tuple[str, int, Dict[str, Any]]]:
    """Yield parsed JSON dictionary components discovered inside data target areas.  # noqa: E501

    Args:
        data_dir (Path): System directory location where files reside.

    Yields:
        Iterator[Tuple[str, int, Dict[str, Any]]]: Combination containing filename,  # noqa: E501
            current line number index, and the parsed content frame.

    Raises:
        FileNotFoundError: If the specified path location target does not exist.  # noqa: E501
    """
    if not data_dir.exists():
        raise FileNotFoundError(f"Directory does not exist: {data_dir}")

    for path in sorted(data_dir.glob("*.jsonl")):
        with path.open("r", encoding="utf-8-sig") as handle:
            for line_number, line in enumerate(handle, start=1):
                line = line.strip()
                if not line:
                    continue
                try:
                    record = json.loads(line)
                except json.JSONDecodeError as exc:
                    print(
                        f"[WARN] Skipping invalid JSON: "
                        f"{path.name}:{line_number}: {exc}"
                    )
                    continue
                if isinstance(record, dict):
                    yield path.name, line_number, record
                else:
                    print(
                        f"[WARN] Skipping non-object row: "
                        f"{path.name}:{line_number}"
                    )


def load_documents(
    data_dir: Path,
    allow_missing_domain: bool,
) -> Tuple[List[Dict[str, Any]], List[str], Dict[str, int]]:
    """Evaluate and format raw files into structured document payloads.

    Ensures filtering controls handle empty text structures, identity
    clashes, or missing scope constraints seamlessly.

    Args:
        data_dir (Path): Input filesystem tracking pointer.
        allow_missing_domain (bool): Evaluation flag determining whether to
            skip records without a defined lookup domain.

    Returns:
        Tuple[List[Dict[str, Any]], List[str], Dict[str, int]]: Combined metrics and  # noqa: E501
            document components prepared for vector database integration.
    """
    documents: List[Dict[str, Any]] = []
    ids: List[str] = []
    seen_ids: Set[str] = set()
    seen_hashes: Set[str] = set()

    stats = {
        "records_seen": 0,
        "records_loaded": 0,
        "missing_lookup_domain": 0,
        "duplicates_by_id": 0,
        "duplicates_by_content": 0,
        "empty_or_short": 0,
    }

    for source_file, line_number, record in iter_jsonl_records(data_dir):
        stats["records_seen"] += 1
        lookup_domain = get_lookup_domain(record)
        if not lookup_domain and not allow_missing_domain:
            stats["missing_lookup_domain"] += 1
            continue

        record_id = build_record_id(record, source_file, line_number)
        content_hash = compute_content_hash(record)
        if record_id in seen_ids:
            stats["duplicates_by_id"] += 1
            continue
        if content_hash in seen_hashes:
            stats["duplicates_by_content"] += 1
            continue

        page_content = build_page_content(record, lookup_domain)
        if len(page_content) < MIN_CONTENT_LENGTH:
            stats["empty_or_short"] += 1
            continue

        metadata = build_metadata(
            record,
            source_file,
            line_number,
            record_id,
            content_hash,
            lookup_domain,
        )
        documents.append({"page_content": page_content, "metadata": metadata})
        ids.append(record_id)
        seen_ids.add(record_id)
        seen_hashes.add(content_hash)
        stats["records_loaded"] += 1

    return documents, ids, stats


def reset_collection(db_dir: Path, collection_name: str) -> None:
    """Purge target storage setups inside persistent storage indexes.

    Args:
        db_dir (Path): Location point managing target database file structures.
        collection_name (str): Identifier name tag targeting data drop segments.  # noqa: E501
    """
    try:
        import chromadb

        client = chromadb.PersistentClient(path=str(db_dir))
        client.delete_collection(collection_name)
        print(f"[INFO] Deleted existing collection: {collection_name}")
    except Exception as exc:
        print(f"[INFO] Collection reset skipped: {exc}")


def batched(items: List[Any], size: int) -> Iterable[List[Any]]:
    """Segment item arrays into smaller chunk blocks for processing efficiency.

    Args:
        items (List[Any]): Complete database storage context matrix.
        size (int): Max element size tracking parameters per window segment.

    Yields:
        Iterable[List[Any]]: Segment block slicing sequence references.
    """
    for index in range(0, len(items), size):
        yield items[index: index + size]


def ingest() -> None:
    """Execute the core workflow loop for vocabulary processing.

    Initializes argument configuration parameters, parses raw dictionary files,
    configures HNSW cosine similarity parameters, and populates Chroma DB index maps.  # noqa: E501
    """
    args = parse_args()
    data_dir = Path(args.data_dir)
    db_dir = Path(args.db_dir)

    print("=== LOOKUP VOCABULARY INGEST ===")
    print(f"JSONL directory : {data_dir}")
    print(f"ChromaDB path   : {db_dir}")
    print(f"Collection      : {args.collection}")
    print(f"Embedding model : {EMBED_MODEL}")
    print(f"Mode            : {args.mode}")
    print("-" * 60)

    if args.mode == "rebuild":
        reset_collection(db_dir, args.collection)
    elif args.mode == "append" and db_dir.exists():
        print("[INFO] Append mode: existing collection will be reused.")
    elif args.mode == "upsert":
        print(
            "[INFO] Upsert mode: existing IDs will be deleted before insert."
        )

    documents, ids, stats = load_documents(data_dir, args.allow_missing_domain)
    print(f"[INFO] Load stats: {stats}")
    if not documents:
        print("[WARN] Nothing to ingest.")
        return

    from langchain_chroma import Chroma
    from langchain_core.documents import Document
    from langchain_ollama import OllamaEmbeddings

    embeddings = OllamaEmbeddings(model=EMBED_MODEL)
    embed_dim = len(embeddings.embed_query("embedding dimension check"))
    print(f"[INFO] Embedding dimension: {embed_dim}")

    vector_db = Chroma(
        collection_name=args.collection,
        persist_directory=str(db_dir),
        embedding_function=embeddings,
        collection_metadata={
            # Explicit distance metric configuration for ChromaDB HNSW index.
            "hnsw:space": "cosine",
            "embedding_model": EMBED_MODEL,
            "embedding_dim": embed_dim,
            "purpose": "metadata lookup vocabularies",
        },
    )

    inserted = 0
    for doc_batch, id_batch in zip(
        batched(documents, args.batch_size),
        batched(ids, args.batch_size),
    ):
        langchain_docs = [
            Document(
                page_content=document["page_content"],
                metadata=document["metadata"],
            )
            for document in doc_batch
        ]
        if args.mode == "upsert":
            try:
                vector_db.delete(ids=id_batch)
            except Exception:
                pass
        vector_db.add_documents(documents=langchain_docs, ids=id_batch)
        inserted += len(doc_batch)
        print(f"[INFO] Inserted {inserted}/{len(documents)}")

    counts: Dict[str, int] = {}
    for document in documents:
        domain = document["metadata"].get("lookup_domain", "")
        counts[domain] = counts.get(domain, 0) + 1

    print(f"[INFO] lookup_domain counts: {counts}")
    print(f"[OK] Ingest complete: {inserted} documents")


if __name__ == "__main__":
    ingest()
