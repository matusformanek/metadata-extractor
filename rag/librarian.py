# -*- coding: utf-8 -*-
"""RAG helper for controlled vocabulary lookup in ChromaDB.

This module provides the database abstraction layer through the Librarian class,
which handles local vector storage probing, explicit metadata domain filtering,
and query semantic augmentation using the Ollama embedding api.
"""

import logging
from typing import Any, Dict, List

from config.settings import (
    CHROMA_COLLECTION,
    CHROMA_PATH,
    EMBED_MODEL,
    OLLAMA_API_URL,
    RAG_FIELD_CONFIG,
)
from utils.system import release_vram


class Librarian:
    """Retrieve controlled vocabulary hints from the local ChromaDB store.

    Manages connections to vector indexes, executes field-restricted cosine
    similarity lookups, handles embedding generation over HTTP, and builds
    context injection blocks for the subsequent LLM generation pass.
    """

    def __init__(self) -> None:
        """Connect to the configured vector collection if it is available."""
        self._init_error = ""

        try:
            import chromadb
            from chromadb.utils import embedding_functions

            client = chromadb.PersistentClient(path=CHROMA_PATH)

            emb_fn = embedding_functions.OllamaEmbeddingFunction(
                model_name=EMBED_MODEL,
                url=OLLAMA_API_URL,
            )

            self.collection = client.get_collection(
                name=CHROMA_COLLECTION,
                embedding_function=emb_fn,
            )

            self._vectorstore = None
            self._collection_count = self.collection.count()
            self._domain_counts = self._load_domain_counts()

        except Exception as exc:
            self._init_error = str(exc)
            logging.warning(f"ChromaDB initialization failed: {exc}")
            self.collection = None
            self._vectorstore = None
            self._collection_count = 0
            self._domain_counts = {}

    def _load_domain_counts(self) -> Dict[str, int]:
        """Inspect collection metadata to calculate item counts per domain.

        Returns:
            Dict[str, int]: Map of lookup_domain names to their record counts.
        """
        if self.collection is None:
            return {}

        try:
            records = self.collection.get(include=["metadatas"])
            counts: Dict[str, int] = {}

            for metadata in records.get("metadatas", []):
                metadata = metadata or {}
                lookup_domain = (
                    metadata.get("lookup_domain")
                    or metadata.get("metadata.lookup_domain")
                )

                if lookup_domain:
                    key = str(lookup_domain)
                    counts[key] = counts.get(key, 0) + 1

            return counts

        except Exception as exc:
            logging.warning(
                f"Could not inspect ChromaDB lookup domains: {exc}"
            )
            return {}

    def status(self) -> str:
        """Generate a diagnostic state description of the vector backend.

        Returns:
            str: Human-readable diagnostic string containing record counts.
        """
        if self.collection is None:
            detail = f": {self._init_error}" if self._init_error else ""
            return f"unavailable at {CHROMA_PATH}{detail}"

        return (
            f"active collection `{CHROMA_COLLECTION}` at {CHROMA_PATH} "
            f"({self._collection_count} records, "
            f"lookup_domains={self._domain_counts or 'none'})"
        )

    def release_embeddings(self) -> None:
        """Explicitly purge the embedding weights from the host VRAM."""
        release_vram(EMBED_MODEL)

    def _augment_query(self, query_text: str, lookup_domain: str) -> str:
        """Add semantic context prefixes to short queries for stable embeddings.

        Args:
            query_text (str): Raw string slice or term extracted from step one.
            lookup_domain (str): Controlled target vocabulary category identifier.

        Returns:
            str: Expanded query string embedding source.
        """
        text = str(query_text).strip()

        if lookup_domain == "resource_type":
            return f"resource type {text}"
        if lookup_domain == "language":
            return f"language {text}"
        if lookup_domain == "publisher":
            return f"publisher {text}"
        if lookup_domain == "rights_uri":
            return f"license {text}"

        return text

    def query_field(
        self,
        query_text: str,
        lookup_domain: str,
        k: int,
    ) -> List[Dict[str, Any]]:
        """Query the vector database for a specific controlled vocabulary term.

        Args:
            query_text (str): Raw string candidate to evaluate.
            lookup_domain (str): Specific domain constraint name.
            k (int): Limit of nearest neighbor matches to inspect.

        Returns:
            List[Dict[str, Any]]: List containing matched documents and metadata.
        """
        if (
            self.collection is None
            or not query_text
            or not str(query_text).strip()
        ):
            return []

        if self._collection_count <= 0:
            logging.info("ChromaDB query skipped: collection is empty.")
            return []

        query_text_augmented = self._augment_query(query_text, lookup_domain)

        try:
            hits = []

            for where_filter in self._lookup_domain_filters(lookup_domain):
                hits = self._query_with_filter(
                    query_text_augmented,
                    k,
                    where_filter,
                )
                if hits:
                    break

            if hits:
                logging.info(
                    f"RAG matched `{query_text}` "
                    f"(augmented: `{query_text_augmented}`) "
                    f"in `{lookup_domain}`"
                )
                return hits

            self._log_unfiltered_probe(query_text_augmented, lookup_domain, k)
            return []

        except Exception as exc:
            logging.warning(
                f"ChromaDB query_field failed ({lookup_domain}): {exc}"
            )
            return []

    @staticmethod
    def _lookup_domain_filters(
        lookup_domain: str,
    ) -> List[Dict[str, Dict[str, str]]]:
        """Build standard alternative metadata dictionary filter variants.

        Args:
            lookup_domain (str): Target value matching internal schema mappings.

        Returns:
            List[Dict[str, Dict[str, str]]]: Collection of where clauses.
        """
        return [
            {"lookup_domain": {"$eq": lookup_domain}},
            {"metadata.lookup_domain": {"$eq": lookup_domain}},
        ]

    def _query_with_filter(
        self,
        query_text: str,
        k: int,
        where_filter: Dict[str, Dict[str, Any]],
        distance_threshold: float = 0.30,
    ) -> List[Dict[str, Any]]:
        """Perform a low-level vector query bounded by a strict cosine cutoff.

        Args:
            query_text (str): Augmented instruction string.
            k (int): Total results requested from database.
            where_filter (Dict): Metadata dict expression evaluated by ChromaDB.
            distance_threshold (float): Cutoff boundary rejecting loose weights.

        Returns:
            List[Dict[str, Any]]: Filtered collection structures matching constraints.
        """
        try:
            import requests as req

            resp = req.post(
                f"{OLLAMA_API_URL}/api/embed",
                json={"model": EMBED_MODEL, "input": query_text},
                timeout=30,
            )
            embedding = resp.json()["embeddings"][0]

            results = self.collection.query(
                query_embeddings=[embedding],
                n_results=min(k, self._collection_count),
                where=where_filter,
                include=["documents", "metadatas", "distances"],
            )

            documents = results.get("documents", [[]])[0]
            metadatas = results.get("metadatas", [[]])[0]
            distances = results.get("distances", [[]])[0]

            accepted = [
                {"document": doc, "metadata": meta or {}}
                for doc, meta, dist in zip(documents, metadatas, distances)
                if dist <= distance_threshold
            ]

            logging.debug(
                f"RAG threshold={distance_threshold} | "
                f"passed={len(accepted)}/{len(distances)} | "
                f"distances={[round(d, 4) for d in distances]}"
            )

            return accepted

        except Exception as exc:
            logging.warning(
                f"ChromaDB filtered query failed ({where_filter}): {exc}"
            )
            return []

    def _log_unfiltered_probe(
        self,
        query_text: str,
        lookup_domain: str,
        k: int,
    ) -> None:
        """Run an open probe across all records for structural troubleshooting.

        Executed only when filtered query workflows return zero valid hits.

        Args:
            query_text (str): Evaluated lookahead string context.
            lookup_domain (str): Intended target domain filter block.
            k (int): Result collection slice count.
        """
        if self.collection is None or self._collection_count <= 0:
            return

        try:
            import requests as req

            resp = req.post(
                f"{OLLAMA_API_URL}/api/embed",
                json={"model": EMBED_MODEL, "input": query_text},
                timeout=30,
            )
            embedding = resp.json()["embeddings"][0]

            probe = self.collection.query(
                query_embeddings=[embedding],
                n_results=min(k, self._collection_count),
                include=["documents", "metadatas"],
            )

            documents = probe.get("documents", [[]])[0]
            metadatas = probe.get("metadatas", [[]])[0]

            preview = [
                {
                    "lookup_domain": (
                        (metadata or {}).get("lookup_domain")
                        or (metadata or {}).get("metadata.lookup_domain")
                    ),
                    "metadata": metadata or {},
                    "document": document,
                }
                for document, metadata in zip(documents, metadatas)
            ]

            logging.info(
                "RAG unfiltered probe for "
                f"value `{query_text}` after empty lookup_domain "
                f"`{lookup_domain}` lookup: {preview}"
            )

        except Exception as exc:
            logging.warning(
                "RAG unfiltered probe failed "
                f"(lookup_domain={lookup_domain}, k={k}): {exc}"
            )

    def build_rag_context(self, candidates: Dict[str, Any]) -> str:
        """Construct the prompt segment with verified controlled definitions.

        Args:
            candidates (Dict[str, Any]): Intermediate metadata dict from Pass 1.

        Returns:
            str: Formatted text block ready for system context appending.
        """
        if self.collection is None:
            return ""

        blocks = []

        for field, config in RAG_FIELD_CONFIG.items():
            value = candidates.get(field)

            if not value:
                logging.info(f"RAG skipped field `{field}`: no candidate.")
                continue

            hits = self.query_field(
                str(value),
                config["lookup_domain"],
                config["k"],
            )

            if not hits:
                logging.info(
                    f"RAG returned no hits for field `{field}` "
                    f"value `{value}` in lookup_domain "
                    f"`{config['lookup_domain']}`."
                )
                continue

            logging.info(
                f"RAG hits for field `{field}` value `{value}`: {len(hits)}"
            )

            lines = [f"- Field `{field}`:"]

            for hit in hits:
                lines.append(f"  chunk: {hit['document']}")

                if hit["metadata"]:
                    meta_str = ", ".join(
                        f"{key}={value}"
                        for key, value in hit["metadata"].items()
                        if value
                    )
                    lines.append(f"  metadata: {meta_str}")

            blocks.append("\n".join(lines))

        if not blocks:
            return ""

        return (
            "### RAG CONTROL DATA "
            "(Use these preferred terms if they match the text):\n"
            + "\n".join(blocks)
        )