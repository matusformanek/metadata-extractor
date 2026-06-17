# -*- coding: utf-8 -*-
"""RAG helper module for controlled vocabulary lookup in ChromaDB.

This module provides the database abstraction layer through the Librarian class,
which handles local vector storage probing, explicit metadata domain filtering,
and query semantic augmentation using the Ollama embedding API.
"""

import logging
from typing import Any, Dict, List

from config.settings import (
    CHROMA_COLLECTION,
    CHROMA_PATH,
    DISTANCE_THRESHOLD,
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
        self.emb_fn = None
        self.collection = None
        self._collection_count = 0
        self._domain_counts = {}

        try:
            import chromadb
            from chromadb.utils import embedding_functions

            client = chromadb.PersistentClient(path=CHROMA_PATH)

            # CRITICAL: Store the embedding function instance to enable native
            # batching and eliminate redundant low-level HTTP client setups.
            self.emb_fn = embedding_functions.OllamaEmbeddingFunction(
                model_name=EMBED_MODEL,
                url=OLLAMA_API_URL,
            )

            self.collection = client.get_collection(
                name=CHROMA_COLLECTION,
                embedding_function=self.emb_fn,
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
        """Inspect collection metadata to calculate item counts per domain safely.

        Returns:
            Dict[str, int]: Map of lookup_domain names to their record counts.
        """
        if self.collection is None:
            return {}

        counts: Dict[str, int] = {}
        try:
            # CRITICAL: Extract unique expected domains from configuration to
            # perform specific queries instead of scanning the full database.
            unique_domains = {
                config["lookup_domain"] for config in RAG_FIELD_CONFIG.values()
            }

            for domain in unique_domains:
                # CRITICAL (OOM Prevention): Request ONLY record IDs via
                # include=["ids"]. This prevents loading huge metadata dicts
                # or document contents into RAM, avoiding Out of Memory
                # crashes on large datasets.
                res_direct = self.collection.get(
                    where={"lookup_domain": {"$eq": domain}},
                    include=[],
                )
                count_direct = len(res_direct.get("ids", []))

                res_meta = self.collection.get(
                    where={"metadata.lookup_domain": {"$eq": domain}},
                    include=[],
                )
                count_meta = len(res_meta.get("ids", []))

                total_domain_count = count_direct + count_meta
                if total_domain_count > 0:
                    counts[domain] = total_domain_count

            return counts

        except Exception as exc:
            logging.warning(
                f"Could not inspect ChromaDB lookup domains safely: {exc}"
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
            lookup_domain (str): Controlled target vocabulary identifier.

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

    def _get_embedding(self, text: str) -> List[float]:
        """Generate vector embedding via the native ChromaDB embedding function.

        Args:
            text (str): Augmented text string.

        Returns:
            List[float]: Numerical vector representation.
        """
        if self.emb_fn is None:
            raise RuntimeError("Embedding function is not initialized.")
        return self.emb_fn([text])[0]

    def query_field(
        self,
        query_text: str,
        lookup_domain: str,
        k: int,
        distance_threshold: float = DISTANCE_THRESHOLD,
        precomputed_embedding: List[float] = None,
    ) -> List[Dict[str, Any]]:
        """Query the vector database for a specific controlled vocabulary term.

        Args:
            query_text (str): Raw string candidate to evaluate.
            lookup_domain (str): Specific domain constraint name.
            k (int): Limit of nearest neighbor matches to inspect.
            distance_threshold (float): Cutoff boundary rejecting loose weights.
            precomputed_embedding (List[float]): Precalculated vector from batch.

        Returns:
            List[Dict[str, Any]]: Matched documents and metadata structures.
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
            # CRITICAL: Use the precomputed embedding from the batch operation
            # if available. This avoids redundant API calls to Ollama.
            if precomputed_embedding is not None:
                embedding = precomputed_embedding
            else:
                embedding = self._get_embedding(query_text_augmented)

            # CRITICAL: Combine alternative metadata structural schemas using
            # a native database level $or logical operator for speed.
            where_filter = {
                "$or": [
                    {"lookup_domain": {"$eq": lookup_domain}},
                    {"metadata.lookup_domain": {"$eq": lookup_domain}},
                ]
            }

            hits = self._query_with_filter(
                embedding,
                k,
                where_filter,
                distance_threshold,
            )

            if hits:
                logging.info(
                    f"RAG matched `{query_text}` "
                    f"(augmented: `{query_text_augmented}`) "
                    f"in `{lookup_domain}`"
                )
                return hits

            return []

        except Exception as exc:
            logging.warning(
                f"ChromaDB query_field failed ({lookup_domain}): {exc}"
            )
            return []

    def _query_with_filter(
        self,
        embedding: List[float],
        k: int,
        where_filter: Dict[str, Dict[str, Any]],
        distance_threshold: float,
    ) -> List[Dict[str, Any]]:
        """Perform a low level vector query bounded by a strict cosine cutoff."""
        try:
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
        embedding: List[float],
        query_text: str,
        lookup_domain: str,
        k: int,
    ) -> None:
        """Run an open probe across all records for structural troubleshooting."""
        if self.collection is None or self._collection_count <= 0:
            return

        try:
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
            candidates (Dict[str, Any]): Metadata dict from the first phase.

        Returns:
            str: Formatted text block ready for system context appending.
        """
        if self.collection is None or self.emb_fn is None:
            return ""

        active_fields = []
        texts_to_embed = []

        # PHASE 1: Collect present candidates and prepare text strings
        for field, config in RAG_FIELD_CONFIG.items():
            value = candidates.get(field)

            if not value:
                logging.info(f"RAG skipped field `{field}`: no candidate.")
                continue

            augmented_text = self._augment_query(
                str(value), config["lookup_domain"]
            )
            active_fields.append((field, config, str(value)))
            texts_to_embed.append(augmented_text)

        if not texts_to_embed:
            return ""

        try:
            # CRITICAL: Execute batch embedding generation in a single network
            # request. This drastically minimizes overall processing latency
            # by removing serial HTTP roundtrips to the local Ollama server.
            embeddings = self.emb_fn(texts_to_embed)
        except Exception as exc:
            logging.warning(f"Batch embedding generation failed: {exc}")
            return ""

        blocks = []

        # PHASE 3: Query database using the precomputed vector arrays
        for (field, config, value), embedding in zip(active_fields, embeddings):
            hits = self.query_field(
                query_text=value,
                lookup_domain=config["lookup_domain"],
                k=config["k"],
                distance_threshold=DISTANCE_THRESHOLD,
                precomputed_embedding=embedding,
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
                        f"{key}= {val}"
                        for key, val in hit["metadata"].items()
                        if val
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