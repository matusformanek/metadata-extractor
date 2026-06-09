# -*- coding: utf-8 -*-
"""Pydantic validation schemas for metadata extraction outputs.

This module defines the structural data models utilized to validate and
sanitize Dublin Core metadata fields extracted from academic documents by local
Large Language Models. It enforces strict type constraints for repository
ingestion and preserves an evidence tracking log for automated evaluation.
"""

from typing import Any, List, Optional, Union

from pydantic import BaseModel, ConfigDict, Field, field_validator


# ==============================================================================
# CORE METADATA VALIDATION SCHEMA
# ==============================================================================

class TargetMetadata(BaseModel):
    """Validated bibliographic metadata prepared for digital repository import.

    This model serves as the final sanitized data layer mapping directly
    to localized Dublin Core schema components. It forbids arbitrary extra fields
    to ensure absolute structure conformity during deserialization.
    """

    model_config = ConfigDict(extra="forbid")

    title: Optional[str] = None
    alternative_title: Optional[str] = None
    authors: List[str] = Field(default_factory=list)
    contributors: List[str] = Field(default_factory=list)
    issued: Optional[str] = None
    publisher: Optional[str] = None
    publication_place: Optional[str] = None
    resource_type: Optional[str] = None
    language: Optional[str] = None
    doi: Optional[str] = None
    isbn: Optional[str] = None
    issn_print: Optional[str] = None
    issn_electronic: Optional[str] = None
    persistent_uri: Optional[str] = None
    abstract: Optional[str] = None
    subjects: List[str] = Field(default_factory=list)
    rights_uri: Optional[str] = None

    @field_validator("authors", "contributors", "subjects", mode="before")
    @classmethod
    def ensure_list(cls, value: Any) -> Any:
        """Convert null or missing list fields from the model into empty lists.

        Args:
            value (Any): The raw unverified input assigned to the list field.

        Returns:
            Any: An initialized list instance if input was null, else original.
        """
        return value if value is not None else []

    @field_validator("issued", mode="before")
    @classmethod
    def coerce_issued_to_string(cls, value: Any) -> Any:
        """Coerce integer publication years into strict string formats.

        This pre-validator prevents serialization failures downstream when the
        extraction model outputs numerical years instead of ISO strings.

        Args:
            value (Any): Raw value provided for the date field.

        Returns:
            Any: String representation of the integer value, else original.
        """
        if isinstance(value, int):
            return str(value)
        return value


# ==============================================================================
# EVALUATION AND EVIDENCE AUDIT TRAIL SCHEMA
# ==============================================================================

class EvidenceLog(BaseModel):
    """Mirror metadata fields containing source text evidence segments.

    This structure holds precise verification fragments or verbatim quotes
    isolated by the extraction agent from the primary document stream. It is
    utilized to evaluate model hallucinations and cross-reference assertions.
    """

    model_config = ConfigDict(extra="ignore")

    title: Union[str, List[str], None] = None
    alternative_title: Union[str, List[str], None] = None
    authors: Union[str, List[str], None] = None
    contributors: Union[str, List[str], None] = None
    issued: Union[str, List[str], None] = None
    publisher: Union[str, List[str], None] = None
    publication_place: Union[str, List[str], None] = None
    resource_type: Union[str, List[str], None] = None
    language: Union[str, List[str], None] = None
    doi: Union[str, List[str], None] = None
    isbn: Union[str, List[str], None] = None
    issn_print: Union[str, List[str], None] = None
    issn_electronic: Union[str, List[str], None] = None
    persistent_uri: Union[str, List[str], None] = None
    abstract: Union[str, List[str], None] = None
    subjects: Union[str, List[str], None] = None
    rights_uri: Union[str, List[str], None] = None


# ==============================================================================
# TOP LEVEL PIPELINE INTERFACE SCHEMA
# ==============================================================================

class AgentOutput(BaseModel):
    """Top-level unified schema expected from the extraction pipeline pass.

    Encapsulates both the normalized bibliographic properties and the supporting
    text segments into a single cohesive response layout from the LLM agent.
    """

    metadata: TargetMetadata
    evidence_log: EvidenceLog