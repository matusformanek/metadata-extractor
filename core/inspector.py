"""Validation and repair layer for LLM extraction output."""

import json
import logging
import re
from typing import Optional

from pydantic import ValidationError

from postprocessing.cleaning import clean_output
from schemas.metadata import AgentOutput, TargetMetadata

logger = logging.getLogger(__name__)


class Inspector:
    """Repair, validate, and audit LLM metadata output."""

    EXEMPT_FIELDS = {"language", "resource_type", "abstract"}

    @staticmethod
    def count_missing_braces(text: str) -> int:
        """Count unmatched opening braces outside JSON strings."""
        depth = 0
        in_string = False
        escape = False

        for char in text:
            if escape:
                escape = False
                continue
            if char == "\\" and in_string:
                escape = True
                continue
            if char == '"':
                in_string = not in_string
                continue
            if not in_string:
                if char == "{":
                    depth += 1
                elif char == "}":
                    depth -= 1

        return max(depth, 0)

    @classmethod
    def validate(
        cls,
        raw_json: str,
        attempt: int,
    ) -> tuple[bool, Optional[TargetMetadata], str]:
        """Validate one raw LLM response and return cleaned metadata."""
        raw_json = raw_json.strip()
        valid_line_endings = (
            '"',
            '",',
            "}",
            "},",
            "]",
            "],",
            ":",
            "true",
            "false",
            "null",
        )

        if raw_json.startswith("```json"):
            raw_json = re.sub(r"^```json\s*", "", raw_json)
            raw_json = re.sub(r"\s*```$", "", raw_json)

        # Pre-parse resolution for multiple duplicate root keys caused by model
        # context loops
        if raw_json.count('"evidence_log"') > 1:
            matches = [
                m.start() for m in re.finditer('"evidence_log"', raw_json)
            ]
            second_log_start = matches[1]
            truncated_json = raw_json[:second_log_start].strip()
            truncated_json = re.sub(r",\s*$", "", truncated_json)
            if not truncated_json.endswith("}"):
                truncated_json += "}"
            raw_json = truncated_json

        # Safe truncation repair for fragmented context ends without breaking
        # long single-line abstracts
        if "evidence_log" in raw_json and not raw_json.endswith("}"):
            if len(raw_json) > 4000:
                lines = raw_json.split("\n")
                if lines:
                    last_line = lines[-1].strip()
                    if len(last_line) > 500 and not last_line.endswith(
                        valid_line_endings
                    ):
                        lines.pop()
                        raw_json = "\n".join(lines).strip()

        if raw_json and not raw_json.endswith("}"):
            if (
                not raw_json.endswith('"')
                and not raw_json.endswith("]")
                and not raw_json.endswith("null")
            ):
                if re.search(r':\s*"[^"]*$', raw_json):
                    raw_json += '"'

            missing_braces = cls.count_missing_braces(raw_json)
            if missing_braces > 0:
                raw_json += "}" * missing_braces

        try:
            data = json.loads(raw_json)

            if "metadata" in data and "evidence_log" in data["metadata"]:
                if "evidence_log" not in data or not data["evidence_log"]:
                    data["evidence_log"] = data["metadata"].pop("evidence_log")

            if "title" in data and "metadata" not in data:
                extracted_meta = {
                    key: value
                    for key, value in data.items()
                    if key != "evidence_log"
                }
                data = {
                    "metadata": extracted_meta,
                    "evidence_log": data.get("evidence_log", {}),
                }

            if "evidence_log" in data and isinstance(
                data["evidence_log"], dict
            ):
                data["evidence_log"] = cls._clean_evidence_log(
                    data["evidence_log"]
                )
                cls._truncate_abstract_evidence(data["evidence_log"])

            if "metadata" in data and isinstance(data["metadata"], dict):
                data["metadata"] = clean_output(data["metadata"])

            # Spustenie typovej validácie štruktúry Pydantic
            parsed = AgentOutput.model_validate(data)

            # Spustenie biznis logiky a kontroly hodnôt (obsahuje podmienený
            # titulok)
            cls._audit_required_content(parsed, attempt)

            return True, parsed.metadata, ""

        except (json.JSONDecodeError, ValidationError) as exc:
            # Kritická chyba: Neplatný JSON alebo nesúlad dátových typov so
            # schémou
            logger.warning(
                "JSON schema validation failed on attempt %d", attempt
            )
            return False, None, f"JSON schema validation failed: {str(exc)}"

        except ValueError as exc:
            # Logická chyba: Vyvolaná metódou _audit_required_content (napr.
            # chýbajúci title pri pokuse 1)
            logger.warning(
                "Logical validation failed on attempt %d: %s",
                attempt,
                str(exc),
            )
            return False, None, f"Logical validation failed: {str(exc)}"

        except Exception as exc:
            # Neočakávaný pád aplikácie pri spracovaní
            logger.error("Parser crashed on attempt %d", attempt)
            return False, None, f"Parser crash: {str(exc)}"

    @staticmethod
    def _clean_evidence_log(evidence_log: dict) -> dict:
        """Normalize common evidence key/value variants produced by the LLM."""
        clean_log = {}
        for key, value in evidence_log.items():
            new_key = (
                key.replace("_evidence", "")
                .replace("_quote", "")
                .replace("_start", "")
                .replace("_end", "")
            )

            # Unnesting entire embedded schemas injected by the model into the
            # evidence log
            if isinstance(value, dict):
                if "quote" in value:
                    value = str(value["quote"])
                elif new_key in value:
                    value = str(value[new_key])
                else:
                    value = ", ".join(str(v) for v in value.values() if v)
            elif isinstance(value, list):
                value = ", ".join(
                    str(
                        item.get("quote", item)
                        if isinstance(item, dict)
                        else item
                    )
                    for item in value
                )
            else:
                value = str(value) if value is not None else ""

            clean_log[new_key] = value

        return clean_log

    @staticmethod
    def _truncate_abstract_evidence(
        evidence_log: dict,
        limit: int = 200,
    ) -> None:
        """Keep abstract evidence compact without changing metadata.abstract."""  # noqa: E501
        abstract = evidence_log.get("abstract")
        if isinstance(abstract, str) and len(abstract) > limit:
            evidence_log["abstract"] = abstract[:limit].rstrip()

    @classmethod
    def _audit_required_content(
        cls,
        parsed: AgentOutput,
        attempt: int,
    ) -> None:
        """Enforce mandatory fields and evidence coverage."""
        meta_dict = parsed.metadata.model_dump()
        ev_dict = parsed.evidence_log.model_dump()
        missing_evidence = []

        for field, value in meta_dict.items():
            if field in cls.EXEMPT_FIELDS:
                continue
            if value and not ev_dict.get(field):
                missing_evidence.append(field)

        if missing_evidence:
            error_message = (
                "CRITICAL ERROR: You extracted values for metadata fields "
                f"{missing_evidence}, but left their evidence_log fields null. "  # noqa: E501
                "EVIDENCE RULE: You MUST provide a verbatim quote for EVERY "
                "extracted field."
            )
            if attempt == 1:
                raise ValueError(error_message)

            logger.info(
                "Missing evidence accepted on attempt 2 for: %s",
                missing_evidence,
            )

        # Podmienená kontrola prítomnosti poľa 'title' na základe čísla pokusu
        if not parsed.metadata.title:
            if attempt == 1:
                raise ValueError(
                    "Field 'title' is missing. It is mandatory for Attempt 1. Find the main document title."  # noqa: E501
                )
            else:
                logger.warning(
                    "Attempt 2: Field 'title' is missing, but accepting partial metadata output."  # noqa: E501
                )

        if attempt == 1:
            if not parsed.metadata.authors:
                raise ValueError(
                    "Field 'authors' is empty. Look closely at the title page "
                    "or header under the title."
                )
            if not parsed.metadata.language:
                raise ValueError(
                    "Field 'language' is missing. Analyze the text vocabulary "
                    "and provide an ISO 639-1 code."
                )
