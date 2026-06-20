# -*- coding: utf-8 -*-
"""Post-processing helpers for extracted bibliographic metadata.

This module provides comprehensive normalization pipelines to clean, format,
and sanitize metadata extracted by Large Language Models. It includes routines
for validating publication dates, striping academic titles, consolidating
language codes to ISO 639-1, and performing deterministic regex parsing on DOIs.  # noqa: E501
"""

import copy
import re
import unicodedata
from typing import Any, Dict, Optional

# Regular expression pattern to identify and strip common academic titles
ACADEMIC_TITLES_REGEX = re.compile(
    r"""
    \b(?:
        prof|doc|as|odbas|
        bc|bca|mgr(?:\.?\s*art)?|mga|ing(?:\.?\s*arch)?|
        mudr|mddr|mvdr|rndr|pharmdr|phdr|judr|paeddr|thdr|thlic|
        ph\.?d|csc|drsc|artd|th\.?d|d\.?sc|
        ba|b\.?sc|beng|ma|m\.?sc|meng|mba|mpa|llm|mphil|
        dba|edd|md|do|dvm|dds|
        dr\.?\s*h\.?\s*c|dr|dis|rsdr|msdr
    )
    \.?
    (?=\s|[,;]|$|\))
    """,
    re.IGNORECASE | re.VERBOSE,
)


def normalize_issued(issued: Any) -> Optional[str]:
    """Normalize an LLM-provided publication date value to ISO format.

    Processes variations including nested date-part dictionaries, lists,
    European formatting, and standalone calendar years.

    Args:
        issued (Any): Raw input date representation from the model.

    Returns:
        Optional[str]: Normalized string (YYYY-MM-DD, YYYY-MM, or YYYY)
            or None if validation fails.
    """
    if not issued:
        return None

    if isinstance(issued, dict):
        parts = issued.get("date-parts", [])
        if parts and isinstance(parts[0], list):
            issued = parts[0]
        elif "year" in issued:
            issued = [
                issued.get("year"),
                issued.get("month"),
                issued.get("day"),
            ]

    if isinstance(issued, (list, tuple)):
        try:
            clean_parts = [
                int(part)
                for part in issued
                if part is not None and str(part).isdigit()
            ]
            if not clean_parts:
                return None

            year = clean_parts[0]
            if not 1000 < year < 2100:
                return None
            if len(clean_parts) >= 3:
                return f"{year:04d}-{clean_parts[1]:02d}-{clean_parts[2]:02d}"
            if len(clean_parts) >= 2:
                return f"{year:04d}-{clean_parts[1]:02d}"
            return str(year)
        except (TypeError, ValueError):
            return None

    issued_str = str(issued).strip()
    iso_match = re.search(r"(\b\d{4}-\d{2}-\d{2}\b)", issued_str)
    if iso_match:
        return iso_match.group(1)

    eu_match = re.search(
        r"(\d{1,2})[\.\ /](\d{1,2})[\.\ /](\d{4})", issued_str
    )
    if eu_match:
        day, month, year = eu_match.groups()
        return f"{int(year):04d}-{int(month):02d}-{int(day):02d}"

    year_match = re.search(r"\b(19|20)\d{2}\b", issued_str)
    if year_match:
        return year_match.group(0)

    return None


def normalize_author_name(name: str) -> Optional[str]:
    """Clean one author or contributor name candidate.

    Strips identifiers, separates merged names, normalizes legacy umlaut encodings,  # noqa: E501
    removes emails, handles brackets/symbols, and purges academic titles.

    Args:
        name (str): Raw string containing a person's name candidate.

    Returns:
        Optional[str]: Cleaned name string or None if it contains no letters.
    """
    if not name:
        return None

    # Strip identifiers from mixed strings while preserving the name
    name = _strip_author_identifiers(name)
    if not re.search(r"[a-zA-ZÀ-ž]", name):
        return None

    # Fix merged names where lowercase is immediately followed by uppercase
    name = re.sub(r"(?<=[a-zÀ-ž])(?=[A-Z])", " ", name)
    name = re.sub(r"(?<=[a-zÀ-ž])(?=[A-Z]\.)", " ", name)

    # Reconstruct legacy character variations
    name = name.replace("¨a", "ä").replace("¨o", "ö").replace("¨u", "ü")

    # Purge unexpected formatting remnants
    name = re.sub(r"\S+@\S+", "", name)
    name = re.sub(r"\[.*?\]", "", name)
    name = re.sub(r"(?<!\w)\d+(?!\w)", "", name)
    name = re.sub(r"\([,\s]*\)", "", name)
    name = re.sub(r"[\u2022\*\u2020\u2021\u00a7]", "", name)
    name = ACADEMIC_TITLES_REGEX.sub("", name)
    name = re.sub(r"^[,\s]+|[,\s]+$", "", name)
    name = re.sub(r"\s+", " ", name)

    return name.strip() or None


def normalize_language(lang: Any) -> Optional[str]:
    """Normalize language names and ISO 639-2 codes to ISO 639-1.

    Args:
        lang (Any): The language string identifier to normalize.

    Returns:
        Optional[str]: Normalized two-letter code or the sanitized original input.  # noqa: E501
    """
    if not lang or not isinstance(lang, str):
        return lang if lang else None

    lang_clean = _strip_accents(lang.lower().strip())
    lang_map = {
        "eng": "en",
        "english": "en",
        "en-us": "en",
        "en-gb": "en",
        "slk": "sk",
        "slo": "sk",
        "slovak": "sk",
        "slovencina": "sk",
        "slovensky": "sk",
        "cze": "cs",
        "ces": "cs",
        "czech": "cs",
        "cestina": "cs",
        "cesky": "cs",
        "ger": "de",
        "deu": "de",
        "german": "de",
        "deutsch": "de",
        "nemecky": "de",
        "fre": "fr",
        "fra": "fr",
        "french": "fr",
        "francais": "fr",
        "francuzsky": "fr",
        "pol": "pl",
        "polish": "pl",
        "polski": "pl",
        "polsky": "pl",
        "hun": "hu",
        "hungarian": "hu",
        "magyar": "hu",
        "madarsky": "hu",
        "spa": "es",
        "esp": "es",
        "spanish": "es",
        "espanol": "es",
        "spanielsky": "es",
        "por": "pt",
        "portuguese": "pt",
        "portugues": "pt",
        "portugalsky": "pt",
        "lat": "la",
        "latin": "la",
        "latinsky": "la",
        "ita": "it",
        "italian": "it",
        "italiano": "it",
        "taliansky": "it",
        "rus": "ru",
        "russian": "ru",
        "rusky": "ru",
        "ukr": "uk",
        "ukrainian": "uk",
        "ukrajinsky": "uk",
        "nld": "nl",
        "dut": "nl",
        "dutch": "nl",
        "nederlands": "nl",
        "holandsky": "nl",
        "swe": "sv",
        "swedish": "sv",
        "svenska": "sv",
        "svedsky": "sv",
        "nor": "no",
        "norwegian": "no",
        "norsk": "no",
        "norsky": "no",
        "dan": "da",
        "danish": "da",
        "dansk": "da",
        "dansky": "da",
        "fin": "fi",
        "finnish": "fi",
        "suomi": "fi",
        "finsky": "fi",
        "ell": "el",
        "gre": "el",
        "greek": "el",
        "grecky": "el",
        "tur": "tr",
        "turkish": "tr",
        "turecky": "tr",
        "ron": "ro",
        "rum": "ro",
        "romanian": "ro",
        "rumunsky": "ro",
        "bul": "bg",
        "bulgarian": "bg",
        "bulharsky": "bg",
        "srp": "sr",
        "serbian": "sr",
        "srpski": "sr",
        "srbsky": "sr",
        "hrv": "hr",
        "croatian": "hr",
        "hrvatski": "hr",
        "chorvatsky": "hr",
        "zho": "zh",
        "chi": "zh",
        "chinese": "zh",
        "cinsky": "zh",
        "zh-cn": "zh",
        "zh-tw": "zh",
        "jpn": "ja",
        "japanese": "ja",
        "japonsky": "ja",
        "kor": "ko",
        "korean": "ko",
        "korejsky": "ko",
        "ara": "ar",
        "arabic": "ar",
        "arabsky": "ar",
        "hin": "hi",
        "hindi": "hi",
        "hindsky": "hi",
    }
    return lang_map.get(lang_clean, lang_clean)


def _strip_accents(value: str) -> str:
    """Return a lowercase lookup-safe variant without diacritics."""
    normalized = unicodedata.normalize("NFKD", value)
    return "".join(
        char for char in normalized if not unicodedata.combining(char)
    )


def _strip_author_identifiers(name: str) -> str:
    """Remove ORCID and URL fragments from an author name candidate."""
    name = re.sub(
        r"https?://orcid\.org/[\d\-X]+",
        "",
        name,
        flags=re.IGNORECASE,
    )
    name = re.sub(
        r"orcid\.org/[\d\-X]+",
        "",
        name,
        flags=re.IGNORECASE,
    )
    name = re.sub(r"\b\d{4}-\d{4}-\d{4}-\d{3}[\dX]\b", "", name)
    name = re.sub(r"https?://\S+", "", name, flags=re.IGNORECASE)
    name = re.sub(r"www\.\S+", "", name, flags=re.IGNORECASE)
    return name


def clean_output(data: Dict[str, Any]) -> Dict[str, Any]:
    """Normalize the extracted metadata dictionary without mutating input.

    Args:
        data (Dict[str, Any]): Original unverified metadata dictionary structure.  # noqa: E501

    Returns:
        Dict[str, Any]): Deep-copied and fully processed flat metadata layout.
    """
    data = copy.deepcopy(data)

    string_fields = [
        "title",
        "alternative_title",
        "issued",
        "publisher",
        "publication_place",
        "resource_type",
        "language",
        "abstract",
        "rights_uri",
        "persistent_uri",
    ]

    for field in string_fields:
        value = data.get(field)
        if isinstance(value, list):
            value = ", ".join(str(item).strip() for item in value if item)

        if isinstance(value, str):
            value = re.sub(r"\[([^\]]+)\]\([^)]+\)", r"\1", value)
            data[field] = value.strip()

    for field in ["authors", "contributors"]:
        value = data.get(field)
        if not value:
            continue

        if isinstance(value, str):
            value = [value]

        if isinstance(value, list):
            data[field] = _clean_person_list(value)

    if data.get("title"):
        data["title"] = re.sub(r"\s+", " ", str(data["title"])).strip(" .")

    if data.get("issued"):
        data["issued"] = normalize_issued(data["issued"])

    if data.get("subjects"):
        subjects = data["subjects"]
        if isinstance(subjects, str):
            subjects = re.split(r"[;,\n]", subjects)
        if isinstance(subjects, list):
            data["subjects"] = [
                sub.strip().capitalize()
                for subject in subjects
                for sub in [subject.strip()]
                if sub
            ]

    if data.get("publisher"):
        data["publisher"] = re.sub(
            r"[\.,/]$",
            "",
            str(data["publisher"]),
        ).strip()

    if data.get("persistent_uri"):
        data["persistent_uri"] = re.sub(
            r"[\.,/]$",
            "",
            str(data["persistent_uri"]),
        ).strip()

    _clean_doi_field(data)

    # Process issn_print and issn_electronic identifiers uniformly
    for field in ["issn_print", "issn_electronic"]:
        if data.get(field):
            data[field] = str(data[field]).upper().replace("issn:", "").strip()
            data[field] = re.sub(r"[\.,/]$", "", data[field])

    if data.get("isbn"):
        data["isbn"] = (
            re.sub(r"[- ]", "", str(data["isbn"]))
            .upper()
            .replace("isbn:", "")
            .strip()
        )
        data["isbn"] = re.sub(r"[\.,/]$", "", data["isbn"])

    if data.get("language"):
        data["language"] = normalize_language(data["language"])

    return data


def _clean_person_list(values: list) -> list:
    """Normalize an author/contributor list and drop non-name fragments."""
    cleaned_list = []

    for item in values:
        if isinstance(item, str):
            parts = re.split(r";|\s+and\s+|\s+&\s+", item)
            for part in parts:
                clean_part = normalize_author_name(part)
                if clean_part and len(clean_part) > 1:
                    cleaned_list.append(clean_part)
        elif isinstance(item, dict):
            name = item.get("name") or item.get("value") or item.get("author")
            clean_name = normalize_author_name(str(name)) if name else None
            if clean_name and len(clean_name) > 1:
                cleaned_list.append(clean_name)

    return cleaned_list


def _clean_doi_field(data: Dict[str, Any]) -> None:
    """Normalize DOI values, extract valid DOI patterns, and reject bad URIs."""  # noqa: E501
    if not data.get("doi"):
        return

    original_doi = str(data["doi"]).strip()

    # 1. Remove known administrative prefixes (case-insensitive)
    prefix_pattern = r"^(https?://(dx\.)?doi\.org/|doi:)"
    cleaned = re.sub(
        prefix_pattern, "", original_doi, flags=re.IGNORECASE
    ).strip()

    # 2. Strip internal whitespaces if the LLM fragmented the string
    cleaned = re.sub(r"\s+", "", cleaned)

    # 3. EXTRACTION: Isolate structural DOI patterns from string roots
    doi_match = re.match(r"^(10\.\d{4,9}/[-._;()/:A-Za-z0-9]+)", cleaned)

    if doi_match:
        cleaned_doi = doi_match.group(1)

        # Drop trailing punctuation marks from resolved string boundaries
        cleaned_doi = re.sub(r"[\.,/]$", "", cleaned_doi).lower()

        data["doi"] = cleaned_doi
        return

    # 4. Handle conversion to persistent URIs if pattern is an alternative
    # handle
    cleaned_lower = cleaned.lower()
    if not data.get("persistent_uri") and (
        "handle" in cleaned_lower or "hdl." in cleaned_lower
    ):
        data["persistent_uri"] = original_doi

    # Nullify field if extraction rules and handle routing fail completely
    data["doi"] = None
