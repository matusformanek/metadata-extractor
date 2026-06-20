# -*- coding: utf-8 -*-
"""Metadata Extraction Evaluation Tool.

This module evaluates the performance of metadata extraction processes
by computing Precision, Recall, and F1-score metrics. It combines
Exact Match and Fuzzy Match (via token_set_ratio) algorithms while
incorporating semantic normalization for custom fields.

Fatal failures, such as corrupted JSON objects or raw text fallbacks,
are strictly penalized as False Negatives to ensure academic and practical
rigor in LLM benchmarking.
"""

from datetime import datetime
import json
import os
from pathlib import Path
import re
from typing import Any, Tuple

try:
    from thefuzz import fuzz
except ImportError:
    print(
        "❌ The 'thefuzz' library is not installed. Run: pip install thefuzz[speedup]"  # noqa: E501
    )
    exit(1)


# ==============================================================================  # noqa: E501
# CONFIGURATION AND GLOBAL CONSTANTS
# ==============================================================================  # noqa: E501

# Paths to the directories containing ground truth data and system outputs
DIR_GROUND = Path("data/ground")
DIR_OUTPUT = Path("data/output")

# Mapping schema between ground truth keys and model output keys
# Format: "ground_truth_field": "model_output_field"
SCHEMA_MAPPING = {
    "doi": "doi",
    "title": "title",
    "authors": "authors",
    "issued": "issued",
    "language": "language",
    "type": "resource_type",
}

# Fields designated for token-based fuzzy string matching
FUZZY_FIELDS = {"title", "authors", "publisher"}

# Minimum similarity score required to consider a fuzzy match successful
# (0-100)
FUZZY_THRESHOLD = 90

# Normalization dictionary for target languages mapping to ISO 639-1 codes
LANGUAGE_MAPPING = {
    # English
    "en": "en",
    "english": "en",
    "angličtina": "en",
    "anglicky": "en",
    # Spanish
    "es": "es",
    "spanish": "es",
    "español": "es",
    "espanol": "es",
    "španielčina": "es",
    "španělština": "es",
    # Portuguese
    "pt": "pt",
    "portuguese": "pt",
    "português": "pt",
    "portugues": "pt",
    "portugalčina": "pt",
    "portugalština": "pt",
    # Russian
    "ru": "ru",
    "russian": "ru",
    "русский": "ru",
    "ruština": "ru",
    # Indonesian
    "id": "id",
    "indonesian": "id",
    "bahasa indonesia": "id",
    "indonézština": "id",
    "indonéština": "id",
    # Ukrainian
    "uk": "uk",
    "ukrainian": "uk",
    "українська": "uk",
    "ukrajinčina": "uk",
    "ukrajinština": "uk",
    # Polish
    "pl": "pl",
    "polish": "pl",
    "polski": "pl",
    "poľština": "pl",
    "polština": "pl",
    # Czech
    "cs": "cs",
    "cz": "cs",
    "czech": "cs",
    "čeština": "cs",
    "česky": "cs",
    # Turkish
    "tr": "tr",
    "turkish": "tr",
    "türkçe": "tr",
    "turečtina": "tr",
    # French
    "fr": "fr",
    "french": "fr",
    "français": "fr",
    "francais": "fr",
    "francúzština": "fr",
    "francouzština": "fr",
    # German
    "de": "de",
    "german": "de",
    "deutsch": "de",
    "nemčina": "de",
    "němčina": "de",
}


# ==============================================================================  # noqa: E501
# UTILITY AND NORMALIZATION FUNCTIONS
# ==============================================================================  # noqa: E501


def normalize_value(val: Any, field_name: str = "") -> Any:
    """Standardize and clean metadata values prior to evaluation.

    This function applies custom normalization rules based on the field type
    to eliminate formatting discrepancies caused by different LLM prompts
    or extraction contexts.

    Args:
        val (Any): The raw value extracted from the JSON object.
        field_name (str): The name of the metadata field being processed.

    Returns:
        Any: Standardized string, sorted list of strings, or None if invalid.
    """
    if val is None:
        return None

    if isinstance(val, list):
        cleaned_list = []
        for v in val:
            norm_v = normalize_value(v, field_name)
            if norm_v:
                cleaned_list.append(norm_v)
        return sorted(cleaned_list) if cleaned_list else None

    cleaned = str(val).strip().lower()
    cleaned = re.sub(r"[‐‑‒–—―]", "-", cleaned)

    if field_name == "authors":
        cleaned = re.sub(r"[,.]", "", cleaned)

    if cleaned in [
        "n/a",
        "none",
        "null",
        "na",
        "",
        "-",
        "unknown",
        "neuvedené",
    ]:
        return None

    if field_name == "issued":
        year_match = re.search(r"\b(18|19|20|21)\d{2}\b", cleaned)
        if year_match:
            cleaned = year_match.group(0)

    if field_name in ["type", "resource_type"]:
        cleaned = re.sub(r"[-_]", " ", cleaned)

        if "review article" in cleaned:
            cleaned = "review"
        elif "dissertation" in cleaned or "thesis" in cleaned:
            cleaned = "thesis"
        elif "article" in cleaned or "journal" in cleaned:
            cleaned = "article"

    if field_name == "language":
        if cleaned in LANGUAGE_MAPPING:
            cleaned = LANGUAGE_MAPPING[cleaned]

    return cleaned.strip() if cleaned.strip() else None


def calculate_f1(tp: int, fp: int, fn: int) -> Tuple[float, float, float]:
    """Calculate Precision, Recall, and the harmonic mean (F1-score)."""
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = (
        (2 * precision * recall) / (precision + recall)
        if (precision + recall) > 0
        else 0.0
    )
    return precision, recall, f1


def format_time(seconds: float) -> str:
    """Format a duration in seconds into a human-readable string."""
    if seconds < 60:
        return f"{seconds:.2f} s"
    mins = int(seconds // 60)
    secs = seconds % 60
    return f"{mins} min {secs:.2f} s"


# ==============================================================================  # noqa: E501
# MAIN EVALUATION PIPELINE
# ==============================================================================  # noqa: E501


def run_evaluation() -> None:
    """Execute the comprehensive metadata evaluation workflow.

    The pipeline loads ground truth and output files, analyzes execution times,
    evaluates matches, aggregates statistics, and records all configurations
    and final macro metrics at the very beginning of the report.
    """
    if not DIR_GROUND.exists() or not DIR_OUTPUT.exists():
        print("❌ CRITICAL ERROR: Data directories do not exist!")
        return

    ground_files = {f.stem for f in DIR_GROUND.glob("*.json")}

    if not ground_files:
        print("❌ No reference data found in data/ground.")
        return

    # --- TIMING METRICS ESTIMATION CODE ---
    json_outputs = list(DIR_OUTPUT.glob("*_metadata.json"))
    raw_outputs = list(DIR_OUTPUT.glob("*_raw.txt"))
    all_files = json_outputs + raw_outputs

    processed_stems = set()
    for f in all_files:
        stem_cleaned = f.stem.replace("_metadata", "").replace("_raw", "")
        processed_stems.add(stem_cleaned)

    n_documents = len(processed_stems)
    avg_time = 0.0
    total_batch_time = 0.0

    if n_documents > 0 and all_files:
        all_files_sorted = sorted(all_files, key=os.path.getmtime)
        t_start = all_files_sorted[0].stat().st_mtime
        t_end = all_files_sorted[-1].stat().st_mtime

        delta_t = t_end - t_start

        if n_documents > 1:
            avg_time = delta_t / (n_documents - 1)
            total_batch_time = delta_t + avg_time
        else:
            avg_time = delta_t
            total_batch_time = delta_t

    # Structuring report blocks
    meta_lines = []
    conflict_lines = []
    timestamp_str = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

    meta_lines.append("=== METADATA EXTRACTION PERFORMANCE REPORT ===")
    meta_lines.append(
        f"Evaluation Date: {datetime.now().strftime('%d.%m.%Y %H:%M:%S')}"
    )
    meta_lines.append(f"Fuzzy Match Threshold: {FUZZY_THRESHOLD}%\n")

    meta_lines.append("=== PERFORMANCE AND TEMPORAL METRICS ===")
    meta_lines.append(f"Number of unique processed documents: {n_documents}")
    if n_documents > 1:
        meta_lines.append(
            f"Average processing time per document: {format_time(avg_time)}"
        )
        meta_lines.append(
            f"Total batch processing time: {format_time(total_batch_time)}"
        )
        meta_lines.append(f"Throughput: {3600 / avg_time:.2f} doc./hour.")
    else:
        meta_lines.append("Insufficient data to compute valid timing metrics.")
    meta_lines.append("-" * 40 + "\n")

    conflict_lines.append("=== CONFLICTS AND MISMATCHES ===")

    stats = {key: {"TP": 0, "FP": 0, "FN": 0} for key in SCHEMA_MAPPING.keys()}
    failed_extractions_count = 0

    # Main iteration and verification loop
    for stem in sorted(ground_files):
        ground_path = DIR_GROUND / f"{stem}.json"
        output_path = DIR_OUTPUT / f"{stem}_metadata.json"
        raw_path = DIR_OUTPUT / f"{stem}_raw.txt"

        try:
            with open(ground_path, "r", encoding="utf-8") as f:
                ground_json = json.load(f)
        except json.JSONDecodeError:
            print(f"⚠️ Skipping {stem}: Ground Truth JSON is corrupted.")
            continue

        output_json = {}
        is_fatal_failure = False
        failure_reason = ""

        if output_path.exists():
            try:
                with open(output_path, "r", encoding="utf-8") as f:
                    output_json = json.load(f)
            except json.JSONDecodeError:
                is_fatal_failure = True
                failure_reason = "CORRUPTED JSON STRUCTURE"
        elif raw_path.exists():
            is_fatal_failure = True
            failure_reason = "RAW TEXT OUTPUT (Model extraction failure)"
        else:
            is_fatal_failure = True
            failure_reason = "MISSING OUTPUT FILE"

        file_has_error = False

        if is_fatal_failure:
            failed_extractions_count += 1
            conflict_lines.append(
                f"\n📄 File: {stem} 🚨 [FATAL FAILURE: {failure_reason}]"
            )
            file_has_error = True

        for ground_key, output_key in SCHEMA_MAPPING.items():
            val_g = normalize_value(ground_json.get(ground_key), ground_key)
            val_o = normalize_value(output_json.get(output_key), output_key)

            if not val_g and not val_o:
                stats[ground_key]["TP"] += 1
                continue

            is_match = False
            fuzzy_score = 0

            if val_g == val_o:
                is_match = True
            elif val_g and val_o and ground_key in FUZZY_FIELDS:
                s_g = " ".join(val_g) if isinstance(val_g, list) else val_g
                s_o = " ".join(val_o) if isinstance(val_o, list) else val_o
                fuzzy_score = fuzz.token_set_ratio(s_g, s_o)
                if fuzzy_score >= FUZZY_THRESHOLD:
                    is_match = True

            if is_match:
                stats[ground_key]["TP"] += 1
            else:
                if not file_has_error:
                    conflict_lines.append(f"\n📄 File: {stem}")
                    file_has_error = True

                if not is_fatal_failure:
                    conflict_lines.append(f"  [X] Field: {ground_key}")
                    conflict_lines.append(
                        f"      G: {ground_json.get(ground_key)} [Normalized: {val_g}]"  # noqa: E501
                    )
                    conflict_lines.append(
                        f"      O: {output_json.get(output_key)} [Normalized: {val_o}]"  # noqa: E501
                    )
                    if fuzzy_score > 0:
                        conflict_lines.append(
                            f"      Fuzzy Score: {fuzzy_score}%"
                        )

                if val_g and val_o:
                    stats[ground_key]["FP"] += 1
                    stats[ground_key]["FN"] += 1
                elif val_o:
                    stats[ground_key]["FP"] += 1
                else:
                    stats[ground_key]["FN"] += 1

    # --- AGGREGATION AND STATISTICAL REPORTING ---
    total_files = len(ground_files)
    successful_files = total_files - failed_extractions_count
    success_rate = (
        (successful_files / total_files * 100) if total_files > 0 else 0.0
    )
    failure_rate = (
        (failed_extractions_count / total_files * 100)
        if total_files > 0
        else 0.0
    )

    # Build final stats text block to place it at the top
    stats_lines = []
    stats_lines.append("=== FINAL STATISTICS ===")
    stats_lines.append(
        f"Successfully processed files: {successful_files} / {total_files} ({success_rate:.2f}%)"  # noqa: E501
    )
    stats_lines.append(
        f"Fatal failures (penalized as FN): {failed_extractions_count} / {total_files} ({failure_rate:.2f}%)\n"  # noqa: E501
    )

    header = (
        f"{'Field':<15} | {'TP':<4} | {'FP':<4} | {'FN':<4} | "
        f"{'P':<8} | {'R':<8} | {'F1':<8}"
    )
    stats_lines.append(header)
    separator = "-" * len(header)
    stats_lines.append(separator)

    f1_list = []
    for g_key in SCHEMA_MAPPING.keys():
        s = stats[g_key]
        p, r, f1 = calculate_f1(s["TP"], s["FP"], s["FN"])
        f1_list.append(f1)
        stat_line = (
            f"{g_key:<15} | {s['TP']:<4} | {s['FP']:<4} | {s['FN']:<4} | "
            f"{p:<8.4f} | {r:<8.4f} | {f1:<8.4f}"
        )
        stats_lines.append(stat_line)

    stats_lines.append(separator)
    macro_score_line = f"MACRO F1 SCORE: {sum(f1_list) / len(f1_list):.4f}"
    stats_lines.append(macro_score_line)
    stats_lines.append("\n" + "=" * 40 + "\n")

    # Combine blocks: Metadata info -> Final Statistics Table -> Detailed
    # Conflicts
    final_report_content = (
        "\n".join(meta_lines)
        + "\n"
        + "\n".join(stats_lines)
        + "\n"
        + "\n".join(conflict_lines)
    )

    # Print consolidated results immediately to console
    print("\n" + "\n".join(meta_lines))
    print("\n".join(stats_lines))

    # Flush full combined content to the persistent text file
    report_name = f"eval_results_{timestamp_str}.txt"
    with open(report_name, "w", encoding="utf-8") as f:
        f.write(final_report_content)

    print(f"✅ Evaluation complete. Report file generated: {report_name}")
    if n_documents > 1:
        print(
            f"⏱️ Average: {format_time(avg_time)} | "
            f"Total Batch: {format_time(total_batch_time)}"
        )


if __name__ == "__main__":
    run_evaluation()
