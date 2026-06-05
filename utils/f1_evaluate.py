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
from typing import Any, Dict, Tuple

try:
    from thefuzz import fuzz
except ImportError:
    print("❌ The 'thefuzz' library is not installed. Run: pip install thefuzz[speedup]")
    exit(1)


# ==============================================================================
# CONFIGURATION AND GLOBAL CONSTANTS
# ==============================================================================

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
FUZZY_FIELDS = {"title", "authors", "publisher", "subjects"}

# Minimum similarity score required to consider a fuzzy match successful (0-100)
FUZZY_THRESHOLD = 85


# ==============================================================================
# UTILITY AND NORMALIZATION FUNCTIONS
# ==============================================================================

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
    # Return None immediately if the input value is missing
    if val is None:
        return None

    # Handle multi-valued fields recursively and ensure deterministic ordering
    if isinstance(val, list):
        cleaned_list = []
        for v in val:
            norm_v = normalize_value(v, field_name)
            if norm_v:
                cleaned_list.append(norm_v)
        return sorted(cleaned_list) if cleaned_list else None

    # Convert to string, strip leading/trailing whitespaces, and lowercase
    cleaned = str(val).strip().lower()

    # Unify various types of hyphens and dashes into a standard hyphen
    cleaned = re.sub(r'[‐‑‒–—―]', '-', cleaned)

    # Specific cleaning rules for author names to bypass punctuation variances
    if field_name == "authors":
        cleaned = re.sub(r'[,.]', '', cleaned)

    # Treat explicit placeholder strings as missing values (None)
    if cleaned in ["n/a", "none", "null", "na", "", "-", "unknown", "neuvedené"]:
        return None

    # Isolate the publication year via regex if processing the 'issued' field
    if field_name == "issued":
        year_match = re.search(r'\b(18|19|20|21)\d{2}\b', cleaned)
        if year_match:
            cleaned = year_match.group(0)

    # Standardize document types to match target structural taxonomies
    if field_name in ["type", "resource_type"]:
        cleaned = re.sub(r'[-_]', ' ', cleaned)

        if "review article" in cleaned:
            cleaned = "review"
        elif "dissertation" in cleaned or "thesis" in cleaned:
            cleaned = "thesis"
        elif "article" in cleaned or "journal" in cleaned:
            cleaned = "article"

    return cleaned.strip() if cleaned.strip() else None


def calculate_f1(tp: int, fp: int, fn: int) -> Tuple[float, float, float]:
    """Calculate Precision, Recall, and the harmonic mean (F1-score).

    Args:
        tp (int): Number of True Positives.
        fp (int): Number of False Positives.
        fn (int): Number of False Negatives.

    Returns:
        Tuple[float, float, float]: Precision, Recall, and F1-score values.
    """
    # Prevent division by zero for precision calculation
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0

    # Prevent division by zero for recall calculation
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0

    # Calculate the F1-score using standard harmonic mean formula
    f1 = (2 * precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0

    return precision, recall, f1


def format_time(seconds: float) -> str:
    """Format a duration in seconds into a human-readable string.

    Args:
        seconds (float): Time duration in seconds.

    Returns:
        str: Formatted time string (seconds only or minutes and seconds).
    """
    if seconds < 60:
        return f"{seconds:.2f} s"
    mins = int(seconds // 60)
    secs = seconds % 60
    return f"{mins} min {secs:.2f} s"


# ==============================================================================
# MAIN EVALUATION PIPELINE
# ==============================================================================

def run_evaluation() -> None:
    """Execute the comprehensive metadata evaluation workflow.

    The pipeline loads ground truth and output files, analyzes the execution
    time patterns using file modification timestamps, performs field-by-field
    matching with optional fuzzy logic, penalizes anomalies, and outputs
    both console summary statistics and a persistent text report.
    """
    # Verify the integrity of the data directory structure before proceeding
    if not DIR_GROUND.exists() or not DIR_OUTPUT.exists():
        print("❌ CRITICAL ERROR: Data directories do not exist!")
        return

    # Extract unique identifiers based on ground truth file names
    ground_files = {f.stem for f in DIR_GROUND.glob("*.json")}

    if not ground_files:
        print("❌ No reference data found in data/ground.")
        return

    # --- TIMING METRICS ESTIMATION CODE ---
    # Gather all output artifacts to construct the processing timeline
    json_outputs = list(DIR_OUTPUT.glob("*_metadata.json"))
    raw_outputs = list(DIR_OUTPUT.glob("*_raw.txt"))
    all_files = json_outputs + raw_outputs

    # Identify unique documents by stripping output-specific suffixes
    processed_stems = set()
    for f in all_files:
        stem_cleaned = f.stem.replace("_metadata", "").replace("_raw", "")
        processed_stems.add(stem_cleaned)

    n_documents = len(processed_stems)
    avg_time = 0.0
    total_batch_time = 0.0

    if n_documents > 0 and all_files:
        # Sort files by modification time to find the start and end boundaries
        all_files_sorted = sorted(all_files, key=os.path.getmtime)
        t_start = all_files_sorted[0].stat().st_mtime
        t_end = all_files_sorted[-1].stat().st_mtime

        delta_t = t_end - t_start

        # Calculate intervals and interpolate total processing duration
        if n_documents > 1:
            avg_time = delta_t / (n_documents - 1)
            total_batch_time = delta_t + avg_time
        else:
            avg_time = delta_t
            total_batch_time = delta_t

    # Initialize log storage and generate current timestamps for reporting
    log_lines = []
    timestamp_str = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

    log_lines.append("=== METADATA EXTRACTION PERFORMANCE REPORT ===")
    log_lines.append(f"Evaluation Date: {datetime.now().strftime('%d.%m.%Y %H:%M:%S')}")
    log_lines.append(f"Fuzzy Match Threshold: {FUZZY_THRESHOLD}%\n")

    log_lines.append("=== PERFORMANCE AND TEMPORAL METRICS ===")
    log_lines.append(f"Number of unique processed documents: {n_documents}")
    if n_documents > 1:
        log_lines.append(f"Average processing time per document: {format_time(avg_time)}")
        log_lines.append(f"Total batch processing time: {format_time(total_batch_time)}")
        log_lines.append(f"Throughput: {3600 / avg_time:.2f} doc./hour.")
    else:
        log_lines.append("Insufficient data to compute valid timing metrics.")
    log_lines.append("-" * 40 + "\n")

    log_lines.append("=== CONFLICTS AND MISMATCHES ===")

    # Initialize confusion matrix counters for each schema attribute
    stats = {key: {"TP": 0, "FP": 0, "FN": 0} for key in SCHEMA_MAPPING.keys()}
    failed_extractions_count = 0

    # Iterate through ground truth records in a deterministic alphabetical order
    for stem in sorted(ground_files):
        ground_path = DIR_GROUND / f"{stem}.json"
        output_path = DIR_OUTPUT / f"{stem}_metadata.json"
        raw_path = DIR_OUTPUT / f"{stem}_raw.txt"

        # Attempt to parse the reference JSON file
        try:
            with open(ground_path, 'r', encoding='utf-8') as f:
                ground_json = json.load(f)
        except json.JSONDecodeError:
            print(f"⚠️ Skipping {stem}: Ground Truth JSON is corrupted.")
            continue

        output_json = {}
        is_fatal_failure = False
        failure_reason = ""

        # Check for model failures and classify the nature of the error
        if output_path.exists():
            try:
                with open(output_path, 'r', encoding='utf-8') as f:
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
            log_lines.append(f"\n📄 File: {stem} 🚨 [FATAL FAILURE: {failure_reason}]")
            file_has_error = True

        # Process each metadata field present in the schema mapping
        for ground_key, output_key in SCHEMA_MAPPING.items():
            val_g = normalize_value(ground_json.get(ground_key), ground_key)
            val_o = normalize_value(output_json.get(output_key), output_key)

            # If both values are missing, it counts as a successful non-extraction
            if not val_g and not val_o:
                stats[ground_key]["TP"] += 1
                continue

            is_match = False
            fuzzy_score = 0

            # Direct string comparison check
            if val_g == val_o:
                is_match = True
            # Fall back to fuzzy matching algorithms for designated fields
            elif val_g and val_o and ground_key in FUZZY_FIELDS:
                s_g = " ".join(val_g) if isinstance(val_g, list) else val_g
                s_o = " ".join(val_o) if isinstance(val_o, list) else val_o
                fuzzy_score = fuzz.token_set_ratio(s_g, s_o)
                if fuzzy_score >= FUZZY_THRESHOLD:
                    is_match = True

            if is_match:
                stats[ground_key]["TP"] += 1
            else:
                # Log detailed error information to help debug extraction weaknesses
                if not file_has_error:
                    log_lines.append(f"\n📄 File: {stem}")
                    file_has_error = True

                if not is_fatal_failure:
                    log_lines.append(f"  [X] Field: {ground_key}")
                    log_lines.append(f"      G: {ground_json.get(ground_key)} [Normalized: {val_g}]")
                    log_lines.append(f"      O: {output_json.get(output_key)} [Normalized: {val_o}]")
                    if fuzzy_score > 0:
                        log_lines.append(f"      Fuzzy Score: {fuzzy_score}%")

                # Adjust metric counts based on the error classification
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
    success_rate = (successful_files / total_files * 100) if total_files > 0 else 0.0
    failure_rate = (failed_extractions_count / total_files * 100) if total_files > 0 else 0.0

    # Display findings in the console terminal
    print("\n=== FINAL STATISTICS ===")
    print(f"✅ Successfully processed files: {successful_files} / {total_files} ({success_rate:.2f}%)")
    if failed_extractions_count > 0:
        print(f"⚠️ Fatal failures (penalized): {failed_extractions_count} / {total_files} ({failure_rate:.2f}%)")

    # Record final summaries into the document log array
    log_lines.append("\n=== FINAL STATISTICS ===")
    log_lines.append(f"Successfully processed files: {successful_files} / {total_files} ({success_rate:.2f}%)")
    log_lines.append(f"Fatal failures (penalized as FN): {failed_extractions_count} / {total_files} ({failure_rate:.2f}%)\n")

    # Construct the formatted data grid header
    header = f"{'Field':<15} | {'TP':<4} | {'FP':<4} | {'FN':<4} | {'P':<8} | {'R':<8} | {'F1':<8}"
    print(header)
    log_lines.append(header)

    separator = "-" * len(header)
    print(separator)
    log_lines.append(separator)

    f1_list = []
    # Calculate performance metrics for each mapped item
    for g_key in SCHEMA_MAPPING.keys():
        s = stats[g_key]
        p, r, f1 = calculate_f1(s["TP"], s["FP"], s["FN"])
        f1_list.append(f1)
        stat_line = f"{g_key:<15} | {s['TP']:<4} | {s['FP']:<4} | {s['FN']:<4} | {p:<8.4f} | {r:<8.4f} | {f1:<8.4f}"
        print(stat_line)
        log_lines.append(stat_line)

    print(separator)
    log_lines.append(separator)

    # Compute and display the global macro F1-score
    macro_score_line = f"MACRO F1 SCORE: {sum(f1_list)/len(f1_list):.4f}"
    print(macro_score_line)
    log_lines.append(macro_score_line)

    # Flush the log lines out to a persistent text report file
    report_name = f"eval_results_{timestamp_str}.txt"
    with open(report_name, 'w', encoding='utf-8') as f:
        f.write("\n".join(log_lines))

    print(f"\n✅ Evaluation complete. Report file generated: {report_name}")
    if n_documents > 1:
        print(f"⏱️ Average: {format_time(avg_time)} | Total Batch: {format_time(total_batch_time)}")


if __name__ == "__main__":
    run_evaluation()