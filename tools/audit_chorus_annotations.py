from __future__ import annotations

import argparse
import csv
import json
import re
import sys
from collections import Counter
from pathlib import Path
from typing import Iterable, Mapping

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from sound_diversity.chorus import (
    evaluate_annotations,
    interval_duration,
    interval_overlap,
    parse_intervals,
)


def read_rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8-sig") as handle:
        return list(csv.DictReader(handle))


def write_rows(path: Path, rows: Iterable[Mapping[str, object]], fields: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def leading_file_id(path: str) -> str:
    match = re.match(r"^(\d+)_", Path(path or "").name)
    return str(int(match.group(1))) if match else ""


def safe_int(value: str) -> int | None:
    try:
        if value in (None, ""):
            return None
        return int(float(str(value)))
    except ValueError:
        return None


def row_confusion(human_present: str, predicted_present: bool) -> str:
    if human_present == "yes" and predicted_present:
        return "TP"
    if human_present == "yes" and not predicted_present:
        return "FN"
    if human_present == "no" and predicted_present:
        return "FP"
    if human_present == "no" and not predicted_present:
        return "TN"
    return "unscored"


def reason_for(row: Mapping[str, str], old: Mapping[str, str], confusion: str, temporal_precision: float, temporal_recall: float) -> str:
    human_present = row.get("human_chorus_present", "").casefold().strip()
    detector_status = row.get("detector_status", "").strip()
    old_error = (old.get("error", "") or "").strip()
    n_blocks = safe_int(old.get("n_chorus_blocks_from_genius", ""))
    n_raw_matches = safe_int(old.get("n_raw_matches", ""))
    predicted_duration = interval_duration(parse_intervals(row.get("predicted_intervals", "")))

    if human_present == "uncertain":
        return "not scored: human uncertain"
    if detector_status == "new_audio_unscored":
        return "old baseline unavailable: clean preview was rebuilt with fresh audio"
    if detector_status == "error":
        lowered = old_error.casefold()
        if "could not find lyrics" in lowered:
            return "old detector error: Genius lyrics not found"
        if "no chorus/refrain/hook" in lowered or "no chorus" in lowered:
            return "old detector error: no labelled chorus/hook/refrain in Genius"
        if "whisper produced no word" in lowered:
            return "old detector error: no ASR word timestamps"
        if "could not determine artist/title" in lowered:
            return "old detector error: metadata unresolved"
        return "old detector error before matching"
    if confusion == "FN":
        if n_blocks == 0:
            return "miss: Genius had no labelled chorus/hook/refrain block"
        if n_raw_matches == 0:
            return "miss: Genius chorus text existed but alignment found no match"
        return "miss: old detector predicted no chorus despite available lyrics/matches"
    if confusion == "FP":
        if predicted_duration >= 25:
            return "false positive: old detector over-merged to most/all of preview"
        return "false positive: predicted chorus where human marked no chorus"
    if confusion == "TP":
        if temporal_recall < 0.60 and temporal_precision >= 0.70:
            return "presence correct: under-captured chorus duration"
        if temporal_precision < 0.60 and temporal_recall >= 0.70:
            return "presence correct: over-extended beyond chorus"
        if temporal_precision < 0.60 or temporal_recall < 0.60:
            return "presence correct: timing only partly overlaps"
        return "good match"
    if confusion == "TN":
        return "correct absence"
    return "not scored"


def audit_rows(annotation_rows: list[dict[str, str]], old_rows: list[dict[str, str]]) -> list[dict[str, object]]:
    old_by_id = {leading_file_id(row.get("audio_path", "")): row for row in old_rows if leading_file_id(row.get("audio_path", ""))}
    audit: list[dict[str, object]] = []
    for row in annotation_rows:
        old = old_by_id.get(leading_file_id(row.get("audio_path", "")), {})
        predicted = parse_intervals(row.get("predicted_intervals", ""))
        actual = parse_intervals(row.get("human_chorus_intervals", ""))
        predicted_duration = interval_duration(predicted)
        actual_duration = interval_duration(actual)
        overlap = interval_overlap(predicted, actual)
        temporal_precision = overlap / predicted_duration if predicted_duration else 0.0
        temporal_recall = overlap / actual_duration if actual_duration else 0.0
        human_present = row.get("human_chorus_present", "").casefold().strip()
        confusion = row_confusion(human_present, bool(predicted))
        scored = human_present in {"yes", "no"}
        audit.append({
            "sample_id": row.get("sample_id", ""),
            "artist": row.get("artist", ""),
            "title": row.get("title", ""),
            "metadata_status": row.get("metadata_status", ""),
            "detector_status": row.get("detector_status", ""),
            "human_chorus_present": row.get("human_chorus_present", ""),
            "human_chorus_intervals": row.get("human_chorus_intervals", ""),
            "predicted_intervals": row.get("predicted_intervals", ""),
            "scored": int(scored),
            "confusion": confusion,
            "predicted_seconds": f"{predicted_duration:.2f}",
            "human_seconds": f"{actual_duration:.2f}",
            "overlap_seconds": f"{overlap:.2f}",
            "row_temporal_precision": f"{temporal_precision:.4f}",
            "row_temporal_recall": f"{temporal_recall:.4f}",
            "duration_error_seconds": f"{predicted_duration - actual_duration:.2f}",
            "old_n_chorus_blocks_from_genius": old.get("n_chorus_blocks_from_genius", ""),
            "old_n_raw_matches": old.get("n_raw_matches", ""),
            "old_error": old.get("error", ""),
            "likely_reason": reason_for(row, old, confusion, temporal_precision, temporal_recall),
            "notes": row.get("notes", ""),
        })
    return audit


def summarize(audit: list[dict[str, object]], annotations: list[dict[str, str]]) -> dict[str, object]:
    scored = [row for row in audit if row["scored"]]
    by_confusion = Counter(str(row["confusion"]) for row in scored)
    by_reason = Counter(str(row["likely_reason"]) for row in scored)
    fn_reasons = Counter(str(row["likely_reason"]) for row in scored if row["confusion"] == "FN")
    fp_reasons = Counter(str(row["likely_reason"]) for row in scored if row["confusion"] == "FP")
    return {
        "rows": len(audit),
        "scored_rows": len(scored),
        "human_present_counts": dict(Counter(row.get("human_chorus_present", "").casefold().strip() or "(blank)" for row in annotations).most_common()),
        "metrics": evaluate_annotations(annotations),
        "confusion_counts": dict(sorted(by_confusion.items())),
        "likely_reason_counts": dict(by_reason.most_common()),
        "false_negative_reasons": dict(fn_reasons.most_common()),
        "false_positive_reasons": dict(fp_reasons.most_common()),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Audit old chorus detector errors against completed annotations.")
    parser.add_argument("--annotations", type=Path, default=Path("data/chorus_annotations_completed.csv"))
    parser.add_argument("--old-results", type=Path, required=True)
    parser.add_argument("--audit-output", type=Path, default=Path("outputs/chorus_rebuild/old_detector_error_audit.csv"))
    parser.add_argument("--summary-output", type=Path, default=Path("outputs/chorus_rebuild/old_detector_error_summary.json"))
    args = parser.parse_args()

    annotations = read_rows(args.annotations)
    old_rows = read_rows(args.old_results) if args.old_results.exists() else []
    audit = audit_rows(annotations, old_rows)
    fields = [
        "sample_id", "artist", "title", "metadata_status", "detector_status",
        "human_chorus_present", "human_chorus_intervals", "predicted_intervals",
        "scored", "confusion", "predicted_seconds", "human_seconds", "overlap_seconds",
        "row_temporal_precision", "row_temporal_recall", "duration_error_seconds",
        "old_n_chorus_blocks_from_genius", "old_n_raw_matches", "old_error",
        "likely_reason", "notes",
    ]
    write_rows(args.audit_output, audit, fields)
    summary = summarize(audit, annotations)
    args.summary_output.parent.mkdir(parents=True, exist_ok=True)
    args.summary_output.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(args.audit_output)
    print(args.summary_output)
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
