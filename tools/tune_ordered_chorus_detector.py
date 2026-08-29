from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import re
import sys
from collections import Counter
from dataclasses import asdict
from pathlib import Path
from typing import Iterable, Mapping, Sequence

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from sound_diversity.chorus import (
    TimedWord,
    detect_counted_sections,
    evaluate_annotations,
    format_intervals,
    interval_duration,
    interval_overlap,
    lyric_tokens,
    merge_intervals,
    parse_intervals,
    parse_lyric_sections,
)
from sound_diversity.core import stable_seed


def read_rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8-sig") as handle:
        return list(csv.DictReader(handle))


def write_rows(path: Path, rows: Iterable[Mapping[str, object]], fields: Sequence[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(fields), extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def leading_file_id(path: str) -> str:
    match = re.match(r"^(\d+)_", Path(path or "").name)
    return str(int(match.group(1))) if match else ""


def cache_key(path: Path, model_name: str) -> str:
    stat = path.stat()
    text = f"{path.resolve()}|{stat.st_size}|{int(stat.st_mtime)}|{model_name}"
    return hashlib.sha1(text.encode("utf-8")).hexdigest()


def transcript_cache_path(cache_dir: Path, audio_path: Path, model_name: str) -> Path:
    return cache_dir / f"{cache_key(audio_path, model_name)}.json"


def words_from_whisper_result(result: Mapping[str, object]) -> list[TimedWord]:
    words: list[TimedWord] = []
    for segment in result.get("segments", []):  # type: ignore[union-attr]
        for item in segment.get("words", []):
            raw = str(item.get("word", "")).strip()
            tokens = lyric_tokens(raw)
            if not tokens:
                continue
            start = float(item["start"])
            end = float(item["end"])
            for token in tokens:
                words.append(TimedWord(token=token, start=start, end=end))
    return words


def load_or_transcribe(audio_path: Path, model, model_name: str, cache_dir: Path) -> list[TimedWord]:
    cache_dir.mkdir(parents=True, exist_ok=True)
    cache_path = transcript_cache_path(cache_dir, audio_path, model_name)
    if cache_path.exists():
        payload = json.loads(cache_path.read_text(encoding="utf-8"))
        return [TimedWord(**item) for item in payload["words"]]

    result = model.transcribe(str(audio_path), word_timestamps=True, verbose=False)
    words = words_from_whisper_result(result)
    cache_path.write_text(
        json.dumps({"audio_path": str(audio_path), "model": model_name, "words": [asdict(w) for w in words]}, indent=2),
        encoding="utf-8",
    )
    return words


def load_cached_transcript(audio_path: Path, model_name: str, cache_dir: Path) -> list[TimedWord] | None:
    cache_path = transcript_cache_path(cache_dir, audio_path, model_name)
    if not cache_path.exists():
        return None
    payload = json.loads(cache_path.read_text(encoding="utf-8"))
    return [TimedWord(**item) for item in payload["words"]]


def chunk_cached_block(block: str, chunk_tokens: int, chunk_step: int, min_tokens: int) -> list[str]:
    tokens = lyric_tokens(block)
    if len(tokens) < min_tokens:
        return []
    if len(tokens) <= chunk_tokens:
        return [" ".join(tokens)]
    chunks: list[str] = []
    seen: set[str] = set()
    starts = list(range(0, max(1, len(tokens) - chunk_tokens + 1), chunk_step))
    last_start = max(0, len(tokens) - chunk_tokens)
    if starts[-1] != last_start:
        starts.append(last_start)
    for start in starts:
        chunk = " ".join(tokens[start:start + chunk_tokens])
        if chunk and chunk not in seen:
            chunks.append(chunk)
            seen.add(chunk)
    return chunks


def cached_chorus_text_to_sections(text: str, chunk_tokens: int, chunk_step: int, min_tokens: int):
    blocks = [block.strip() for block in (text or "").split(" || ") if block.strip()]
    lines = [
        line for block in blocks
        for line in chunk_cached_block(block, chunk_tokens, chunk_step, min_tokens)
    ]
    if not lines:
        return []
    pseudo_lyrics = "[Chorus]\n" + "\n".join(lines)
    return parse_lyric_sections(pseudo_lyrics)


def make_detection_rows(
    annotations: list[dict[str, str]],
    old_results: list[dict[str, str]],
    model_name: str,
    cache_dir: Path,
    chunk_tokens: int,
    chunk_step: int,
    min_chunk_tokens: int,
    transcribe_limit: int | None = None,
) -> tuple[list[dict[str, object]], dict[str, object]]:
    import whisper

    old_by_id = {leading_file_id(row.get("audio_path", "")): row for row in old_results if leading_file_id(row.get("audio_path", ""))}
    rows: list[dict[str, object]] = []
    to_transcribe = []
    for annotation in annotations:
        old = old_by_id.get(leading_file_id(annotation.get("audio_path", "")), {})
        sections = cached_chorus_text_to_sections(
            old.get("chorus_text_from_genius", ""),
            chunk_tokens=chunk_tokens,
            chunk_step=chunk_step,
            min_tokens=min_chunk_tokens,
        )
        audio_path = Path(annotation.get("audio_path", ""))
        target_available = bool(sections)
        audio_exists = audio_path.exists()
        detection_status = "ready" if target_available and audio_exists else "no_cached_genius_text" if not target_available else "missing_audio"
        row = {
            **annotation,
            "_old_chorus_text": old.get("chorus_text_from_genius", ""),
            "_target_available": target_available,
            "_audio_exists": audio_exists,
            "_detection_status": detection_status,
            "_sections": sections,
            "_words": None,
        }
        rows.append(row)
        if detection_status == "ready":
            to_transcribe.append(row)

    if transcribe_limit is not None:
        to_transcribe = to_transcribe[:transcribe_limit]

    model = None
    for index, row in enumerate(to_transcribe, start=1):
        audio_path = Path(str(row["audio_path"]))
        print(f"Transcribing {index}/{len(to_transcribe)}: {row['sample_id']} {row['artist']} - {row['title']}", flush=True)
        try:
            cached = load_cached_transcript(audio_path, model_name, cache_dir)
            if cached is not None:
                row["_words"] = cached
            else:
                if model is None:
                    model = whisper.load_model(model_name)
                row["_words"] = load_or_transcribe(audio_path, model, model_name, cache_dir)
            row["_detection_status"] = "transcribed"
        except Exception as exc:  # noqa: BLE001 - script should keep going and expose row-level failures
            row["_detection_status"] = f"transcription_error: {exc}"

    summary = {
        "rows": len(rows),
        "ready_rows": sum(1 for row in rows if row["_target_available"] and row["_audio_exists"]),
        "transcribed_rows": sum(1 for row in rows if row["_detection_status"] == "transcribed"),
        "status_counts": dict(Counter(str(row["_detection_status"]).split(":", 1)[0] for row in rows)),
    }
    return rows, summary


def split_ids(rows: list[Mapping[str, object]], holdout_size: int, seed: int) -> tuple[set[str], set[str]]:
    labelled = [
        row for row in rows
        if str(row.get("human_chorus_present", "")).casefold() in {"yes", "no"}
        and row.get("_target_available")
        and row.get("_words")
    ]
    yes = [row for row in labelled if str(row.get("human_chorus_present", "")).casefold() == "yes"]
    no = [row for row in labelled if str(row.get("human_chorus_present", "")).casefold() == "no"]
    for bucket_name, bucket in (("yes", yes), ("no", no)):
        bucket.sort(key=lambda row: stable_seed(seed, "ordered-chorus-holdout", bucket_name, str(row["sample_id"])))

    holdout_yes = min(len(yes), max(1, round(holdout_size * len(yes) / max(1, len(labelled)))))
    holdout_no = min(len(no), holdout_size - holdout_yes)
    holdout = {str(row["sample_id"]) for row in yes[:holdout_yes] + no[:holdout_no]}
    tune = {str(row["sample_id"]) for row in labelled if str(row["sample_id"]) not in holdout}
    return tune, holdout


def expand_intervals(
    intervals: Sequence[tuple[float, float]], pad_seconds: float, preview_end: float = 30.0
) -> list[tuple[float, float]]:
    if pad_seconds <= 0:
        return list(intervals)
    return merge_intervals(
        (
            (max(0.0, start - pad_seconds), min(preview_end, end + pad_seconds))
            for start, end in intervals
        ),
        gap_tolerance=0.0,
    )


def predict_for_params(
    rows: list[Mapping[str, object]],
    min_score: float,
    min_query_tokens: int,
    max_iou: float,
    gap_tolerance: float,
    pad_seconds: float,
) -> list[dict[str, str]]:
    predictions: list[dict[str, str]] = []
    for row in rows:
        output = {key: str(value) for key, value in row.items() if not key.startswith("_")}
        sections = row.get("_sections") or []
        words = row.get("_words") or []
        if sections and words:
            intervals, matches = detect_counted_sections(
                sections, words, min_score=min_score, min_query_tokens=min_query_tokens,
                max_iou=max_iou, gap_tolerance=gap_tolerance,
            )
            intervals = expand_intervals(intervals, pad_seconds=pad_seconds)
            output["ordered_predicted_intervals"] = format_intervals(intervals)
            output["ordered_n_matches"] = str(len(matches))
        else:
            output["ordered_predicted_intervals"] = ""
            output["ordered_n_matches"] = "0"
        output["ordered_detection_status"] = str(row.get("_detection_status", ""))
        predictions.append(output)
    return predictions


def metric_f1(precision: float, recall: float) -> float:
    return 2 * precision * recall / (precision + recall) if precision + recall else 0.0


def evaluate_prediction_rows(rows: list[dict[str, str]], id_subset: set[str] | None = None) -> dict[str, float | int]:
    selected = [row for row in rows if id_subset is None or row.get("sample_id", "") in id_subset]
    remapped = []
    for row in selected:
        item = dict(row)
        item["predicted_intervals"] = row.get("ordered_predicted_intervals", "")
        remapped.append(item)
    metrics = evaluate_annotations(remapped)
    metrics["presence_f1"] = metric_f1(float(metrics["presence_precision"]), float(metrics["presence_recall"]))
    metrics["temporal_f1"] = metric_f1(float(metrics["temporal_precision"]), float(metrics["temporal_recall"]))
    metrics["objective"] = 0.45 * float(metrics["presence_f1"]) + 0.55 * float(metrics["temporal_f1"])
    return metrics


def row_level_outputs(predictions: list[dict[str, str]]) -> list[dict[str, str]]:
    output = []
    for row in predictions:
        predicted = parse_intervals(row.get("ordered_predicted_intervals", ""))
        actual = parse_intervals(row.get("human_chorus_intervals", ""))
        pred_seconds = interval_duration(predicted)
        actual_seconds = interval_duration(actual)
        overlap = interval_overlap(predicted, actual)
        human = row.get("human_chorus_present", "").casefold()
        predicted_present = bool(predicted)
        if human == "yes" and predicted_present:
            confusion = "TP"
        elif human == "yes":
            confusion = "FN"
        elif human == "no" and predicted_present:
            confusion = "FP"
        elif human == "no":
            confusion = "TN"
        else:
            confusion = "unscored"
        output.append({
            **row,
            "ordered_confusion": confusion,
            "ordered_predicted_seconds": f"{pred_seconds:.2f}",
            "human_seconds": f"{actual_seconds:.2f}",
            "ordered_overlap_seconds": f"{overlap:.2f}",
            "ordered_row_temporal_precision": f"{(overlap / pred_seconds if pred_seconds else 0.0):.4f}",
            "ordered_row_temporal_recall": f"{(overlap / actual_seconds if actual_seconds else 0.0):.4f}",
        })
    return output


def float_grid(text: str) -> list[float]:
    return [float(item.strip()) for item in text.split(",") if item.strip()]


def int_grid(text: str) -> list[int]:
    return [int(item.strip()) for item in text.split(",") if item.strip()]


def main() -> int:
    parser = argparse.ArgumentParser(description="Tune ordered chorus matcher using cached old Genius chorus text.")
    parser.add_argument("--annotations", type=Path, default=Path("data/chorus_annotations_completed.csv"))
    parser.add_argument("--old-results", type=Path, required=True)
    parser.add_argument("--model", default="small")
    parser.add_argument("--cache-dir", type=Path, default=Path("data/cache/chorus_transcripts"))
    parser.add_argument("--holdout-size", type=int, default=10)
    parser.add_argument("--seed", type=int, default=20260708)
    parser.add_argument("--min-scores", default="0.58,0.62,0.66,0.70,0.74,0.78")
    parser.add_argument("--min-query-tokens", default="3,4,5")
    parser.add_argument("--max-ious", default="0.25,0.35,0.50")
    parser.add_argument("--gap-tolerances", default="0.15,0.35,0.75")
    parser.add_argument("--pads", default="0,2,4,6")
    parser.add_argument("--transcribe-limit", type=int)
    parser.add_argument("--chunk-tokens", type=int, default=8)
    parser.add_argument("--chunk-step", type=int, default=4)
    parser.add_argument("--min-chunk-tokens", type=int, default=4)
    parser.add_argument("--grid-output", type=Path, default=Path("outputs/chorus_rebuild/ordered_detector_tuning_grid.csv"))
    parser.add_argument("--predictions-output", type=Path, default=Path("outputs/chorus_rebuild/ordered_detector_predictions.csv"))
    parser.add_argument("--summary-output", type=Path, default=Path("outputs/chorus_rebuild/ordered_detector_tuning_summary.json"))
    args = parser.parse_args()

    annotations = read_rows(args.annotations)
    old_results = read_rows(args.old_results)
    rows, prep_summary = make_detection_rows(
        annotations, old_results, args.model, args.cache_dir,
        chunk_tokens=args.chunk_tokens, chunk_step=args.chunk_step,
        min_chunk_tokens=args.min_chunk_tokens, transcribe_limit=args.transcribe_limit
    )
    tune_ids, holdout_ids = split_ids(rows, holdout_size=args.holdout_size, seed=args.seed)

    grid_rows: list[dict[str, object]] = []
    best: tuple[float, dict[str, object], list[dict[str, str]]] | None = None
    for min_score in float_grid(args.min_scores):
        for min_query_tokens in int_grid(args.min_query_tokens):
                for max_iou in float_grid(args.max_ious):
                    for gap_tolerance in float_grid(args.gap_tolerances):
                        for pad_seconds in float_grid(args.pads):
                            predictions = predict_for_params(
                                rows, min_score, min_query_tokens, max_iou, gap_tolerance, pad_seconds
                            )
                            tune_metrics = evaluate_prediction_rows(predictions, tune_ids)
                            holdout_metrics = evaluate_prediction_rows(predictions, holdout_ids)
                            all_metrics = evaluate_prediction_rows(predictions)
                            record = {
                                "min_score": min_score,
                                "min_query_tokens": min_query_tokens,
                                "max_iou": max_iou,
                                "gap_tolerance": gap_tolerance,
                                "pad_seconds": pad_seconds,
                                **{f"tune_{k}": v for k, v in tune_metrics.items()},
                                **{f"holdout_{k}": v for k, v in holdout_metrics.items()},
                                **{f"all_{k}": v for k, v in all_metrics.items()},
                            }
                            grid_rows.append(record)
                            score = float(tune_metrics["objective"])
                            if best is None or score > best[0] or (
                                math.isclose(score, best[0])
                                and float(tune_metrics["presence_recall"]) > float(best[1]["tune_presence_recall"])
                            ):
                                best = (score, record, predictions)

    if best is None:
        raise SystemExit("No parameter combinations were evaluated.")

    grid_fields = list(grid_rows[0].keys())
    write_rows(args.grid_output, grid_rows, grid_fields)

    best_record = best[1]
    best_predictions = row_level_outputs(best[2])
    prediction_fields = [
        "sample_id", "artist", "title", "human_chorus_present", "human_chorus_intervals",
        "predicted_intervals", "ordered_predicted_intervals", "ordered_confusion",
        "ordered_detection_status", "ordered_n_matches", "ordered_predicted_seconds",
        "human_seconds", "ordered_overlap_seconds", "ordered_row_temporal_precision",
        "ordered_row_temporal_recall", "notes",
    ]
    write_rows(args.predictions_output, best_predictions, prediction_fields)

    summary = {
        "model": args.model,
        "cached_chorus_text_chunking": {
            "chunk_tokens": args.chunk_tokens,
            "chunk_step": args.chunk_step,
            "min_chunk_tokens": args.min_chunk_tokens,
        },
        "preparation": prep_summary,
        "tune_ids": sorted(tune_ids),
        "holdout_ids": sorted(holdout_ids),
        "best_parameters": {
            "min_score": best_record["min_score"],
            "min_query_tokens": best_record["min_query_tokens"],
            "max_iou": best_record["max_iou"],
            "gap_tolerance": best_record["gap_tolerance"],
            "pad_seconds": best_record["pad_seconds"],
        },
        "best_metrics": best_record,
        "outputs": {
            "grid": str(args.grid_output),
            "predictions": str(args.predictions_output),
        },
    }
    args.summary_output.parent.mkdir(parents=True, exist_ok=True)
    args.summary_output.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(args.grid_output)
    print(args.predictions_output)
    print(args.summary_output)
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
