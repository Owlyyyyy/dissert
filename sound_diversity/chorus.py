from __future__ import annotations

import csv
import math
import random
import re
import unicodedata
from dataclasses import dataclass
from difflib import SequenceMatcher
from pathlib import Path
from typing import Iterable, Mapping, Sequence

from .core import is_usable_artist, stable_seed


COUNTED_SECTION_KINDS = frozenset({"chorus", "hook", "refrain", "pre_chorus", "post_chorus"})
ANNOTATION_FIELDS = [
    "sample_id", "audio_path", "artist", "title", "metadata_source", "metadata_status",
    "detector_status", "predicted_chorus_percent", "predicted_intervals",
    "human_chorus_present", "human_chorus_intervals",
    "human_pre_chorus_intervals", "human_post_chorus_intervals",
    "reviewer", "notes",
]


def normalize_lyric_text(value: str) -> str:
    value = unicodedata.normalize("NFKD", value or "")
    value = "".join(c for c in value if not unicodedata.combining(c)).casefold()
    value = re.sub(r"[^a-z0-9']+", " ", value)
    return " ".join(value.split())


def lyric_tokens(value: str) -> list[str]:
    return normalize_lyric_text(value).split()


def classify_section_header(header: str) -> str:
    """Classify a Genius-style section without folding pre/post into chorus."""
    text = normalize_lyric_text(header.replace("-", " "))
    if re.search(r"\bpre chorus\b", text):
        return "pre_chorus"
    if re.search(r"\bpost chorus\b", text):
        return "post_chorus"
    if re.search(r"\bchorus\b", text):
        return "chorus"
    if re.search(r"\bhook\b", text):
        return "hook"
    if re.search(r"\brefrain\b", text):
        return "refrain"
    if re.search(r"\bverse\b", text):
        return "verse"
    if re.search(r"\bbridge\b", text):
        return "bridge"
    if re.search(r"\bintro\b", text):
        return "intro"
    if re.search(r"\boutro\b", text):
        return "outro"
    return "other"


@dataclass(frozen=True)
class LyricSection:
    header: str
    kind: str
    lines: tuple[str, ...]

    @property
    def counted_as_chorus(self) -> bool:
        return self.kind in COUNTED_SECTION_KINDS


def parse_lyric_sections(lyrics: str) -> list[LyricSection]:
    header_pattern = re.compile(r"^\s*\[([^]]+)]\s*$")
    sections: list[LyricSection] = []
    header = ""
    kind = "unlabelled"
    lines: list[str] = []

    def save() -> None:
        if header or lines:
            cleaned = tuple(line.strip() for line in lines if line.strip())
            sections.append(LyricSection(header, kind, cleaned))

    for raw_line in lyrics.splitlines():
        match = header_pattern.match(raw_line)
        if match:
            save()
            header = match.group(1).strip()
            kind = classify_section_header(header)
            lines = []
        elif raw_line.strip():
            lines.append(raw_line.strip())
    save()
    return sections


def resolve_track_identity(row: Mapping[str, str]) -> tuple[str, str, str]:
    """Prefer track-level provenance and never use a compilation's Various credit."""
    artist_candidates = (
        ("track_artist", row.get("track_artist", "")),
        ("artist", row.get("artist", "")),
        ("apple_artist", row.get("apple_artist", "")),
        ("release_artist", row.get("release_artist", "")),
    )
    artist_source, artist = next(
        ((source, str(value).strip()) for source, value in artist_candidates
         if is_usable_artist(str(value))),
        ("missing", ""),
    )
    title_candidates = (
        row.get("track_title", ""), row.get("title", ""), row.get("apple_track", "")
    )
    title = next((str(value).strip() for value in title_candidates if str(value).strip()), "")
    return artist, title, artist_source


def legacy_filename_title(audio_path: str, fallback: str = "") -> str:
    """Recover a legacy preview title without deleting version qualifiers."""
    stem = Path(audio_path or "").stem
    match = re.match(r"^(?:\d+_)?(?:.+?)_-_(.+)$", stem)
    if not match:
        return fallback
    return match.group(1).replace("_", " ").strip() or fallback


@dataclass(frozen=True)
class TimedWord:
    token: str
    start: float
    end: float


@dataclass(frozen=True)
class AlignmentCandidate:
    start_index: int
    end_index: int
    start: float
    end: float
    score: float
    section_kind: str
    lyric_line: str


def ordered_line_candidates(
    lyric_line: str,
    words: Sequence[TimedWord],
    section_kind: str,
    min_score: float = 0.72,
    min_query_tokens: int = 4,
) -> list[AlignmentCandidate]:
    """Find order-sensitive line matches while allowing modest ASR insertions/deletions."""
    query = lyric_tokens(lyric_line)
    if len(query) < min_query_tokens or not words:
        return []
    target = [word.token for word in words]
    minimum = max(min_query_tokens, math.floor(len(query) * 0.70))
    maximum = min(len(target), math.ceil(len(query) * 1.30))
    found: list[AlignmentCandidate] = []
    for window_size in range(minimum, maximum + 1):
        for start in range(0, len(target) - window_size + 1):
            end = start + window_size
            score = SequenceMatcher(None, query, target[start:end], autojunk=False).ratio()
            if score >= min_score:
                found.append(AlignmentCandidate(
                    start, end, words[start].start, words[end - 1].end, score,
                    section_kind, lyric_line,
                ))
    return found


def interval_iou(left: tuple[float, float], right: tuple[float, float]) -> float:
    overlap = max(0.0, min(left[1], right[1]) - max(left[0], right[0]))
    union = max(left[1], right[1]) - min(left[0], right[0])
    return overlap / union if union > 0 else 0.0


def select_distinct_matches(
    candidates: Iterable[AlignmentCandidate], max_iou: float = 0.35
) -> list[AlignmentCandidate]:
    """Keep the strongest candidate in each local region instead of unioning all hits."""
    selected: list[AlignmentCandidate] = []
    for candidate in sorted(candidates, key=lambda x: (-x.score, x.start, x.end)):
        interval = (candidate.start, candidate.end)
        if all(interval_iou(interval, (item.start, item.end)) <= max_iou for item in selected):
            selected.append(candidate)
    return sorted(selected, key=lambda x: (x.start, x.end))


def merge_intervals(
    intervals: Iterable[tuple[float, float]], gap_tolerance: float = 0.35
) -> list[tuple[float, float]]:
    ordered = sorted((float(start), float(end)) for start, end in intervals if end > start)
    if not ordered:
        return []
    merged = [ordered[0]]
    for start, end in ordered[1:]:
        previous_start, previous_end = merged[-1]
        if start <= previous_end + gap_tolerance:
            merged[-1] = (previous_start, max(previous_end, end))
        else:
            merged.append((start, end))
    return merged


def detect_counted_sections(
    sections: Sequence[LyricSection],
    words: Sequence[TimedWord],
    min_score: float = 0.72,
    min_query_tokens: int = 4,
    max_iou: float = 0.35,
    gap_tolerance: float = 0.35,
) -> tuple[list[tuple[float, float]], list[AlignmentCandidate]]:
    candidates: list[AlignmentCandidate] = []
    for section in sections:
        if not section.counted_as_chorus:
            continue
        for line in section.lines:
            candidates.extend(ordered_line_candidates(
                line, words, section.kind, min_score, min_query_tokens
            ))
    selected = select_distinct_matches(candidates, max_iou=max_iou)
    intervals = merge_intervals(
        ((item.start, item.end) for item in selected),
        gap_tolerance=gap_tolerance,
    )
    return intervals, selected


def format_intervals(intervals: Iterable[tuple[float, float]]) -> str:
    return ";".join(f"{start:.2f}-{end:.2f}" for start, end in intervals)


def parse_time_value(value: str) -> float:
    value = (value or "").strip()
    minsec_match = re.fullmatch(r"(\d+)\s*:\s*(\d+(?:\.\d+)?)", value)
    if minsec_match:
        minutes, seconds = minsec_match.groups()
        return int(minutes) * 60 + float(seconds)
    seconds_match = re.fullmatch(r"\d+(?:\.\d+)?", value)
    if seconds_match:
        return float(value)
    raise ValueError(f"Invalid time value: {value!r}")


def parse_intervals(value: str) -> list[tuple[float, float]]:
    if not (value or "").strip():
        return []
    parsed = []
    for item in value.split(";"):
        if "-" not in item:
            raise ValueError(f"Invalid interval: {item!r}")
        start_text, end_text = item.split("-", 1)
        start = parse_time_value(start_text)
        end = parse_time_value(end_text)
        if end <= start:
            raise ValueError(f"Interval end must be after start: {item!r}")
        parsed.append((start, end))
    return merge_intervals(parsed, gap_tolerance=0.0)


def interval_duration(intervals: Iterable[tuple[float, float]]) -> float:
    return sum(end - start for start, end in merge_intervals(intervals, gap_tolerance=0.0))


def interval_overlap(
    predicted: Sequence[tuple[float, float]], actual: Sequence[tuple[float, float]]
) -> float:
    return sum(
        max(0.0, min(p_end, a_end) - max(p_start, a_start))
        for p_start, p_end in predicted for a_start, a_end in actual
    )


def evaluate_annotations(rows: Sequence[Mapping[str, str]]) -> dict[str, float | int]:
    labelled = [row for row in rows if row.get("human_chorus_present", "").casefold() in {"yes", "no"}]
    tp = fp = tn = fn = 0
    overlap = predicted_duration = actual_duration = 0.0
    for row in labelled:
        predicted = parse_intervals(row.get("predicted_intervals", ""))
        actual = parse_intervals(row.get("human_chorus_intervals", ""))
        predicted_present = bool(predicted)
        actual_present = row.get("human_chorus_present", "").casefold() == "yes"
        tp += int(predicted_present and actual_present)
        fp += int(predicted_present and not actual_present)
        tn += int(not predicted_present and not actual_present)
        fn += int(not predicted_present and actual_present)
        overlap += interval_overlap(predicted, actual)
        predicted_duration += interval_duration(predicted)
        actual_duration += interval_duration(actual)
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    temporal_precision = overlap / predicted_duration if predicted_duration else 0.0
    temporal_recall = overlap / actual_duration if actual_duration else 0.0
    return {
        "labelled_rows": len(labelled), "true_positive": tp, "false_positive": fp,
        "true_negative": tn, "false_negative": fn, "presence_precision": precision,
        "presence_recall": recall, "temporal_precision": temporal_precision,
        "temporal_recall": temporal_recall,
    }


def stratified_annotation_sample(
    rows: Sequence[Mapping[str, str]], size: int = 40, seed: int = 20260627
) -> list[dict[str, str]]:
    """Sample errors, zeroes, middle values, and high values without duplicate songs."""
    strata: dict[str, list[Mapping[str, str]]] = {"error": [], "zero": [], "middle": [], "high": []}
    for row in rows:
        if row.get("status") != "ok":
            strata["error"].append(row)
            continue
        try:
            percentage = float(row.get("chorus_percent", ""))
        except (TypeError, ValueError):
            strata["error"].append(row)
            continue
        strata["zero" if percentage == 0 else "high" if percentage >= 75 else "middle"].append(row)

    targets = {"error": size // 5, "zero": size * 3 // 10, "middle": size // 5, "high": size - (size // 5 + size * 3 // 10 + size // 5)}
    chosen: list[Mapping[str, str]] = []
    seen: set[tuple[str, str]] = set()
    for name in ("error", "zero", "middle", "high"):
        candidates = list(strata[name])
        random.Random(stable_seed(seed, "chorus-annotation", name)).shuffle(candidates)
        taken = 0
        for row in candidates:
            key = (normalize_lyric_text(row.get("artist", "")), normalize_lyric_text(row.get("title", "")))
            if key in seen:
                continue
            chosen.append(row)
            seen.add(key)
            taken += 1
            if taken >= targets[name]:
                break
    if len(chosen) < size:
        leftovers = [row for group in strata.values() for row in group if (normalize_lyric_text(row.get("artist", "")), normalize_lyric_text(row.get("title", ""))) not in seen]
        random.Random(stable_seed(seed, "chorus-annotation", "fill")).shuffle(leftovers)
        chosen.extend(leftovers[:size - len(chosen)])

    output = []
    for index, row in enumerate(chosen[:size], 1):
        artist = row.get("artist", "")
        title = legacy_filename_title(row.get("audio_path", ""), row.get("title", ""))
        output.append({
            "sample_id": f"A{index:03d}", "audio_path": row.get("audio_path", ""),
            "artist": artist, "title": title,
            "metadata_source": row.get("metadata_source", ""),
            "metadata_status": row.get("metadata_status", "") or ("filename_only" if is_usable_artist(artist) else "metadata_unresolved"),
            "detector_status": row.get("status", ""),
            "predicted_chorus_percent": row.get("chorus_percent", ""),
            "predicted_intervals": row.get("intervals", ""),
            "human_chorus_present": "", "human_chorus_intervals": "",
            "human_pre_chorus_intervals": "", "human_post_chorus_intervals": "",
            "reviewer": "", "notes": "",
        })
    return output


def clean_annotation_source_rows(
    clean_manifest: Sequence[Mapping[str, str]], legacy_results: Sequence[Mapping[str, str]]
) -> list[dict[str, str]]:
    """Join validated files to legacy predictions without restoring legacy identity fields."""
    legacy_by_id: dict[str, Mapping[str, str]] = {}
    for row in legacy_results:
        match = re.match(r"^(\d+)_", Path(row.get("audio_path", "")).name)
        if match:
            legacy_by_id[str(int(match.group(1)))] = row
    output = []
    for row in clean_manifest:
        if row.get("status") != "ok" or not is_usable_artist(row.get("artist", "")):
            continue
        legacy = legacy_by_id.get(str(int(row["song_id"])), {})
        same_audio = row.get("audio_source") == "verified_legacy_copy"
        output.append({
            "audio_path": row.get("output_path", ""), "artist": row.get("artist", ""),
            "title": row.get("title", ""), "metadata_source": row.get("artist_source", ""),
            "metadata_status": f"verified_{row.get('artist_source', 'artist')}_artist",
            "status": legacy.get("status", "") if same_audio else "new_audio_unscored",
            "chorus_percent": legacy.get("chorus_percent", "") if same_audio else "",
            "intervals": legacy.get("intervals", "") if same_audio else "",
        })
    return output


def read_dict_rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8-sig") as handle:
        return list(csv.DictReader(handle))


def write_dict_rows(path: Path, rows: Iterable[Mapping[str, object]], fields: Sequence[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)
