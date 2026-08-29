from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import re
import sys
import time
from dataclasses import asdict
from difflib import SequenceMatcher
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from sound_diversity.chorus import (
    TimedWord,
    detect_counted_sections,
    format_intervals,
    interval_duration,
    lyric_tokens,
    merge_intervals,
    parse_lyric_sections,
)
from sound_diversity.core import is_usable_artist, load_env_file, normalize


RESULT_FIELDS = [
    "song_id",
    "audio_path",
    "artist",
    "title",
    "metadata_source",
    "metadata_status",
    "lyrics_source",
    "lyrics_status",
    "genius_title",
    "genius_artist",
    "genius_url",
    "duration_seconds",
    "n_counted_sections",
    "n_transcript_words",
    "n_matches",
    "ordered_intervals",
    "chorus_seconds",
    "chorus_percent",
    "status",
    "error",
]


def read_rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8-sig") as handle:
        return list(csv.DictReader(handle))


def write_rows(path: Path, rows: Iterable[Mapping[str, Any]], fields: Sequence[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with tmp.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(fields), extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)
    tmp.replace(path)


def leading_file_id(path: str) -> str:
    match = re.match(r"^(\d+)_", Path(path or "").name)
    return str(int(match.group(1))) if match else ""


def strip_title_version(title: str) -> str:
    stripped = re.sub(r"\s*[\[(][^\])]*(?:remix|version|edit|live|demo|remaster|instrumental)[^\])]*[\])]\s*", " ", title or "", flags=re.I)
    return " ".join(stripped.split()) or title


def chunk_cached_block(block: str, chunk_tokens: int, chunk_step: int, min_tokens: int) -> list[str]:
    tokens = lyric_tokens(block)
    if len(tokens) < min_tokens:
        return []
    if len(tokens) <= chunk_tokens:
        return [" ".join(tokens)]
    starts = list(range(0, max(1, len(tokens) - chunk_tokens + 1), chunk_step))
    last_start = max(0, len(tokens) - chunk_tokens)
    if starts[-1] != last_start:
        starts.append(last_start)
    chunks: list[str] = []
    seen: set[str] = set()
    for start in starts:
        chunk = " ".join(tokens[start:start + chunk_tokens])
        if chunk and chunk not in seen:
            chunks.append(chunk)
            seen.add(chunk)
    return chunks


def cached_chorus_text_to_sections(text: str, chunk_tokens: int, chunk_step: int, min_tokens: int):
    blocks = [block.strip() for block in (text or "").split(" || ") if block.strip()]
    lines = [
        line
        for block in blocks
        for line in chunk_cached_block(block, chunk_tokens, chunk_step, min_tokens)
    ]
    if not lines:
        return []
    return parse_lyric_sections("[Chorus]\n" + "\n".join(lines))


def expand_intervals(
    intervals: Sequence[tuple[float, float]], pad_seconds: float, preview_end: float
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


def cache_key(path: Path, model_name: str) -> str:
    stat = path.stat()
    text = f"{path.resolve()}|{stat.st_size}|{int(stat.st_mtime)}|{model_name}"
    return hashlib.sha1(text.encode("utf-8")).hexdigest()


def transcript_cache_path(cache_dir: Path, audio_path: Path, model_name: str) -> Path:
    return cache_dir / f"{cache_key(audio_path, model_name)}.json"


def words_from_whisper_result(result: Mapping[str, Any]) -> list[TimedWord]:
    words: list[TimedWord] = []
    for segment in result.get("segments", []):
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
        json.dumps({"audio_path": str(audio_path), "model": model_name, "words": [asdict(word) for word in words]}, indent=2),
        encoding="utf-8",
    )
    return words


def audio_duration_seconds(audio_path: Path) -> float:
    from mutagen import File as MutagenFile

    audio = MutagenFile(str(audio_path))
    if audio is None or audio.info is None:
        raise ValueError("Could not read audio duration")
    return float(audio.info.length)


def genius_cache_path(cache_dir: Path, artist: str, title: str) -> Path:
    key = hashlib.sha1(f"{normalize(artist)}|{normalize(title)}".encode("utf-8")).hexdigest()
    return cache_dir / f"{key}.json"


def build_genius_client(token: str):
    import requests

    session = requests.Session()
    session.headers.update({
        "Authorization": f"Bearer {token}",
        # The official API accepts Bearer auth; the public lyric page is easier to
        # fetch with a browser-like UA than the lyricsgenius search route.
        "User-Agent": "Mozilla/5.0 SoundDiversityResearch/0.1",
    })
    return session


def genius_hit_score(hit: Mapping[str, Any], artist: str, title: str) -> float:
    result = hit.get("result", {})
    hit_title = normalize(result.get("title", ""))
    hit_artist = normalize((result.get("primary_artist") or {}).get("name", ""))
    wanted_title = normalize(strip_title_version(title))
    wanted_artist = normalize(artist)
    title_score = SequenceMatcher(None, wanted_title, hit_title, autojunk=False).ratio()
    artist_score = SequenceMatcher(None, wanted_artist, hit_artist, autojunk=False).ratio()
    if wanted_title and wanted_title in hit_title:
        title_score = max(title_score, 0.92)
    if wanted_artist and wanted_artist in hit_artist:
        artist_score = max(artist_score, 0.88)
    return 0.72 * title_score + 0.28 * artist_score


def extract_lyrics_from_genius_page(html: str) -> str:
    from bs4 import BeautifulSoup

    soup = BeautifulSoup(html, "html.parser")
    containers = soup.select('[data-lyrics-container="true"]')
    if containers:
        return "\n".join(container.get_text("\n") for container in containers).strip()
    return ""


def fetch_genius_lyrics(genius, artist: str, title: str, cache_dir: Path) -> dict[str, str]:
    cache_dir.mkdir(parents=True, exist_ok=True)
    cache_path = genius_cache_path(cache_dir, artist, title)
    if cache_path.exists():
        cached = json.loads(cache_path.read_text(encoding="utf-8"))
        cached_error = cached.get("error", "") or ""
        # Do not let auth / anti-bot fetch failures poison the cache forever.
        # A later run with a valid token or changed network conditions should retry.
        if cached.get("status") != "fetch_error" and "401 Client Error" not in cached_error and "403 Client Error" not in cached_error:
            return cached

    attempts = [(artist, title)]
    stripped_title = strip_title_version(title)
    if normalize(stripped_title) != normalize(title):
        attempts.append((artist, stripped_title))

    last_error = ""
    for attempt_artist, attempt_title in attempts:
        try:
            response = genius.get(
                "https://api.genius.com/search",
                params={"q": f"{attempt_title} {attempt_artist}"},
                timeout=20,
            )
            response.raise_for_status()
            hits = response.json().get("response", {}).get("hits", [])
        except Exception as exc:  # noqa: BLE001 - preserve row-level fetch failure
            last_error = str(exc)
            time.sleep(1)
            continue

        if not hits:
            last_error = "Official Genius API returned no hits"
            continue

        scored = sorted(
            ((genius_hit_score(hit, attempt_artist, attempt_title), hit) for hit in hits),
            key=lambda item: item[0],
            reverse=True,
        )
        best_score, best_hit = scored[0]
        result = best_hit.get("result", {})
        url = result.get("url", "")
        if best_score < 0.55 or not url:
            last_error = f"No confident Genius hit; best_score={best_score:.3f}"
            continue

        try:
            page = genius.get(url, timeout=20)
            page.raise_for_status()
            lyrics = extract_lyrics_from_genius_page(page.text)
        except Exception as exc:  # noqa: BLE001
            last_error = str(exc)
            time.sleep(1)
            continue

        if lyrics:
            payload = {
                "status": "ok",
                "artist": (result.get("primary_artist") or {}).get("name", "") or "",
                "title": result.get("title", "") or "",
                "url": url,
                "lyrics": lyrics,
                "query_artist": attempt_artist,
                "query_title": attempt_title,
                "match_score": f"{best_score:.4f}",
            }
            cache_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
            return payload
        last_error = "Genius page fetched but no lyric containers were found"

    is_fetch_error = "401 Client Error" in last_error or "403 Client Error" in last_error
    payload = {
        "status": "fetch_error" if is_fetch_error else "not_found",
        "artist": "",
        "title": "",
        "url": "",
        "lyrics": "",
        "error": last_error or "Genius search returned no lyrics",
    }
    # Cache real misses, but retry auth/anti-bot fetch failures on future runs.
    if not is_fetch_error:
        cache_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    return payload


def load_token_file(path: Path) -> str:
    """Read a Genius token file without logging the secret.

    Supports either a raw token file or an env-style line such as
    GENIUS_TOKEN=...
    """
    text = path.read_text(encoding="utf-8-sig").strip()
    for raw_line in text.splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        if "=" in line:
            key, value = line.split("=", 1)
            if key.strip().casefold() in {"genius_token", "genius_access_token", "client_access_token"}:
                return value.strip().strip("\"'")
        return line.strip().strip("\"'")
    return ""


def legacy_text_lookup(old_results: Sequence[Mapping[str, str]]) -> dict[str, str]:
    output: dict[str, str] = {}
    for row in old_results:
        song_id = leading_file_id(row.get("audio_path", ""))
        text = row.get("chorus_text_from_genius", "")
        if song_id and text:
            output[song_id] = text
    return output


def output_row(row: Mapping[str, str], **updates: Any) -> dict[str, Any]:
    return {
        "song_id": row.get("song_id", ""),
        "audio_path": row.get("output_path", "") or row.get("audio_path", ""),
        "artist": row.get("artist", ""),
        "title": row.get("title", ""),
        "metadata_source": row.get("artist_source", "") or row.get("metadata_source", ""),
        "metadata_status": row.get("metadata_status", ""),
        "lyrics_source": "",
        "lyrics_status": "",
        "genius_title": "",
        "genius_artist": "",
        "genius_url": "",
        "duration_seconds": "",
        "n_counted_sections": "0",
        "n_transcript_words": "0",
        "n_matches": "0",
        "ordered_intervals": "",
        "chorus_seconds": "",
        "chorus_percent": "",
        "status": "",
        "error": "",
        **updates,
    }


def eligible_manifest_rows(rows: Sequence[Mapping[str, str]]) -> list[Mapping[str, str]]:
    if "status" in rows[0]:
        return [row for row in rows if row.get("status") == "ok"]
    return list(rows)


def run(args: argparse.Namespace) -> list[dict[str, Any]]:
    if args.env_file:
        load_env_file(Path(args.env_file))

    manifest_rows = eligible_manifest_rows(read_rows(Path(args.manifest)))
    if args.limit:
        manifest_rows = manifest_rows[: args.limit]

    old_results = read_rows(Path(args.old_results)) if args.old_results else []
    legacy_text_by_id = legacy_text_lookup(old_results)

    token = args.genius_token or os.environ.get("GENIUS_TOKEN", "")
    if not token and args.genius_token_file:
        token = load_token_file(Path(args.genius_token_file))
    genius = None
    if args.lyrics_mode in {"genius", "genius_or_cached"}:
        if token:
            genius = build_genius_client(token)
        elif args.lyrics_mode == "genius":
            raise RuntimeError("GENIUS_TOKEN is not set and --genius-token was not provided")

    import whisper

    model = None
    existing_by_id: dict[str, dict[str, str]] = {}
    if args.resume and Path(args.output).exists():
        existing_rows = read_rows(Path(args.output))
        existing_by_id = {str(row.get("song_id", "")): row for row in existing_rows if row.get("song_id", "")}

    results: list[dict[str, Any]] = []
    processed_new = 0
    for index, row in enumerate(manifest_rows, start=1):
        artist = row.get("artist", "")
        title = row.get("title", "")
        audio_path = Path(row.get("output_path", "") or row.get("audio_path", ""))
        song_id = str(row.get("song_id", "") or leading_file_id(str(audio_path)))
        if args.resume and song_id in existing_by_id:
            results.append(existing_by_id[song_id])
            continue
        if args.max_new_rows is not None and processed_new >= args.max_new_rows:
            break
        print(f"[{index}/{len(manifest_rows)}] {artist} - {title}", flush=True)
        processed_new += 1

        try:
            if not is_usable_artist(artist):
                raise ValueError("metadata_unresolved_artist")
            if not title.strip():
                raise ValueError("metadata_unresolved_title")
            if not audio_path.exists():
                raise FileNotFoundError(str(audio_path))

            duration = audio_duration_seconds(audio_path)
            lyrics_source = ""
            lyrics_status = ""
            genius_title = genius_artist = genius_url = ""
            sections = []

            if genius is not None and args.lyrics_mode in {"genius", "genius_or_cached"}:
                payload = fetch_genius_lyrics(genius, artist, title, Path(args.lyrics_cache_dir))
                lyrics_status = payload.get("status", "")
                if lyrics_status == "ok":
                    lyrics_source = "genius"
                    genius_title = payload.get("title", "")
                    genius_artist = payload.get("artist", "")
                    genius_url = payload.get("url", "")
                    sections = parse_lyric_sections(payload.get("lyrics", ""))
                elif args.lyrics_mode == "genius":
                    raise ValueError(f"genius_lyrics_not_found: {payload.get('error', '')}")

            if not sections and args.lyrics_mode in {"cached", "genius_or_cached"}:
                cached = legacy_text_by_id.get(song_id, "")
                if cached:
                    lyrics_source = "legacy_cached_chorus_text"
                    lyrics_status = "ok"
                    sections = cached_chorus_text_to_sections(
                        cached,
                        chunk_tokens=args.chunk_tokens,
                        chunk_step=args.chunk_step,
                        min_tokens=args.min_chunk_tokens,
                    )

            counted_sections = [section for section in sections if section.counted_as_chorus]
            if not counted_sections:
                results.append(output_row(
                    row,
                    song_id=song_id,
                    lyrics_source=lyrics_source,
                    lyrics_status=lyrics_status or "unavailable",
                    duration_seconds=f"{duration:.4f}",
                    status="no_counted_sections",
                    error="No chorus/hook/refrain sections available for ordered matching",
                ))
                continue

            if model is None:
                model = whisper.load_model(args.model)
            words = load_or_transcribe(audio_path, model, args.model, Path(args.transcript_cache_dir))
            if not words:
                raise ValueError("Whisper produced no word timestamps")

            intervals, matches = detect_counted_sections(
                sections,
                words,
                min_score=args.min_score,
                min_query_tokens=args.min_query_tokens,
                max_iou=args.max_iou,
                gap_tolerance=args.gap_tolerance,
            )
            intervals = expand_intervals(intervals, args.pad_seconds, preview_end=min(duration, args.preview_seconds))
            chorus_seconds = interval_duration(intervals)
            chorus_percent = 100 * chorus_seconds / duration if duration > 0 else 0.0

            results.append(output_row(
                row,
                song_id=song_id,
                lyrics_source=lyrics_source,
                lyrics_status=lyrics_status,
                genius_title=genius_title,
                genius_artist=genius_artist,
                genius_url=genius_url,
                duration_seconds=f"{duration:.4f}",
                n_counted_sections=str(len(counted_sections)),
                n_transcript_words=str(len(words)),
                n_matches=str(len(matches)),
                ordered_intervals=format_intervals(intervals),
                chorus_seconds=f"{chorus_seconds:.4f}",
                chorus_percent=f"{chorus_percent:.4f}",
                status="ok",
                error="",
            ))
        except Exception as exc:  # noqa: BLE001 - batch runner records row-level errors
            results.append(output_row(row, song_id=song_id, status="error", error=str(exc)))

        if args.save_every and index % args.save_every == 0:
            write_rows(Path(args.output), results, RESULT_FIELDS)

    write_rows(Path(args.output), results, RESULT_FIELDS)
    return results


def parser() -> argparse.ArgumentParser:
    root = argparse.ArgumentParser(description="Run tuned ordered chorus detector on clean preview manifest.")
    root.add_argument("--manifest", default="data/chorus_preview_manifest_clean.csv")
    root.add_argument("--old-results", default="")
    root.add_argument("--output", default="outputs/chorus_rebuild/ordered_chorus_results_clean.csv")
    root.add_argument("--lyrics-mode", choices=["genius", "cached", "genius_or_cached"], default="genius_or_cached")
    root.add_argument("--env-file", help="Optional env file containing GENIUS_TOKEN")
    root.add_argument("--genius-token", help="Optional Genius token; prefer --env-file to avoid shell history")
    root.add_argument("--genius-token-file", help="Optional file containing raw Genius token or GENIUS_TOKEN=...")
    root.add_argument("--lyrics-cache-dir", default="data/cache/genius_lyrics")
    root.add_argument("--transcript-cache-dir", default="data/cache/chorus_transcripts")
    root.add_argument("--model", default="tiny")
    root.add_argument("--min-score", type=float, default=0.56)
    root.add_argument("--min-query-tokens", type=int, default=4)
    root.add_argument("--max-iou", type=float, default=0.50)
    root.add_argument("--gap-tolerance", type=float, default=0.75)
    root.add_argument("--pad-seconds", type=float, default=6.0)
    root.add_argument("--preview-seconds", type=float, default=30.0)
    root.add_argument("--chunk-tokens", type=int, default=8)
    root.add_argument("--chunk-step", type=int, default=4)
    root.add_argument("--min-chunk-tokens", type=int, default=4)
    root.add_argument("--limit", type=int)
    root.add_argument("--resume", action="store_true", help="Keep existing output rows and process only new manifest rows")
    root.add_argument("--max-new-rows", type=int, help="With --resume, process at most this many additional rows")
    root.add_argument("--save-every", type=int, default=5)
    return root


def main() -> None:
    args = parser().parse_args()
    rows = run(args)
    ok = sum(1 for row in rows if row["status"] == "ok")
    no_sections = sum(1 for row in rows if row["status"] == "no_counted_sections")
    errors = sum(1 for row in rows if row["status"] == "error")
    print(f"Wrote {args.output}")
    print(f"ok={ok} no_counted_sections={no_sections} error={errors}")


if __name__ == "__main__":
    main()
