from __future__ import annotations

import argparse
import sys
from difflib import SequenceMatcher
from pathlib import Path
from typing import Any, Iterable, Mapping

import requests

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from sound_diversity.chorus import parse_lyric_sections
from sound_diversity.core import is_usable_artist, normalize, read_csv, stable_seed, write_csv
from sound_diversity.preview_rebuild import download_preview, safe_filename_part

from run_ordered_chorus_detector import (
    build_genius_client,
    fetch_genius_lyrics,
    load_token_file,
    strip_title_version,
)


def safe_print(message: object) -> None:
    text = str(message)
    encoding = getattr(sys.stdout, "encoding", None) or "utf-8"
    sys.stdout.write(text.encode(encoding, errors="replace").decode(encoding, errors="replace") + "\n")
    sys.stdout.flush()


MANIFEST_FIELDS = [
    "song_id",
    "label",
    "year",
    "release_id",
    "track_position",
    "release_artist",
    "track_artist",
    "artist",
    "artist_source",
    "title",
    "metadata_status",
    "old_artist",
    "old_apple_artist",
    "old_apple_track",
    "apple_track_id",
    "apple_artist",
    "apple_track",
    "apple_collection",
    "preview_url",
    "match_score",
    "lyrics_status",
    "genius_title",
    "genius_artist",
    "genius_url",
    "genius_match_score",
    "n_genius_sections",
    "n_counted_sections",
    "counted_section_kinds",
    "audio_source",
    "output_path",
    "status",
    "error",
]


def text_similarity(left: str, right: str) -> float:
    left_norm = normalize(left)
    right_norm = normalize(right)
    if not left_norm or not right_norm:
        return 0.0
    score = SequenceMatcher(None, left_norm, right_norm, autojunk=False).ratio()
    if left_norm in right_norm or right_norm in left_norm:
        score = max(score, 0.90)
    return score


def credible_genius_match(
    artist: str,
    title: str,
    genius_artist: str,
    genius_title: str,
    min_artist_score: float,
    min_title_score: float,
) -> tuple[bool, str]:
    title_score = max(
        text_similarity(title, genius_title),
        text_similarity(strip_title_version(title), genius_title),
    )
    artist_score = text_similarity(artist, genius_artist)
    if title_score < min_title_score:
        return False, f"weak Genius title match ({title_score:.3f})"
    if artist_score < min_artist_score:
        return False, f"weak Genius artist match ({artist_score:.3f})"
    return True, ""


def unique_clean_candidates(rows: Iterable[Mapping[str, str]], seed: int) -> list[dict[str, str]]:
    by_apple_id: dict[str, dict[str, str]] = {}
    for row in rows:
        apple_id = str(row.get("apple_track_id", "")).strip()
        artist = str(row.get("artist", "")).strip()
        title = str(row.get("track_title", "") or row.get("apple_track", "")).strip()
        preview_url = str(row.get("preview_url", "")).strip()
        if not apple_id or not preview_url or not title:
            continue
        if row.get("artist_usable") != "1" or not is_usable_artist(artist):
            continue
        if normalize(artist) in {"various", "various artists"}:
            continue
        if apple_id not in by_apple_id:
            by_apple_id[apple_id] = dict(row)

    candidates = list(by_apple_id.values())
    candidates.sort(
        key=lambda row: stable_seed(
            seed,
            "chorus-500-genius-labeled",
            row.get("apple_track_id", ""),
            row.get("artist", ""),
            row.get("track_title", ""),
        )
    )
    return candidates


def existing_manifest(path: Path) -> dict[str, dict[str, str]]:
    if not path.exists():
        return {}
    return {str(row.get("song_id", "")): row for row in read_csv(path) if row.get("song_id", "")}


def ordered_existing_rows(existing: Mapping[str, dict[str, str]]) -> list[dict[str, str]]:
    def sort_key(row: Mapping[str, str]) -> tuple[int, str]:
        song_id = str(row.get("song_id", ""))
        return (0, f"{int(song_id):020d}") if song_id.isdigit() else (1, song_id)

    return sorted(existing.values(), key=sort_key)


def output_row(row: Mapping[str, str], **updates: Any) -> dict[str, Any]:
    result = {field: "" for field in MANIFEST_FIELDS}
    result.update(
        {
            "song_id": str(row.get("apple_track_id", "")).strip(),
            "label": row.get("label", ""),
            "year": row.get("year", ""),
            "release_id": row.get("discogs_release_id", ""),
            "track_position": row.get("position", ""),
            "release_artist": row.get("release_artist", ""),
            "track_artist": row.get("track_artist", ""),
            "artist": row.get("artist", ""),
            "artist_source": row.get("artist_source", ""),
            "title": row.get("track_title", "") or row.get("apple_track", ""),
            "metadata_status": f"sampled_tracks_{row.get('artist_source', 'artist')}_artist",
            "old_artist": row.get("old_artist", ""),
            "old_apple_artist": row.get("old_apple_artist", ""),
            "old_apple_track": row.get("old_apple_track", ""),
            "apple_track_id": row.get("apple_track_id", ""),
            "apple_artist": row.get("apple_artist", ""),
            "apple_track": row.get("apple_track", ""),
            "apple_collection": row.get("apple_collection", ""),
            "preview_url": row.get("preview_url", ""),
            "match_score": row.get("match_score", ""),
        }
    )
    result.update(updates)
    return result


def build_manifest(args: argparse.Namespace) -> dict[str, int]:
    token = args.genius_token
    if not token and args.genius_token_file:
        token = load_token_file(Path(args.genius_token_file))
    if not token:
        raise RuntimeError("A Genius token is required via --genius-token-file or --genius-token")

    genius = build_genius_client(token)
    download_session = requests.Session()
    download_session.headers.update({"User-Agent": "SoundDiversityResearch/0.1"})

    sampled_rows = read_csv(Path(args.sampled_tracks))
    candidates = unique_clean_candidates(sampled_rows, seed=args.seed)
    output_folder = Path(args.output_folder)
    output_manifest = Path(args.output_manifest)
    output_folder.mkdir(parents=True, exist_ok=True)

    existing = existing_manifest(output_manifest)
    ok_count = sum(
        1
        for row in existing.values()
        if row.get("status") == "ok" and Path(row.get("output_path", "")).exists()
    )
    new_checks = 0
    new_downloads = 0

    for index, row in enumerate(candidates, 1):
        if ok_count >= args.target_ok:
            break
        apple_id = str(row.get("apple_track_id", "")).strip()
        if not apple_id:
            continue

        prior = existing.get(apple_id)
        if prior and prior.get("status") == "ok" and Path(prior.get("output_path", "")).exists():
            continue
        if prior and prior.get("status") == "excluded" and not args.recheck_excluded:
            continue
        if args.max_new_checks is not None and new_checks >= args.max_new_checks:
            break

        artist = row.get("artist", "")
        title = row.get("track_title", "") or row.get("apple_track", "")
        new_checks += 1

        try:
            payload = fetch_genius_lyrics(genius, artist, title, Path(args.lyrics_cache_dir))
            lyrics_status = payload.get("status", "") or "unavailable"
            if lyrics_status != "ok":
                existing[apple_id] = output_row(
                    row,
                    lyrics_status=lyrics_status,
                    status="excluded",
                    error=f"genius_lyrics_not_found: {payload.get('error', '')}",
                )
                safe_print(f"[{ok_count}/{args.target_ok}] excluded no lyrics: {artist} - {title}")
                continue

            genius_title = payload.get("title", "")
            genius_artist = payload.get("artist", "")
            is_match, match_error = credible_genius_match(
                artist,
                title,
                genius_artist,
                genius_title,
                min_artist_score=args.min_genius_artist_score,
                min_title_score=args.min_genius_title_score,
            )
            if not is_match:
                existing[apple_id] = output_row(
                    row,
                    lyrics_status=lyrics_status,
                    genius_title=genius_title,
                    genius_artist=genius_artist,
                    genius_url=payload.get("url", ""),
                    genius_match_score=payload.get("match_score", ""),
                    status="excluded",
                    error=f"genius_match_rejected: {match_error}",
                )
                safe_print(f"[{ok_count}/{args.target_ok}] excluded weak Genius match: {artist} - {title}")
                continue

            sections = parse_lyric_sections(payload.get("lyrics", ""))
            counted_sections = [section for section in sections if section.counted_as_chorus and section.lines]
            if not counted_sections:
                existing[apple_id] = output_row(
                    row,
                    lyrics_status=lyrics_status,
                    genius_title=genius_title,
                    genius_artist=genius_artist,
                    genius_url=payload.get("url", ""),
                    genius_match_score=payload.get("match_score", ""),
                    n_genius_sections=str(len(sections)),
                    n_counted_sections="0",
                    status="excluded",
                    error="no_chorus_like_genius_sections",
                )
                safe_print(f"[{ok_count}/{args.target_ok}] excluded no chorus label: {artist} - {title}")
                continue

            filename = f"{int(apple_id):012d}_{safe_filename_part(artist)}_-_{safe_filename_part(title)}.m4a"
            target = output_folder / filename
            download_preview(download_session, row["preview_url"], target)
            new_downloads += 1
            ok_count += 1
            existing[apple_id] = output_row(
                row,
                lyrics_status=lyrics_status,
                genius_title=genius_title,
                genius_artist=genius_artist,
                genius_url=payload.get("url", ""),
                genius_match_score=payload.get("match_score", ""),
                n_genius_sections=str(len(sections)),
                n_counted_sections=str(len(counted_sections)),
                counted_section_kinds=";".join(sorted({section.kind for section in counted_sections})),
                audio_source="sampled_tracks_apple_preview_download",
                output_path=str(target.resolve()),
                status="ok",
                error="",
            )
            safe_print(f"[{ok_count}/{args.target_ok}] ok {artist} - {title}")
        except Exception as exc:  # noqa: BLE001 - row-level collection should continue
            existing[apple_id] = output_row(
                row,
                status="excluded",
                error=f"{type(exc).__name__}: {exc}",
            )
            safe_print(f"[{ok_count}/{args.target_ok}] excluded error: {artist} - {title}: {exc}")

        write_csv(output_manifest, ordered_existing_rows(existing), MANIFEST_FIELDS)

    write_csv(output_manifest, ordered_existing_rows(existing), MANIFEST_FIELDS)
    return {
        "candidates": len(candidates),
        "manifest_rows": len(existing),
        "ok": sum(1 for row in existing.values() if row.get("status") == "ok"),
        "excluded": sum(1 for row in existing.values() if row.get("status") == "excluded"),
        "new_checks": new_checks,
        "new_downloads": new_downloads,
    }


def parser() -> argparse.ArgumentParser:
    root = argparse.ArgumentParser(description="Build a 500-preview manifest pre-filtered to songs with usable Genius chorus-like sections.")
    root.add_argument("--sampled-tracks", default="data/sampled_tracks.csv")
    root.add_argument("--output-folder", default="outputs/chorus_500_genius_labeled/previews_clean")
    root.add_argument("--output-manifest", default="data/chorus_preview_manifest_500_genius_labeled.csv")
    root.add_argument("--target-ok", type=int, default=500)
    root.add_argument("--max-new-checks", type=int)
    root.add_argument("--seed", type=int, default=20260627)
    root.add_argument("--lyrics-cache-dir", default="data/cache/genius_lyrics")
    root.add_argument("--genius-token", help="Optional Genius token; prefer --genius-token-file to avoid shell history")
    root.add_argument("--genius-token-file", help="Optional file containing raw Genius token or GENIUS_TOKEN=...")
    root.add_argument("--min-genius-artist-score", type=float, default=0.58)
    root.add_argument("--min-genius-title-score", type=float, default=0.70)
    root.add_argument("--recheck-excluded", action="store_true")
    return root


def main() -> None:
    args = parser().parse_args()
    summary = build_manifest(args)
    safe_print(summary)


if __name__ == "__main__":
    main()
