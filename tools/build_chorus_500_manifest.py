from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any, Iterable, Mapping

import requests

ROOT = Path(__file__).resolve().parents[1]

import sys

if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from sound_diversity.core import is_usable_artist, normalize, read_csv, stable_seed, write_csv
from sound_diversity.preview_rebuild import download_preview, safe_filename_part


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
    "audio_source",
    "output_path",
    "status",
    "error",
]


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
            "chorus-500-preview",
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
    new_attempts = 0
    session = requests.Session()
    session.headers.update({"User-Agent": "SoundDiversityResearch/0.1"})

    for index, row in enumerate(candidates, 1):
        if ok_count >= args.target_ok:
            break
        apple_id = str(row.get("apple_track_id", "")).strip()
        if not apple_id:
            continue
        prior = existing.get(apple_id)
        if prior and prior.get("status") == "ok" and Path(prior.get("output_path", "")).exists():
            continue
        if args.max_new_attempts is not None and new_attempts >= args.max_new_attempts:
            break

        artist = row.get("artist", "")
        title = row.get("track_title", "") or row.get("apple_track", "")
        filename = f"{int(apple_id):012d}_{safe_filename_part(artist)}_-_{safe_filename_part(title)}.m4a"
        target = output_folder / filename
        new_attempts += 1
        try:
            download_preview(session, row["preview_url"], target)
            existing[apple_id] = output_row(
                row,
                audio_source="sampled_tracks_apple_preview_download",
                output_path=str(target.resolve()),
                status="ok",
                error="",
            )
            ok_count += 1
            print(f"[{ok_count}/{args.target_ok}] ok {artist} - {title}", flush=True)
        except Exception as exc:  # noqa: BLE001 - row-level collection should continue
            existing[apple_id] = output_row(
                row,
                output_path=str(target.resolve()),
                status="excluded",
                error=f"{type(exc).__name__}: {exc}",
            )
            print(f"[{index}/{len(candidates)}] excluded {artist} - {title}: {exc}", flush=True)

        write_csv(output_manifest, ordered_existing_rows(existing), MANIFEST_FIELDS)

    write_csv(output_manifest, ordered_existing_rows(existing), MANIFEST_FIELDS)
    return {
        "candidates": len(candidates),
        "manifest_rows": len(existing),
        "ok": sum(1 for row in existing.values() if row.get("status") == "ok"),
        "excluded": sum(1 for row in existing.values() if row.get("status") == "excluded"),
        "new_attempts": new_attempts,
    }


def parser() -> argparse.ArgumentParser:
    root = argparse.ArgumentParser(description="Build a clean 500-preview chorus manifest from sampled_tracks.csv.")
    root.add_argument("--sampled-tracks", default="data/sampled_tracks.csv")
    root.add_argument("--output-folder", default="outputs/chorus_500/previews_clean")
    root.add_argument("--output-manifest", default="data/chorus_preview_manifest_500_clean.csv")
    root.add_argument("--target-ok", type=int, default=500)
    root.add_argument("--max-new-attempts", type=int)
    root.add_argument("--seed", type=int, default=20260627)
    return root


def main() -> None:
    args = parser().parse_args()
    summary = build_manifest(args)
    print(summary)


if __name__ == "__main__":
    main()
