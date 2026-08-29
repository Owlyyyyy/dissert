from __future__ import annotations

import os
import re
import shutil
import time
from pathlib import Path
from typing import Any

import requests

from .core import (
    ROOT, DiscogsClient, ITunesClient, JsonCache, Settings, discogs_artist_credit,
    is_usable_artist, load_env_file, normalize, read_csv, write_csv,
)


MANIFEST_FIELDS = [
    "song_id", "label", "year", "release_id", "track_position",
    "release_artist", "track_artist", "artist", "artist_source", "title",
    "old_artist", "old_apple_artist", "old_apple_track", "apple_track_id",
    "apple_artist", "apple_track", "preview_url", "match_score",
    "audio_source", "output_path", "status", "error",
]


def safe_filename_part(value: str, maximum: int = 90) -> str:
    value = re.sub(r'[<>:"/\\|?*\x00-\x1f]', " ", value or "")
    value = " ".join(value.split()).strip(" .")
    return (value or "Unknown")[:maximum].rstrip(" .")


def find_release_track(detail: dict[str, Any], position: str, title: str) -> dict[str, Any] | None:
    tracks = [item for item in detail.get("tracklist", []) if item.get("type_", "track") == "track"]
    exact_position = [item for item in tracks if str(item.get("position", "")).strip() == position.strip()]
    if len(exact_position) == 1:
        return exact_position[0]
    normalized_title = normalize(title)
    title_matches = [item for item in exact_position or tracks if normalize(item.get("title", "")) == normalized_title]
    return title_matches[0] if len(title_matches) == 1 else None


def corrected_identity(detail: dict[str, Any], position: str, old_title: str) -> tuple[str, str, str, str, str]:
    track = find_release_track(detail, position, old_title)
    if track is None:
        raise ValueError("track_not_found_in_release")
    release_artist = discogs_artist_credit(detail.get("artists"))
    track_artist = discogs_artist_credit(track.get("artists"))
    if is_usable_artist(track_artist):
        artist, source = track_artist, "track"
    elif is_usable_artist(release_artist):
        artist, source = release_artist, "release"
    else:
        raise ValueError("metadata_unresolved_no_usable_track_or_release_artist")
    title = (track.get("title") or old_title).strip()
    if not title:
        raise ValueError("metadata_unresolved_missing_title")
    return artist, title, source, release_artist, track_artist


def apple_match_is_valid(client: ITunesClient, artist: str, title: str, apple_artist: str, apple_title: str) -> tuple[bool, float]:
    artist_score = client.similarity(artist, apple_artist)
    title_score = client.similarity(title, apple_title)
    score = 0.45 * artist_score + 0.55 * title_score
    valid = artist_score >= 0.70 and title_score >= 0.70 and score >= client.threshold
    return valid, score


def download_preview(session: requests.Session, url: str, target: Path) -> None:
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.with_suffix(target.suffix + ".part")
    for attempt in range(5):
        try:
            response = session.get(url, timeout=45)
            if response.status_code >= 500:
                raise requests.HTTPError(f"HTTP {response.status_code}")
            response.raise_for_status()
            temporary.write_bytes(response.content)
            temporary.replace(target)
            return
        except requests.RequestException:
            if attempt == 4:
                raise
            time.sleep(min(2 ** (attempt + 1), 20))


def rebuild_preview_folder(
    source_manifest: Path,
    old_preview_folder: Path,
    output_folder: Path,
    output_manifest: Path,
    env_file: Path,
    limit: int | None = None,
) -> None:
    load_env_file(env_file)
    token = os.environ.get("DISCOGS_TOKEN", "")
    if not token:
        raise ValueError("DISCOGS_TOKEN is missing")
    settings = Settings.load()
    cache = JsonCache(ROOT / "data/cache")
    discogs = DiscogsClient(token, settings.discogs_user_agent, cache)
    itunes = ITunesClient(settings.itunes_country, settings.itunes_match_threshold, cache)
    session = requests.Session()
    session.headers.update({"User-Agent": "SoundDiversityResearch/0.1"})

    available_names = {path.name for path in old_preview_folder.iterdir() if path.is_file()}
    source_rows = [
        row for row in read_csv(source_manifest)
        if Path(row.get("preview_path", "")).name in available_names
    ]
    source_rows.sort(key=lambda row: int(row["song_id"]))
    if limit is not None:
        source_rows = source_rows[:limit]

    existing = {row["song_id"]: row for row in read_csv(output_manifest)} if output_manifest.exists() else {}
    release_cache: dict[str, dict[str, Any]] = {}
    output_folder.mkdir(parents=True, exist_ok=True)

    for index, row in enumerate(source_rows, 1):
        prior = existing.get(row["song_id"])
        if prior and prior.get("status") == "ok" and Path(prior.get("output_path", "")).exists():
            print(f"[{index}/{len(source_rows)}] cached {prior['artist']} - {prior['title']}")
            continue
        result = {field: "" for field in MANIFEST_FIELDS}
        result.update({
            "song_id": row["song_id"], "label": row.get("label", ""),
            "year": row.get("year", ""), "release_id": row.get("release_id", ""),
            "track_position": row.get("track_position", ""),
            "old_artist": row.get("artist", ""),
            "old_apple_artist": row.get("itunes_artist_name", ""),
            "old_apple_track": row.get("itunes_track_name", ""),
        })
        try:
            release_id = row["release_id"]
            if release_id not in release_cache:
                release_cache[release_id] = discogs.release(int(release_id))
            artist, title, artist_source, release_artist, track_artist = corrected_identity(
                release_cache[release_id], row.get("track_position", ""), row.get("song", "")
            )
            result.update({
                "release_artist": release_artist, "track_artist": track_artist,
                "artist": artist, "artist_source": artist_source, "title": title,
            })
            if normalize(artist) in {"various", "various artists"}:
                raise ValueError("resolved_artist_is_various")

            valid_old, old_score = apple_match_is_valid(
                itunes, artist, title, row.get("itunes_artist_name", ""), row.get("itunes_track_name", "")
            )
            match: dict[str, Any]
            old_path = old_preview_folder / Path(row.get("preview_path", "")).name
            if valid_old and old_path.exists() and row.get("preview_url"):
                match = {
                    "trackId": row.get("itunes_track_id", ""),
                    "artistName": row.get("itunes_artist_name", ""),
                    "trackName": row.get("itunes_track_name", ""),
                    "previewUrl": row.get("preview_url", ""),
                    "matchScore": old_score,
                }
                audio_source = "verified_legacy_copy"
            else:
                found = itunes.search_track(artist, title, retry_cached_403=True)
                if found is None:
                    raise ValueError(f"apple_{itunes.last_lookup_status}")
                match = found
                audio_source = "fresh_apple_download"

            filename = f"{int(row['song_id']):06d}_{safe_filename_part(artist)}_-_{safe_filename_part(title)}.m4a"
            target = output_folder / filename
            if audio_source == "verified_legacy_copy":
                shutil.copy2(old_path, target)
            else:
                download_preview(session, match["previewUrl"], target)
            result.update({
                "apple_track_id": match.get("trackId", ""),
                "apple_artist": match.get("artistName", ""),
                "apple_track": match.get("trackName", ""),
                "preview_url": match.get("previewUrl", ""),
                "match_score": f"{float(match['matchScore']):.6f}",
                "audio_source": audio_source, "output_path": str(target.resolve()),
                "status": "ok",
            })
            print(f"[{index}/{len(source_rows)}] {audio_source}: {artist} - {title}")
        except Exception as exc:
            result["status"] = "excluded"
            result["error"] = f"{type(exc).__name__}: {exc}"
            print(f"[{index}/{len(source_rows)}] EXCLUDED {row.get('artist')} - {row.get('song')}: {exc}")
        existing[row["song_id"]] = result
        ordered = [existing[key] for key in sorted(existing, key=int)]
        write_csv(output_manifest, ordered, MANIFEST_FIELDS)

