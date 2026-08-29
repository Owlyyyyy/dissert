from __future__ import annotations

import argparse
import os
import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from sound_diversity.cli import (
    APPLE_FIELDS,
    ATTEMPT_FIELDS,
    RELEASE_FIELDS,
    SAMPLE_FIELDS,
    TRACK_FIELDS,
    safe_print,
)
from sound_diversity.core import (
    ROOT,
    DiscogsClient,
    ITunesClient,
    JsonCache,
    Settings,
    discogs_artist_credit,
    is_discogs_release,
    is_usable_artist,
    load_env_file,
    normalize,
    read_csv,
    stable_seed,
    write_csv,
)


DATA = ROOT / "data"
LABELS_PATH = DATA / "benchmark_resolved_labels_draft.csv"
RELEASES_PATH = DATA / "benchmark_releases.csv"
TRACKS_PATH = DATA / "benchmark_track_candidates.csv"
SAMPLED_PATH = DATA / "benchmark_sampled_tracks.csv"
ATTEMPTS_PATH = DATA / "benchmark_itunes_attempts.csv"


def token_from_file(path: str | None) -> str:
    if path:
        env_path = Path(path)
        load_env_file(env_path)
        token = os.environ.get("DISCOGS_TOKEN")
        if not token:
            raw = env_path.read_text(encoding="utf-8-sig").strip()
            if raw and "=" not in raw and "\n" not in raw:
                token = raw
        if token:
            return token
    token = os.environ.get("DISCOGS_TOKEN", "")
    if not token:
        raise RuntimeError("Discogs token not found")
    return token


def discogs_client(args) -> tuple[Settings, DiscogsClient]:
    settings = Settings.load()
    token = token_from_file(args.env_file)
    return settings, DiscogsClient(
        token, settings.discogs_user_agent, JsonCache(DATA / "cache")
    )


def labels() -> list[dict[str, str]]:
    rows = read_csv(LABELS_PATH)
    unresolved = [row["label"] for row in rows if not row["discogs_label_id"].isdigit()]
    if unresolved:
        raise RuntimeError(f"Unresolved benchmark label IDs: {', '.join(unresolved)}")
    return rows


def collect_releases(args) -> None:
    settings, discogs = discogs_client(args)
    del settings
    requested = set(args.labels or [])
    rows = labels()
    if requested:
        rows = [row for row in rows if row["label"] in requested]
    existing = read_csv(RELEASES_PATH) if RELEASES_PATH.exists() else []
    existing = [
        row for row in existing
        if not requested or row["label"] not in requested
    ]
    seen = {(row["label"], row["year"], row["discogs_release_id"]) for row in existing}
    output = list(existing)

    for label in rows:
        for year in range(int(label["start_year"]), int(label["end_year"]) + 1):
            got = 0
            page = 1
            while got < args.max_releases_per_label_year and page <= args.max_pages:
                data = discogs.get(
                    "/database/search",
                    {
                        "label": label["label"],
                        "year": year,
                        "type": "release",
                        "per_page": 100,
                        "page": page,
                    },
                )
                results = data.get("results", [])
                if not results:
                    break
                for item in results:
                    release_id = item.get("id")
                    key = (label["label"], str(year), str(release_id))
                    if (
                        not release_id
                        or key in seen
                        or not is_discogs_release(item)
                        or str(item.get("year", "")) != str(year)
                    ):
                        continue
                    seen.add(key)
                    output.append({
                        "label": label["label"],
                        "discogs_label_id": label["discogs_label_id"],
                        "year": year,
                        "discogs_release_id": release_id,
                        "artist": item.get("title", "").split(" - ", 1)[0],
                        "release_title": item.get("title", ""),
                        "format": ",".join(item.get("format", []) or []),
                        "catno": "",
                        "resource_url": item.get("resource_url", ""),
                    })
                    got += 1
                    if got >= args.max_releases_per_label_year:
                        break
                pagination = data.get("pagination", {})
                if page >= int(pagination.get("pages", page)):
                    break
                page += 1
            safe_print(f"{label['label']} {year}: benchmark releases {got}")
            write_csv(RELEASES_PATH, output, RELEASE_FIELDS)


def discover_tracks(args) -> None:
    settings, discogs = discogs_client(args)
    releases = read_csv(RELEASES_PATH)
    requested = set(args.labels or [])
    if requested:
        releases = [row for row in releases if row["label"] in requested]
    existing = read_csv(TRACKS_PATH) if TRACKS_PATH.exists() else []
    existing = [
        row for row in existing
        if not requested or row["label"] not in requested
    ]
    completed = {(row["label"], row["discogs_release_id"]) for row in existing}
    output = list(existing)

    grouped: dict[tuple[str, int], list[dict[str, str]]] = {}
    for row in releases:
        grouped.setdefault((row["label"], int(row["year"])), []).append(row)

    for (label, year), group in sorted(grouped.items()):
        group = sorted(group, key=lambda row: int(row["discogs_release_id"]))
        random.Random(stable_seed(settings.seed, "benchmark", label, year, "releases")).shuffle(group)
        processed = 0
        for row in group:
            release_key = (label, row["discogs_release_id"])
            if release_key in completed:
                continue
            detail = discogs.release(int(row["discogs_release_id"]))
            if detail.get("_missing_resource") or detail.get("_failed_resource"):
                completed.add(release_key)
                continue
            release_artist = discogs_artist_credit(detail.get("artists")) or row["artist"]
            for track in detail.get("tracklist", []):
                title = (track.get("title") or "").strip()
                if not title or track.get("type_", "track") != "track":
                    continue
                track_artist = discogs_artist_credit(track.get("artists"))
                best_artist = track_artist or release_artist
                output.append({
                    "label": label,
                    "year": year,
                    "discogs_release_id": row["discogs_release_id"],
                    "release_artist": release_artist,
                    "track_artist": track_artist,
                    "artist": best_artist,
                    "artist_source": "track" if track_artist else "release",
                    "artist_usable": int(is_usable_artist(best_artist)),
                    "release_title": row["release_title"],
                    "track_title": title,
                    "position": track.get("position", ""),
                })
            completed.add(release_key)
            processed += 1
            if processed % 100 == 0:
                write_csv(TRACKS_PATH, output, TRACK_FIELDS)
        safe_print(f"{label} {year}: benchmark tracklists processed {processed}")
        write_csv(TRACKS_PATH, output, TRACK_FIELDS)


def match_year_sample(args) -> None:
    settings = Settings.load()
    itunes = ITunesClient(
        settings.itunes_country,
        settings.itunes_match_threshold,
        JsonCache(DATA / "cache"),
    )
    tracks = read_csv(TRACKS_PATH)
    sampled = read_csv(SAMPLED_PATH) if SAMPLED_PATH.exists() else []
    attempts = read_csv(ATTEMPTS_PATH) if ATTEMPTS_PATH.exists() else []

    sampled_by_year: dict[int, list[dict[str, str]]] = {}
    for row in sampled:
        sampled_by_year.setdefault(int(row["year"]), []).append(row)
    attempted_keys = {
        (int(row["year"]), int(row["sample_order"]))
        for row in attempts
        if row.get("sample_order")
    }

    grouped: dict[int, list[dict[str, str]]] = {}
    for row in tracks:
        if row.get("artist_usable") != "1" or not is_usable_artist(row.get("artist", "")):
            continue
        grouped.setdefault(int(row["year"]), []).append(row)

    for year in sorted(grouped):
        year_sampled = sampled_by_year.setdefault(year, [])
        used_apple_ids = {str(row["apple_track_id"]) for row in year_sampled}
        if len(year_sampled) >= args.tracks_per_year:
            safe_print(f"{year}: already matched {len(year_sampled)}/{args.tracks_per_year}")
            continue
        dedup = {}
        for row in grouped[year]:
            dedup.setdefault(
                (normalize(row["artist"]), normalize(row["track_title"])),
                row,
            )
        candidates = sorted(
            dedup.values(),
            key=lambda row: (
                normalize(row["artist"]),
                normalize(row["track_title"]),
                row["label"],
                int(row["discogs_release_id"]),
            ),
        )
        random.Random(stable_seed(settings.seed, "benchmark", year, "tracks")).shuffle(candidates)
        matched_before = len(year_sampled)
        for sample_order, row in enumerate(candidates, 1):
            if len(year_sampled) >= args.tracks_per_year:
                break
            key = (year, sample_order)
            if key in attempted_keys:
                continue
            match = itunes.search_track(row["artist"], row["track_title"])
            attempt = {
                **row,
                "sample_order": sample_order,
                "match_status": "matched" if match else itunes.last_lookup_status,
            }
            if not match:
                attempts.append(attempt)
                attempted_keys.add(key)
                continue
            apple = {
                "apple_track_id": match.get("trackId"),
                "apple_artist": match.get("artistName", ""),
                "apple_track": match.get("trackName", ""),
                "apple_collection": match.get("collectionName", ""),
                "apple_release_date": match.get("releaseDate", ""),
                "apple_track_url": match.get("trackViewUrl", ""),
                "preview_url": match.get("previewUrl", ""),
                "match_score": f"{match['matchScore']:.6f}",
            }
            if str(apple["apple_track_id"]) in used_apple_ids:
                attempts.append({**attempt, **apple, "match_status": "duplicate_match"})
                attempted_keys.add(key)
                continue
            result = {**row, "sample_order": sample_order, **apple}
            attempts.append({**attempt, **apple})
            sampled.append(result)
            year_sampled.append(result)
            used_apple_ids.add(str(apple["apple_track_id"]))
            attempted_keys.add(key)
            if len(year_sampled) % 25 == 0:
                write_csv(ATTEMPTS_PATH, attempts, ATTEMPT_FIELDS)
                write_csv(SAMPLED_PATH, sampled, SAMPLE_FIELDS)
                safe_print(f"{year}: matched {len(year_sampled)}/{args.tracks_per_year}")
        write_csv(ATTEMPTS_PATH, attempts, ATTEMPT_FIELDS)
        write_csv(SAMPLED_PATH, sampled, SAMPLE_FIELDS)
        safe_print(
            f"{year}: matched {matched_before}->{len(year_sampled)}/{args.tracks_per_year}"
        )


def parser() -> argparse.ArgumentParser:
    root = argparse.ArgumentParser(description="Benchmark-only broad-market sampling pipeline")
    sub = root.add_subparsers(dest="command", required=True)

    collect = sub.add_parser("collect-releases")
    collect.add_argument("--env-file", required=True)
    collect.add_argument("--label", dest="labels", action="append")
    collect.add_argument("--max-releases-per-label-year", type=int, default=20)
    collect.add_argument("--max-pages", type=int, default=3)
    collect.set_defaults(func=collect_releases)

    discover = sub.add_parser("discover-tracks")
    discover.add_argument("--env-file", required=True)
    discover.add_argument("--label", dest="labels", action="append")
    discover.set_defaults(func=discover_tracks)

    match = sub.add_parser("match-year-sample")
    match.add_argument("--tracks-per-year", type=int, default=250)
    match.set_defaults(func=match_year_sample)

    return root


def main() -> None:
    args = parser().parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
