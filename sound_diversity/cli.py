from __future__ import annotations

import argparse
import json
import os
import random
import subprocess
import tempfile
from pathlib import Path

import numpy as np
import requests

from .core import (
    PANEL_FIELDS, ROOT, DiscogsClient, ITunesClient, JsonCache, Settings,
    discogs_artist_credit, is_discogs_release, is_usable_artist, load_env_file,
    mean_pairwise_diversity, normalize, panel_skeleton, read_csv, stable_seed,
    write_csv,
)


DATA = ROOT / "data"
EMBEDDING_DIR = DATA / "embeddings_full30"

TRACK_FIELDS = ["label", "year", "discogs_release_id", "release_artist", "track_artist", "artist", "artist_source", "artist_usable", "release_title", "track_title", "position"]
RELEASE_FIELDS = ["label", "discogs_label_id", "year", "discogs_release_id", "artist", "release_title", "format", "catno", "resource_url"]
APPLE_FIELDS = ["apple_track_id", "apple_artist", "apple_track", "apple_collection", "apple_release_date", "apple_track_url", "preview_url", "match_score"]
SAMPLE_FIELDS = TRACK_FIELDS + ["sample_order"] + APPLE_FIELDS
ATTEMPT_FIELDS = TRACK_FIELDS + ["sample_order", "match_status"] + APPLE_FIELDS


def safe_print(message: str) -> None:
    """Print progress without turning unsupported console characters into failures."""
    try:
        print(message)
    except UnicodeEncodeError:
        print(message.encode("ascii", errors="backslashreplace").decode("ascii"))


def decode_preview_audio(path: Path, sample_rate: int) -> np.ndarray:
    """Decode a preview, using explicit FFmpeg conversion for M4A edge cases."""
    import librosa

    try:
        audio, _ = librosa.load(str(path), sr=sample_rate, mono=True)
    except Exception:
        audio = np.empty(0, dtype=np.float32)
    if len(audio):
        return audio

    wav_path = None
    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as handle:
        wav_path = Path(handle.name)
    try:
        subprocess.run(
            [
                "ffmpeg", "-v", "error", "-y", "-i", str(path),
                "-ac", "1", "-ar", str(sample_rate), str(wav_path),
            ],
            check=True,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.PIPE,
        )
        audio, _ = librosa.load(str(wav_path), sr=sample_rate, mono=True)
    finally:
        if wav_path is not None:
            wav_path.unlink(missing_ok=True)
    if not len(audio):
        raise ValueError("decoded preview contains no audio samples")
    return audio


def clients(args):
    if args.env_file:
        load_env_file(Path(args.env_file))
    settings = Settings.load()
    cache = JsonCache(DATA / "cache")
    discogs = DiscogsClient(os.environ.get("DISCOGS_TOKEN", ""), settings.discogs_user_agent, cache)
    return settings, cache, discogs


def resolve_labels(args) -> None:
    _, _, discogs = clients(args)
    labels = read_csv(ROOT / "configs/labels.csv")
    candidates = []
    resolved = []
    for label in labels:
        if label["discogs_label_id"].isdigit():
            resolved.append(label)
            print(f"{label['label']}: {label['discogs_label_id']} (configured)")
            continue
        results = discogs.search_labels(label["label"])
        exact = [r for r in results if normalize(r.get("title", "")) == normalize(label["label"])]
        for rank, item in enumerate(results[:10], 1):
            candidates.append({"requested_label": label["label"], "rank": rank, "discogs_label_id": item.get("id", ""), "discogs_title": item.get("title", ""), "resource_url": item.get("resource_url", ""), "exact_normalized_match": int(item in exact)})
        chosen = exact[0] if len(exact) == 1 and results and exact[0] == results[0] else None
        resolved.append({**label, "discogs_label_id": chosen.get("id", "REVIEW") if chosen else "REVIEW"})
        print(f"{label['label']}: {resolved[-1]['discogs_label_id']}")
    write_csv(DATA / "label_resolution_candidates.csv", candidates, ["requested_label", "rank", "discogs_label_id", "discogs_title", "resource_url", "exact_normalized_match"])
    write_csv(DATA / "resolved_labels.csv", resolved, list(labels[0]))


def collect_releases(args) -> None:
    _, _, discogs = clients(args)
    labels = read_csv(DATA / "resolved_labels.csv")
    valid_labels = {row["label"] for row in labels}
    requested = set(args.labels or [])
    if requested:
        unknown = requested - valid_labels
        if unknown:
            raise ValueError(f"Unknown labels: {', '.join(sorted(unknown))}")
        labels = [row for row in labels if row["label"] in requested]
        existing_path = DATA / "releases.csv"
        output = [
            row for row in (read_csv(existing_path) if existing_path.exists() else [])
            if row["label"] in valid_labels and row["label"] not in requested
        ]
    else:
        output = []
    for label in labels:
        if not label["discogs_label_id"].isdigit():
            print(f"Skipping unresolved label: {label['label']}")
            continue
        start, end = int(label["start_year"]), int(label["end_year"])
        releases = discogs.label_releases(int(label["discogs_label_id"]))
        seen = set()
        for item in releases:
            year = int(item.get("year") or 0)
            release_id = item.get("id")
            if not is_discogs_release(item) or not release_id or not start <= year <= end or release_id in seen:
                continue
            seen.add(release_id)
            output.append({"label": label["label"], "discogs_label_id": label["discogs_label_id"], "year": year, "discogs_release_id": release_id, "artist": item.get("artist", ""), "release_title": item.get("title", ""), "format": item.get("format", ""), "catno": item.get("catno", ""), "resource_url": item.get("resource_url", "")})
        print(f"{label['label']}: {len(seen)} releases in window")
        write_csv(DATA / "releases.csv", output, RELEASE_FIELDS)


def audit_label_ids(args) -> None:
    _, _, discogs = clients(args)
    for label_id in args.ids:
        profile = discogs.get(f"/labels/{label_id}")
        first_page = discogs.get(f"/labels/{label_id}/releases", {"page": 1, "per_page": 1})
        total = first_page.get("pagination", {}).get("items", "?")
        summary = " ".join((profile.get("profile") or "").split())[:240]
        print(f"{label_id}\t{profile.get('name', '')}\t{total} releases\t{summary}")


def discover_tracks(args) -> None:
    settings, _, discogs = clients(args)
    releases = read_csv(DATA / "releases.csv")
    valid_labels = {row["label"] for row in read_csv(DATA / "resolved_labels.csv")}
    requested = set(args.labels or [])
    if requested:
        unknown = requested - valid_labels
        if unknown:
            raise ValueError(f"Unknown labels: {', '.join(sorted(unknown))}")
        releases = [row for row in releases if row["label"] in requested]
    existing_path = DATA / "track_candidates.csv"
    output = read_csv(existing_path) if existing_path.exists() else []
    output = [
        row for row in output
        if row["label"] in valid_labels and (not requested or row["label"] not in requested)
    ]
    completed = {(row["label"], row["discogs_release_id"]) for row in output}
    processed_since_start = 0
    grouped = {}
    for row in releases:
        grouped.setdefault((row["label"], int(row["year"])), []).append(row)
    for (label, year), group in sorted(grouped.items()):
        rng = random.Random(stable_seed(settings.seed, label, year, "releases"))
        group = sorted(group, key=lambda r: int(r["discogs_release_id"]))
        rng.shuffle(group)
        for row in group:
            release_id = row["discogs_release_id"]
            release_key = (label, release_id)
            if release_key in completed:
                continue
            detail = discogs.release(int(release_id))
            release_artist = discogs_artist_credit(detail.get("artists")) or row["artist"]
            for track in detail.get("tracklist", []):
                title = (track.get("title") or "").strip()
                if not title or track.get("type_", "track") != "track":
                    continue
                track_artist = discogs_artist_credit(track.get("artists"))
                best_artist = track_artist or release_artist
                artist_source = "track" if track_artist else "release"
                output.append({
                    "label": label, "year": year,
                    "discogs_release_id": release_id,
                    "release_artist": release_artist,
                    "track_artist": track_artist,
                    "artist": best_artist,
                    "artist_source": artist_source,
                    "artist_usable": int(is_usable_artist(best_artist)),
                    "release_title": row["release_title"],
                    "track_title": title,
                    "position": track.get("position", ""),
                })
            completed.add(release_key)
            processed_since_start += 1
            if processed_since_start % 250 == 0:
                write_csv(existing_path, output, TRACK_FIELDS)
        write_csv(existing_path, output, TRACK_FIELDS)
        print(f"{label} {year}: tracklists checkpointed")


def match_and_sample(args) -> None:
    settings = Settings.load()
    cache = JsonCache(DATA / "cache")
    itunes = ITunesClient(settings.itunes_country, settings.itunes_match_threshold, cache)
    tracks = read_csv(DATA / "track_candidates.csv")
    valid_labels = {row["label"] for row in read_csv(DATA / "resolved_labels.csv")}
    requested = set(args.labels or [])
    if requested:
        unknown = requested - valid_labels
        if unknown:
            raise ValueError(f"Unknown labels: {', '.join(sorted(unknown))}")
    grouped = {}
    for row in tracks:
        if requested and row["label"] not in requested:
            continue
        if row.get("artist_usable") != "1" or not is_usable_artist(row.get("artist", "")):
            continue
        grouped.setdefault((row["label"], int(row["year"])), []).append(row)
    if requested:
        selected = [
            row for row in read_csv(DATA / "sampled_tracks.csv")
            if row["label"] in valid_labels and row["label"] not in requested
        ]
        attempts = [
            row for row in read_csv(DATA / "itunes_attempts.csv")
            if row["label"] in valid_labels and row["label"] not in requested
        ]
    else:
        selected = []
        attempts = []
    for (label, year), group in sorted(grouped.items()):
        dedup = {}
        for row in group:
            dedup.setdefault((normalize(row["artist"]), normalize(row["track_title"])), row)
        candidates = sorted(dedup.values(), key=lambda r: (normalize(r["artist"]), normalize(r["track_title"]), int(r["discogs_release_id"])))
        random.Random(stable_seed(settings.seed, label, year, "tracks")).shuffle(candidates)
        matched = 0
        attempted = 0
        used_apple_ids = set()
        for sample_order, row in enumerate(candidates, 1):
            if matched >= settings.sample_size:
                break
            attempted += 1
            match = itunes.search_track(row["artist"], row["track_title"])
            attempt = {**row, "sample_order": sample_order, "match_status": "matched" if match else itunes.last_lookup_status}
            if not match:
                attempts.append(attempt)
                continue
            apple = {"apple_track_id": match.get("trackId"), "apple_artist": match.get("artistName", ""), "apple_track": match.get("trackName", ""), "apple_collection": match.get("collectionName", ""), "apple_release_date": match.get("releaseDate", ""), "apple_track_url": match.get("trackViewUrl", ""), "preview_url": match.get("previewUrl", ""), "match_score": f"{match['matchScore']:.6f}"}
            if apple["apple_track_id"] in used_apple_ids:
                attempts.append({**attempt, **apple, "match_status": "duplicate_match"})
                continue
            used_apple_ids.add(apple["apple_track_id"])
            attempts.append({**attempt, **apple})
            selected.append({**row, "sample_order": sample_order, **apple})
            matched += 1
        print(f"{label} {year}: attempted {attempted}, matched {matched}/{settings.sample_size} Apple previews")
        write_csv(DATA / "itunes_attempts.csv", attempts, ATTEMPT_FIELDS)
        write_csv(DATA / "sampled_tracks.csv", selected, SAMPLE_FIELDS)


def retry_cached_403(args) -> None:
    """Retry cached Apple 403s only for label-years that remain below target."""
    settings = Settings.load()
    cache = JsonCache(DATA / "cache")
    itunes = ITunesClient(settings.itunes_country, settings.itunes_match_threshold, cache)
    tracks = read_csv(DATA / "track_candidates.csv")
    valid_labels = {row["label"] for row in read_csv(DATA / "resolved_labels.csv")}
    requested = set(args.labels or [])
    if requested:
        unknown = requested - valid_labels
        if unknown:
            raise ValueError(f"Unknown labels: {', '.join(sorted(unknown))}")
    grouped = {}
    for row in tracks:
        if requested and row["label"] not in requested:
            continue
        if row.get("artist_usable") != "1" or not is_usable_artist(row.get("artist", "")):
            continue
        grouped.setdefault((row["label"], int(row["year"])), []).append(row)

    selected = [
        row for row in read_csv(DATA / "sampled_tracks.csv")
        if row["label"] in valid_labels
    ]
    selected_by_group = {}
    for row in selected:
        selected_by_group.setdefault((row["label"], int(row["year"])), []).append(row)
    attempts = [
        row for row in read_csv(DATA / "itunes_attempts.csv")
        if row["label"] in valid_labels
    ]
    attempt_map = {
        (row["label"], int(row["year"]), int(row["sample_order"])): row
        for row in attempts
    }

    retried_total = 0
    added_total = 0
    for (label, year), group in sorted(grouped.items()):
        group_selected = selected_by_group.setdefault((label, year), [])
        if len(group_selected) >= settings.sample_size:
            continue
        dedup = {}
        for row in group:
            dedup.setdefault((normalize(row["artist"]), normalize(row["track_title"])), row)
        candidates = sorted(
            dedup.values(),
            key=lambda row: (
                normalize(row["artist"]), normalize(row["track_title"]),
                int(row["discogs_release_id"]),
            ),
        )
        random.Random(stable_seed(settings.seed, label, year, "tracks")).shuffle(candidates)
        used_apple_ids = {str(row["apple_track_id"]) for row in group_selected}
        before = len(group_selected)
        retried = 0
        for sample_order, row in enumerate(candidates, 1):
            if len(group_selected) >= settings.sample_size:
                break
            key = (label, year, sample_order)
            prior = attempt_map.get(key)
            if prior is None or prior.get("match_status") != "http_403":
                continue
            retried += 1
            retried_total += 1
            match = itunes.search_track(
                row["artist"], row["track_title"], retry_cached_403=True
            )
            attempt = {
                **row,
                "sample_order": sample_order,
                "match_status": "matched" if match else itunes.last_lookup_status,
            }
            if not match:
                attempt_map[key] = attempt
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
            apple_id = str(apple["apple_track_id"])
            if apple_id in used_apple_ids:
                attempt_map[key] = {
                    **attempt, **apple, "match_status": "duplicate_match"
                }
                continue
            used_apple_ids.add(apple_id)
            attempt_map[key] = {**attempt, **apple}
            result = {**row, "sample_order": sample_order, **apple}
            selected.append(result)
            group_selected.append(result)
            added_total += 1
        safe_print(
            f"{label} {year}: retried {retried} cached 403s, "
            f"matched {before}->{len(group_selected)}/{settings.sample_size}"
        )

    selected.sort(
        key=lambda row: (row["label"], int(row["year"]), int(row["sample_order"]))
    )
    attempts = sorted(
        attempt_map.values(),
        key=lambda row: (row["label"], int(row["year"]), int(row["sample_order"])),
    )
    write_csv(DATA / "itunes_attempts.csv", attempts, ATTEMPT_FIELDS)
    write_csv(DATA / "sampled_tracks.csv", selected, SAMPLE_FIELDS)
    safe_print(
        f"Retry complete: retried {retried_total} cached 403s and added {added_total} matches"
    )


def prune_removed_labels(args) -> None:
    """Remove labels absent from the current registry from every derived CSV."""
    labels = read_csv(DATA / "resolved_labels.csv")
    valid_labels = {row["label"] for row in labels}
    specs = [
        (DATA / "releases.csv", RELEASE_FIELDS),
        (DATA / "track_candidates.csv", TRACK_FIELDS),
        (DATA / "sampled_tracks.csv", SAMPLE_FIELDS),
        (DATA / "itunes_attempts.csv", ATTEMPT_FIELDS),
        (DATA / "label_year_song_data.csv", SAMPLE_FIELDS),
    ]
    for path, fields in specs:
        if not path.exists():
            continue
        rows = read_csv(path)
        kept = [row for row in rows if row.get("label") in valid_labels]
        write_csv(path, kept, fields)
        safe_print(f"{path.name}: removed {len(rows) - len(kept)}, kept {len(kept)}")


def embed(args) -> None:
    import librosa
    import torch
    from transformers import ClapModel, ClapProcessor

    settings = Settings.load()
    samples_path = Path(args.samples) if getattr(args, "samples", None) else DATA / "sampled_tracks.csv"
    rows = read_csv(samples_path)
    if args.limit is not None:
        rows = rows[:args.limit]
    out_dir = EMBEDDING_DIR
    out_dir.mkdir(parents=True, exist_ok=True)
    sample_stem = samples_path.stem
    if samples_path.name == "sampled_tracks.csv":
        failure_path = DATA / "embedding_full30_failures.csv"
        metadata_path = DATA / "embedding_full30_metadata.csv"
    else:
        failure_path = DATA / f"embedding_full30_failures_{sample_stem}.csv"
        metadata_path = DATA / f"embedding_full30_metadata_{sample_stem}.csv"
    failure_map = {
        row["apple_track_id"]: row
        for row in (read_csv(failure_path) if failure_path.exists() else [])
        if not (out_dir / f"{row['apple_track_id']}.npy").exists()
    }
    failure_fields = ["apple_track_id", "label", "year", "artist", "track_title", "error_type", "error"]
    if failure_path.exists():
        write_csv(failure_path, failure_map.values(), failure_fields)
    metadata_map = {
        row["apple_track_id"]: row
        for row in (read_csv(metadata_path) if metadata_path.exists() else [])
    }
    metadata_fields = ["apple_track_id", "duration_seconds", "chunks_used", "method", "model", "revision"]
    device = "cuda" if torch.cuda.is_available() and not args.cpu else "cpu"
    torch.manual_seed(settings.seed)
    np.random.seed(settings.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(settings.seed)
    torch.use_deterministic_algorithms(True, warn_only=True)
    model = ClapModel.from_pretrained(settings.clap_model, revision=settings.clap_revision).to(device).eval()
    processor = ClapProcessor.from_pretrained(settings.clap_model, revision=settings.clap_revision)
    session = requests.Session()
    session.headers.update({"User-Agent": "SoundDiversityResearch/0.1"})
    for index, row in enumerate(rows, 1):
        target = out_dir / f"{row['apple_track_id']}.npy"
        if target.exists():
            continue
        try:
            attempts = 0
            while True:
                try:
                    response = session.get(row["preview_url"], timeout=45)
                except requests.RequestException:
                    attempts += 1
                    if attempts >= 5:
                        raise
                    import time
                    time.sleep(min(2 ** attempts, 30))
                    continue
                if 500 <= response.status_code < 600 and attempts < 4:
                    attempts += 1
                    import time
                    time.sleep(min(2 ** attempts, 30))
                    continue
                response.raise_for_status()
                break
            temp_path = None
            with tempfile.NamedTemporaryFile(suffix=".m4a", delete=False) as handle:
                handle.write(response.content)
                handle.flush()
                temp_path = Path(handle.name)
            try:
                audio = decode_preview_audio(temp_path, settings.clap_sample_rate)
            finally:
                if temp_path is not None:
                    temp_path.unlink(missing_ok=True)
            chunk_samples = 10 * settings.clap_sample_rate
            chunks = [
                audio[start:start + chunk_samples]
                for start in (0, chunk_samples, 2 * chunk_samples)
                if start < len(audio)
            ]
            if not chunks:
                raise ValueError("decoded preview contains no audio samples")
            inputs = processor(audios=chunks, sampling_rate=settings.clap_sample_rate, return_tensors="pt")
            inputs = {key: value.to(device) for key, value in inputs.items()}
            with torch.inference_mode():
                chunk_vectors = model.get_audio_features(**inputs)
                vector = chunk_vectors.mean(dim=0)
                vector = vector / torch.linalg.vector_norm(vector)
                vector = vector.cpu().numpy()
            np.save(target, vector.astype(np.float32))
            metadata_map[row["apple_track_id"]] = {
                "apple_track_id": row["apple_track_id"],
                "duration_seconds": f"{len(audio) / settings.clap_sample_rate:.6f}",
                "chunks_used": len(chunks),
                "method": "mean_of_fixed_10s_chunks_then_l2_normalize",
                "model": settings.clap_model,
                "revision": settings.clap_revision,
            }
            write_csv(metadata_path, metadata_map.values(), metadata_fields)
            if row["apple_track_id"] in failure_map:
                del failure_map[row["apple_track_id"]]
                write_csv(failure_path, failure_map.values(), failure_fields)
            safe_print(f"Embedded {index}/{len(rows)}: {row['label']} {row['year']} {row['apple_track']}")
        except Exception as exc:
            failure_map[row["apple_track_id"]] = {
                "apple_track_id": row["apple_track_id"], "label": row["label"],
                "year": row["year"], "artist": row["artist"],
                "track_title": row["track_title"], "error_type": type(exc).__name__,
                "error": str(exc)[:500],
            }
            write_csv(failure_path, failure_map.values(), failure_fields)
            safe_print(f"FAILED {index}/{len(rows)}: {row['apple_track_id']} {type(exc).__name__}: {exc}")


def build_panel(args) -> None:
    labels_path = DATA / "resolved_labels.csv"
    labels = read_csv(labels_path if labels_path.exists() else ROOT / "configs/labels.csv")
    panel = panel_skeleton(labels)
    releases = read_csv(DATA / "releases.csv") if (DATA / "releases.csv").exists() else []
    counts = {}
    for row in releases:
        counts[(row["label"], int(row["year"]))] = counts.get((row["label"], int(row["year"])), 0) + 1
    samples = read_csv(DATA / "sampled_tracks.csv") if (DATA / "sampled_tracks.csv").exists() else []
    grouped = {}
    for row in samples:
        path = EMBEDDING_DIR / f"{row['apple_track_id']}.npy"
        if path.exists():
            grouped.setdefault((row["label"], int(row["year"])), []).append(np.load(path))
    for row in panel:
        key = (row["label"], int(row["year"]))
        row["release_count"] = counts.get(key, 0)
        vectors = grouped.get(key, [])
        row["n_clap_tracks"] = len(vectors)
        diversity = mean_pairwise_diversity(np.stack(vectors)) if len(vectors) >= 2 else None
        row["sound_diversity"] = f"{diversity:.10f}" if diversity is not None else ""
    output = DATA / "full_panel.csv"
    write_csv(output, panel, PANEL_FIELDS)
    provenance = {"settings": json.loads((ROOT / "configs/settings.json").read_text()), "rows": len(panel), "definition": "1 - mean of all off-diagonal pairwise cosine similarities", "embedding_method": "split each preview into fixed 0-10s, 10-20s, and 20-30s chunks; mean available CLAP chunk vectors; L2-normalize", "apple_version_policy": "artist/title similarity is used; version qualifiers such as live, remix, demo, instrumental, or remaster are not required to agree", "apple_403_retry_policy": "cached HTTP 403 searches are retried for label-years below the 12-track target while preserving prior successful matches", "missing_rule": "sound_diversity is NA/blank for fewer than 2 successful CLAP embeddings"}
    (DATA / "provenance.json").write_text(json.dumps(provenance, indent=2), encoding="utf-8")
    print(output)


def skeleton(args) -> None:
    rows = panel_skeleton(read_csv(ROOT / "configs/labels.csv"))
    write_csv(DATA / "panel_skeleton.csv", rows, PANEL_FIELDS)
    print(DATA / "panel_skeleton.csv")


def parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Build the label-year sound-diversity panel")
    sub = p.add_subparsers(dest="command", required=True)
    for name, fn, needs_token in [
        ("skeleton", skeleton, False), ("resolve-labels", resolve_labels, True),
        ("collect-releases", collect_releases, True), ("discover-tracks", discover_tracks, True),
        ("audit-label-ids", audit_label_ids, True),
        ("match-sample", match_and_sample, False), ("retry-403", retry_cached_403, False),
        ("prune-removed-labels", prune_removed_labels, False),
        ("embed", embed, False),
        ("build-panel", build_panel, False),
    ]:
        command = sub.add_parser(name)
        if needs_token:
            command.add_argument("--env-file")
        if name in {"collect-releases", "discover-tracks", "match-sample", "retry-403"}:
            command.add_argument(
                "--label", dest="labels", action="append",
                help="Process only this label; repeat for multiple labels",
            )
        if name == "embed":
            command.add_argument("--cpu", action="store_true")
            command.add_argument("--limit", type=int)
            command.add_argument(
                "--samples",
                help="Sample CSV to embed; defaults to data/sampled_tracks.csv",
            )
        if name == "audit-label-ids":
            command.add_argument("ids", nargs="+", type=int)
        command.set_defaults(func=fn)
    return p


def main() -> None:
    args = parser().parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
