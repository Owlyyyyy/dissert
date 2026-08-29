from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"
EMBEDDINGS = DATA / "embeddings_full30"


FIELDS = [
    "track_role",
    "audio_distance",
    "track_album",
    "track",
    "artist",
    "album",
    "label",
    "year",
    "apple_track_id",
    "apple_track_url",
]


def latex_escape(value: object) -> str:
    text = "" if pd.isna(value) else str(value)
    replacements = {
        "\\": r"\textbackslash{}",
        "&": r"\&",
        "%": r"\%",
        "$": r"\$",
        "#": r"\#",
        "_": r"\_",
        "{": r"\{",
        "}": r"\}",
        "~": r"\textasciitilde{}",
        "^": r"\textasciicircum{}",
    }
    for old, new in replacements.items():
        text = text.replace(old, new)
    return text


def load_unique_tracks(song_path: Path) -> tuple[pd.DataFrame, np.ndarray]:
    songs = pd.read_csv(song_path)
    rows = []
    vectors = []
    seen = set()

    print(f"Reading {len(songs):,} song rows from {song_path}", flush=True)
    for row_number, row in enumerate(songs.itertuples(index=False), 1):
        apple_id = str(row.apple_track_id)
        if apple_id in seen:
            continue
        path = EMBEDDINGS / f"{apple_id}.npy"
        if not path.exists():
            continue
        vec = np.load(path).astype(np.float64)
        norm = np.linalg.norm(vec)
        if norm == 0:
            continue
        seen.add(apple_id)
        vectors.append(vec / norm)
        rows.append(row._asdict())
        if len(vectors) % 1000 == 0:
            print(f"Loaded {len(vectors):,} unique embeddings...", flush=True)

    print(f"Loaded {len(vectors):,} unique embedded tracks.", flush=True)
    return pd.DataFrame(rows), np.vstack(vectors)


def nearest_to_target(distances: np.ndarray, target: float, used: set[int]) -> int:
    order = np.argsort(np.abs(distances - target))
    for idx in order:
        idx = int(idx)
        if idx not in used:
            return idx
    raise RuntimeError("no unused candidate left")


def build_table(song_path: Path, prefix: str) -> pd.DataFrame:
    tracks, matrix = load_unique_tracks(song_path)
    n = len(tracks)
    if n < 8:
        raise RuntimeError("Need at least 8 embedded tracks")

    # Average cosine distance from each candidate track to every other track:
    # d_i = 1 - mean_j!=i cos(e_i, e_j)
    print("Choosing focal track near median average distance...", flush=True)
    summed = matrix.sum(axis=0)
    mean_cos_to_others = (matrix @ summed - 1.0) / (n - 1)
    avg_dist_to_others = 1.0 - mean_cos_to_others
    targets = [
        ("Very similar track 1", 0.05),
        ("Very similar track 2", 0.08),
        ("Moderately similar track", 0.15),
        ("Moderately distant track", 0.25),
        ("Distant track", 0.35),
        ("Very distant track", 0.45),
        ("Most distant example", 0.60),
    ]

    median_avg_distance = float(np.median(avg_dist_to_others))
    central_candidates = np.argsort(np.abs(avg_dist_to_others - median_avg_distance))[:500]
    target_values = [target for _, target in targets[:-1]]

    best_score = None
    focal_idx = int(central_candidates[0])
    for candidate_idx in central_candidates:
        candidate_distances = np.maximum(1.0 - matrix @ matrix[int(candidate_idx)], 0.0)
        score = abs(avg_dist_to_others[int(candidate_idx)] - median_avg_distance)
        for target in target_values:
            score += 2.0 * float(np.min(np.abs(candidate_distances - target)))
        if best_score is None or score < best_score:
            best_score = score
            focal_idx = int(candidate_idx)

    focal = matrix[focal_idx]
    distances = 1.0 - matrix @ focal
    distances = np.maximum(distances, 0.0)

    used = {focal_idx}
    targets[-1] = ("Most distant example", float(np.quantile(distances, 0.995)))

    selected = [("Focal track", focal_idx)]
    for role, target in targets:
        idx = nearest_to_target(distances, target, used)
        used.add(idx)
        selected.append((role, idx))

    out_rows = []
    for role, idx in selected:
        row = tracks.iloc[idx]
        track = row.get("apple_track") or row.get("track_title")
        artist = row.get("apple_artist") or row.get("artist")
        album = row.get("apple_collection") or row.get("release_title")
        track_album = f"{track} — {artist} ({album})"
        out_rows.append(
            {
                "track_role": role,
                "audio_distance": f"{distances[idx]:.3f}",
                "track_album": track_album,
                "track": track,
                "artist": artist,
                "album": album,
                "label": row.get("label"),
                "year": int(row.get("year")),
                "apple_track_id": row.get("apple_track_id"),
                "apple_track_url": row.get("apple_track_url"),
            }
        )

    output = pd.DataFrame(out_rows, columns=FIELDS)
    print("Writing outputs...", flush=True)
    csv_path = DATA / f"{prefix}_clap_distance_interpretation_table.csv"
    tex_path = DATA / f"{prefix}_clap_distance_interpretation_table.tex"
    output.to_csv(csv_path, index=False)

    lines = [
        r"\begin{tabular}{lll}",
        r"\toprule",
        r"Track & Audio distance & Track / Album \\",
        r"\midrule",
    ]
    for row in output.itertuples(index=False):
        lines.append(
            f"{latex_escape(row.track_role)} & "
            f"{latex_escape(row.audio_distance)} & "
            f"{latex_escape(row.track_album)} \\\\"
        )
    lines.extend([r"\bottomrule", r"\end{tabular}"])
    tex_path.write_text("\n".join(lines) + "\n", encoding="utf-8")

    meta = pd.DataFrame(
        [
            {
                "song_source": str(song_path),
                "embedded_unique_tracks": n,
                "focal_selection_rule": "track with average CLAP distance to all other embedded tracks closest to the dataset median",
                "median_average_distance_to_others": f"{median_avg_distance:.10f}",
                "focal_average_distance_to_others": f"{avg_dist_to_others[focal_idx]:.10f}",
                "distance_definition": "1 - cosine similarity between L2-normalized CLAP embeddings",
            }
        ]
    )
    meta.to_csv(DATA / f"{prefix}_clap_distance_interpretation_metadata.csv", index=False)

    print(csv_path)
    print(tex_path)
    print(DATA / f"{prefix}_clap_distance_interpretation_metadata.csv")
    print(output.to_string(index=False))
    return output


def main() -> None:
    build_table(DATA / "label_year_song_data.csv", "main_sample")


if __name__ == "__main__":
    main()
