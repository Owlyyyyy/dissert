from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from sound_diversity.core import ROOT, read_csv, stable_seed, write_csv


DATA = ROOT / "data"
DEFAULT_EMBEDDINGS = DATA / "embeddings_full30"
OUT_FIELDS = [
    "year",
    "available_tracks",
    "target_tracks",
    "repetitions",
    "mean_market_diversity",
    "sd_market_diversity",
    "median_market_diversity",
    "min_market_diversity",
    "max_market_diversity",
]
CHANGE_FIELDS = [
    "year",
    "market_diversity",
    "lag_market_diversity",
    "annual_change",
    "abs_annual_change",
]


def load_vectors(rows: list[dict[str, str]], embedding_dir: Path) -> dict[str, np.ndarray]:
    vectors = {}
    for row in rows:
        apple_id = str(row.get("apple_track_id", "")).strip()
        if not apple_id or apple_id in vectors:
            continue
        path = embedding_dir / f"{apple_id}.npy"
        if path.exists():
            vectors[apple_id] = np.load(path)
    return vectors


def fast_mean_pairwise_diversity(matrix: np.ndarray) -> float:
    """Compute 1 - mean off-diagonal cosine similarity without materializing n x n.

    For unit vectors, the sum of all pairwise dot products has a closed form:
    (||sum_i v_i||^2 - n) / 2 for unordered pairs. Existing project embeddings
    are already L2-normalized, but this re-normalizes defensively.
    """
    matrix = np.asarray(matrix, dtype=np.float64)
    if matrix.ndim != 2 or len(matrix) < 2:
        raise ValueError("need at least two embeddings")
    norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    if np.any(norms == 0):
        raise ValueError("zero-length embedding")
    unit = matrix / norms
    n = len(unit)
    summed = unit.sum(axis=0)
    ordered_off_diagonal_sum = float(summed @ summed - n)
    mean_similarity = ordered_off_diagonal_sum / (n * (n - 1))
    return 1.0 - mean_similarity


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Compute annual broad-market or sample-wide diversity from song rows "
            "and existing CLAP embeddings."
        )
    )
    parser.add_argument(
        "--songs",
        default=str(DATA / "benchmark_sampled_tracks.csv"),
        help="Song-level CSV. Use benchmark_sampled_tracks.csv for the broad benchmark, or label_year_song_data.csv for the current study-sample benchmark.",
    )
    parser.add_argument("--embedding-dir", default=str(DEFAULT_EMBEDDINGS))
    parser.add_argument("--tracks-per-year", type=int, default=1000)
    parser.add_argument("--repetitions", type=int, default=100)
    parser.add_argument("--seed", type=int, default=20260627)
    parser.add_argument(
        "--prefix",
        default="market_benchmark",
        help="Output prefix inside data/, e.g. market_benchmark or study_sample_benchmark.",
    )
    args = parser.parse_args()

    songs_path = Path(args.songs)
    embedding_dir = Path(args.embedding_dir)
    rows = read_csv(songs_path)
    vectors = load_vectors(rows, embedding_dir)

    by_year: dict[int, list[dict[str, str]]] = {}
    seen_by_year: dict[int, set[str]] = {}
    for row in rows:
        apple_id = str(row.get("apple_track_id", "")).strip()
        if apple_id not in vectors:
            continue
        year = int(row["year"])
        seen = seen_by_year.setdefault(year, set())
        if apple_id in seen:
            continue
        seen.add(apple_id)
        by_year.setdefault(year, []).append(row)

    output = []
    for year in sorted(by_year):
        available = by_year[year]
        available_ids = [str(row["apple_track_id"]) for row in available]
        n = min(args.tracks_per_year, len(available_ids))
        if n < 2:
            output.append({
                "year": year,
                "available_tracks": len(available_ids),
                "target_tracks": args.tracks_per_year,
                "repetitions": args.repetitions,
                "mean_market_diversity": "",
                "sd_market_diversity": "",
                "median_market_diversity": "",
                "min_market_diversity": "",
                "max_market_diversity": "",
            })
            continue
        estimates = []
        for rep in range(args.repetitions):
            rng = random.Random(stable_seed(args.seed, "market_benchmark", year, rep))
            chosen = rng.sample(available_ids, n)
            matrix = np.stack([vectors[apple_id] for apple_id in chosen])
            estimates.append(fast_mean_pairwise_diversity(matrix))
        arr = np.array(estimates, dtype=float)
        output.append({
            "year": year,
            "available_tracks": len(available_ids),
            "target_tracks": args.tracks_per_year,
            "repetitions": args.repetitions,
            "mean_market_diversity": f"{arr.mean():.10f}",
            "sd_market_diversity": f"{arr.std(ddof=1):.10f}" if len(arr) > 1 else "",
            "median_market_diversity": f"{np.median(arr):.10f}",
            "min_market_diversity": f"{arr.min():.10f}",
            "max_market_diversity": f"{arr.max():.10f}",
        })

    changes = []
    previous = None
    for row in output:
        value = row["mean_market_diversity"]
        if value == "":
            previous = None
            continue
        value_float = float(value)
        if previous is not None:
            change = value_float - previous[1]
            changes.append({
                "year": row["year"],
                "market_diversity": f"{value_float:.10f}",
                "lag_market_diversity": f"{previous[1]:.10f}",
                "annual_change": f"{change:.10f}",
                "abs_annual_change": f"{abs(change):.10f}",
            })
        previous = (row["year"], value_float)

    benchmark_path = DATA / f"{args.prefix}_annual_diversity.csv"
    changes_path = DATA / f"{args.prefix}_annual_changes.csv"
    provenance_path = DATA / f"{args.prefix}_provenance.json"
    write_csv(benchmark_path, output, OUT_FIELDS)
    write_csv(changes_path, changes, CHANGE_FIELDS)
    provenance_path.write_text(
        json.dumps(
            {
                "songs": str(songs_path),
                "embedding_dir": str(embedding_dir),
                "tracks_per_year": args.tracks_per_year,
                "repetitions": args.repetitions,
                "seed": args.seed,
                "definition": "1 - mean of all off-diagonal pairwise cosine similarities among sampled CLAP embeddings",
                "note": "Use benchmark_sampled_tracks.csv for broad-market benchmark; label_year_song_data.csv only gives study-sample benchmark.",
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    print(benchmark_path)
    print(changes_path)
    print(provenance_path)


if __name__ == "__main__":
    main()
