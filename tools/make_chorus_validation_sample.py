from __future__ import annotations

import argparse
from pathlib import Path
from typing import Mapping

import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from sound_diversity.chorus import ANNOTATION_FIELDS
from sound_diversity.core import read_csv, stable_seed, write_csv


def annotation_row(index: int, row: Mapping[str, str]) -> dict[str, str]:
    return {
        "sample_id": f"T{index:03d}",
        "audio_path": row.get("output_path", ""),
        "artist": row.get("artist", ""),
        "title": row.get("title", ""),
        "metadata_source": row.get("artist_source", ""),
        "metadata_status": row.get("metadata_status", ""),
        # Keep detector fields blank so the annotation sheet is blind.
        "detector_status": "",
        "predicted_chorus_percent": "",
        "predicted_intervals": "",
        "human_chorus_present": "",
        "human_chorus_intervals": "",
        "human_pre_chorus_intervals": "",
        "human_post_chorus_intervals": "",
        "reviewer": "",
        "notes": "",
    }


def make_sample(args: argparse.Namespace) -> list[dict[str, str]]:
    rows = [row for row in read_csv(Path(args.manifest)) if row.get("status") == "ok"]
    rows.sort(
        key=lambda row: stable_seed(
            args.seed,
            "chorus-100-validation",
            row.get("song_id", ""),
            row.get("artist", ""),
            row.get("title", ""),
        )
    )
    if len(rows) < args.size:
        raise ValueError(f"Only {len(rows)} eligible rows; need {args.size}")
    sample = [annotation_row(index, row) for index, row in enumerate(rows[: args.size], 1)]
    write_csv(Path(args.output), sample, ANNOTATION_FIELDS)
    return sample


def parser() -> argparse.ArgumentParser:
    root = argparse.ArgumentParser(description="Create a blind manual validation sample for chorus detection.")
    root.add_argument("--manifest", default="data/chorus_preview_manifest_500_clean.csv")
    root.add_argument("--output", default="data/chorus_validation_sample_100.csv")
    root.add_argument("--size", type=int, default=100)
    root.add_argument("--seed", type=int, default=20260627)
    return root


def main() -> None:
    args = parser().parse_args()
    sample = make_sample(args)
    print(f"Wrote {args.output} with {len(sample)} rows")


if __name__ == "__main__":
    main()
