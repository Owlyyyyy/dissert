from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from sound_diversity.core import (
    ROOT,
    DiscogsClient,
    JsonCache,
    Settings,
    load_env_file,
    normalize,
    read_csv,
    write_csv,
)


DATA = ROOT / "data"
FIELDS = [
    "requested_label",
    "rank",
    "discogs_label_id",
    "discogs_title",
    "resource_url",
    "exact_normalized_match",
]


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Search Discogs candidate IDs for benchmark-only labels."
    )
    parser.add_argument("--env-file", required=True)
    parser.add_argument(
        "--config",
        default=str(ROOT / "configs" / "benchmark_labels.csv"),
        help="Benchmark label registry to search.",
    )
    args = parser.parse_args()

    env_path = Path(args.env_file)
    load_env_file(env_path)
    token = os.environ.get("DISCOGS_TOKEN")
    if not token:
        # Also accept a plain one-line token file, while never printing it.
        raw = env_path.read_text(encoding="utf-8-sig").strip()
        if raw and "=" not in raw and "\n" not in raw:
            token = raw
    if not token:
        raise RuntimeError(
            "Discogs token not found. Expected DISCOGS_TOKEN=... or a plain one-line token file."
        )
    settings = Settings.load()
    discogs = DiscogsClient(
        token,
        settings.discogs_user_agent,
        JsonCache(DATA / "cache"),
    )

    labels = read_csv(Path(args.config))
    candidates = []
    draft = []
    for row in labels:
        requested = row["label"]
        if row["discogs_label_id"].isdigit():
            draft.append(row)
            continue
        results = discogs.search_labels(requested)
        exact = [
            item for item in results
            if normalize(item.get("title", "")) == normalize(requested)
        ]
        chosen = exact[0] if exact and results and exact[0] == results[0] else None
        draft.append({
            **row,
            "discogs_label_id": str(chosen.get("id")) if chosen else "REVIEW",
        })
        for rank, item in enumerate(results[:10], 1):
            candidates.append({
                "requested_label": requested,
                "rank": rank,
                "discogs_label_id": item.get("id", ""),
                "discogs_title": item.get("title", ""),
                "resource_url": item.get("resource_url", ""),
                "exact_normalized_match": int(item in exact),
            })
        print(f"{requested}: {draft[-1]['discogs_label_id']}")

    write_csv(DATA / "benchmark_label_resolution_candidates.csv", candidates, FIELDS)
    write_csv(DATA / "benchmark_resolved_labels_draft.csv", draft, list(labels[0]))
    print(DATA / "benchmark_label_resolution_candidates.csv")
    print(DATA / "benchmark_resolved_labels_draft.csv")


if __name__ == "__main__":
    main()
