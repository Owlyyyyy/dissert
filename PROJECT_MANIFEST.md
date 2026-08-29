# Project manifest

This folder is a GitHub-ready code export for the sound-diversity project. It
contains the reproducible scripts, package code, tests, and configs used for the
main label-year panel, benchmark diversity index, summary statistics, and related
chorus/preview validation work.

## Main sound-diversity panel

- `sound_diversity/core.py` — shared helpers, Discogs/iTunes clients, artist
  cleaning, stable seeds, cosine-distance diversity.
- `sound_diversity/cli.py` — main command-line pipeline:
  `resolve-labels`, `collect-releases`, `discover-tracks`, `match-sample`,
  `retry-403`, `embed`, and `build-panel`.
- `configs/labels.csv` — donor and treated label registry.
- `configs/settings.json` — reproducibility settings.

## Benchmark diversity index

- `configs/benchmark_labels.csv` — benchmark-only labels.
- `scripts/audit_benchmark_labels.py` — Discogs ID candidate search for benchmark
  labels.
- `scripts/benchmark_pipeline.py` — benchmark release/track discovery and Apple
  matching.
- `scripts/make_market_benchmark_diversity.py` — annual benchmark diversity and
  annual-change tables from CLAP embeddings.
- `scripts/make_benchmark_label_genre_table.py` — benchmark label/genre/share
  table.

## Tables and interpretation

- `scripts/make_summary_statistics.py` — main panel and song-level summary
  statistics.
- `scripts/make_clap_distance_interpretation_table.py` — focal-track CLAP
  distance interpretation table.

## Chorus / preview validation

- `sound_diversity/preview_rebuild.py`
- `sound_diversity/preview_rebuild_cli.py`
- `sound_diversity/chorus.py`
- `sound_diversity/chorus_cli.py`
- `tools/*.py`
- `tools/*.mjs`

## Tests

- `tests/test_core.py`

## Not included

The export excludes secrets, caches, audio previews, CLAP embedding arrays, large
generated CSVs, backup folders, and logs. Those should stay local unless they are
published through a separate data archive.
