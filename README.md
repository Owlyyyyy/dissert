# Sound-diversity panel

This project builds a complete label-year panel for 40 donor labels and four treated labels (2005–2025, with the specified later donor starts). `release_count` comes from Discogs and is independent of Apple preview availability. Valid Discogs tracks are put in a deterministic random order per label-year and searched on iTunes until 12 distinct preview matches are found or the candidate pool is exhausted. Sound diversity is:

`1 - mean(all unique pairwise cosine similarities between CLAP embeddings)`

The outcome is blank/NA when fewer than two tracks can be embedded. Preview audio is processed in a temporary file and is not retained; only embeddings and match provenance are checkpointed.

## Reproducibility choices

- Seed: `20260627`
- Storefront: US
- Model: `laion/clap-htsat-unfused`, revision `8fa0f1c`
- Input audio: mono, 48 kHz
- Full-preview embedding: fixed 0–10 s, 10–20 s, and 20–30 s chunks are embedded separately; available chunk vectors are averaged and L2-normalized
- Target sample: 12 distinct Apple-preview matches per label-year
- Apple version policy: artist/title similarity is used; qualifiers such as live, remix, demo, instrumental, or remaster are not required to agree between Discogs and Apple
- Replacement rule: failed, forbidden, and duplicate Apple matches are replaced using the next track in the deterministic Discogs order
- Cached 403 retry: underfilled label-years retry cached Apple HTTP 403 searches while preserving all successful matches
- Compilation handling: track-level artist credit first, release artist only as fallback
- Invalid artists: `Various`, `Various Artists`, unknown, and blank credits are excluded before sampling
- `n_clap_tracks` may still be below 12 when the candidate pool is exhausted or a matched preview later fails during download/embedding

## Setup

```powershell
python -m venv .venv
.venv\Scripts\python -m pip install -r requirements.txt
```

FFmpeg must be installed in a form usable by `librosa`/`soundfile` for Apple M4A previews. The Discogs token file must define `DISCOGS_TOKEN`; it is never copied into the project.

## Run

Each command is checkpointed and safe to resume:

```powershell
python -m sound_diversity.cli skeleton
python -m sound_diversity.cli resolve-labels --env-file "C:\path\to\token.env"
# Review data/label_resolution_candidates.csv and correct any REVIEW IDs in data/resolved_labels.csv.
python -m sound_diversity.cli collect-releases --env-file "C:\path\to\token.env"
python -m sound_diversity.cli discover-tracks --env-file "C:\path\to\token.env"
python -m sound_diversity.cli match-sample
python -m sound_diversity.cli retry-403
python -m sound_diversity.cli embed
python -m sound_diversity.cli build-panel
```

Final output: `data/full_panel.csv`. Matching evidence remains in `data/sampled_tracks.csv`, and exact methodological settings are written to `data/provenance.json`.

## Benchmark diversity index

The SDID donor/control panel is intentionally kept separate from the broader
benchmark exercise. Donor labels are chosen for comparability with treated
labels; benchmark labels are chosen to provide a broader market-scale yardstick
for interpreting the size of an estimated treatment effect.

Benchmark-only labels live in `configs/benchmark_labels.csv`. They should not be
merged into `configs/labels.csv` unless the causal design is being changed on
purpose.

Draft Discogs ID candidates for benchmark labels:

```powershell
python scripts\audit_benchmark_labels.py --env-file "C:\path\to\token.env"
```

Review `data/benchmark_label_resolution_candidates.csv`, then save verified IDs
before collecting benchmark releases/tracks. Once benchmark song rows and
embeddings exist, compute annual benchmark diversity:

```powershell
python scripts\make_market_benchmark_diversity.py `
  --songs data\benchmark_sampled_tracks.csv `
  --prefix market_benchmark `
  --tracks-per-year 1000 `
  --repetitions 100
```

For a narrower “study-sample benchmark” using the current song-level table:

```powershell
python scripts\make_market_benchmark_diversity.py `
  --songs data\label_year_song_data.csv `
  --prefix study_sample_benchmark_250 `
  --tracks-per-year 250 `
  --repetitions 100
```

The benchmark definition is the same as the main outcome:
`1 - mean(all unique pairwise cosine similarities between CLAP embeddings)`.
The annual-change file can be used to say whether an ATT is large relative to
typical year-to-year movement in the benchmark diversity index.

## Chorus detector rebuild

The rebuilt chorus foundation lives in `sound_diversity/chorus.py`. Its methodological defaults are:

- `[Artist: Chorus]` and `[Chorus: Artist]` both classify as `chorus`.
- `pre-chorus` and `post-chorus` are retained as separate section types and excluded from chorus duration.
- `chorus`, `hook`, and `refrain` count in the primary definition.
- Track-level artist provenance is preferred; `Various` is never sent to a lyrics service as if it were an artist.
- Title qualifiers such as remix, live, demo, instrumental, and remaster are preserved.
- Lyric matching is order-sensitive, and only distinct best matches are retained. Dense overlapping fuzzy hits are not unioned into a whole-preview interval.
- Human presence and timing accuracy are evaluated separately. `uncertain` annotations are excluded from headline metrics.

Regenerate a legacy preview set from Discogs track-level credits and validated Apple matches:

```powershell
python -m sound_diversity.preview_rebuild_cli `
  --source-manifest "C:\path\to\song_level_with_previews_retry_relaxed.csv" `
  --old-preview-folder "C:\path\to\legacy_previews" `
  --output-folder outputs\chorus_rebuild\previews_clean `
  --output-manifest data\chorus_preview_manifest_clean.csv `
  --env-file "C:\path\to\discogs_token.env"
```

The rebuild prefers a Discogs track artist, falls back to a usable release artist, and excludes rows that still resolve only to `Various`. A legacy audio file is copied only if its recorded Apple artist/title passes the current matching thresholds; otherwise Apple is searched again and the preview is freshly downloaded. Every decision and exclusion is checkpointed in the output manifest.

Create a deterministic, stratified 40-preview annotation set from the legacy results:

```powershell
python -m sound_diversity.chorus_cli make-annotations `
  --results "C:\path\to\chorus_duration_results.csv" `
  --clean-manifest data\chorus_preview_manifest_clean.csv `
  --output data\chorus_annotations.csv `
  --size 40
```

Enter `yes`, `no`, or `uncertain`. Human worksheet intervals may use preview-clock notation such as `0:08-0:21;0:25-0:30`; import/evaluation code normalizes these to seconds internally. Annotate without viewing the old predictions. After annotation:

```powershell
python -m sound_diversity.chorus_cli evaluate `
  --annotations data\chorus_annotations.csv `
  --output data\chorus_validation_metrics.json
```

The completed annotation workbook can be normalized to CSV, audited against the old detector, and used to tune the ordered matcher:

```powershell
python tools\import_completed_chorus_workbook.py `
  --workbook "C:\path\to\chorus_annotation_workbook.xlsx" `
  --source-csv data\chorus_annotations_clean.csv `
  --output data\chorus_annotations_completed.csv

python tools\audit_chorus_annotations.py `
  --annotations data\chorus_annotations_completed.csv `
  --old-results "C:\path\to\chorus_duration_results.csv" `
  --audit-output outputs\chorus_rebuild\old_detector_error_audit.csv `
  --summary-output outputs\chorus_rebuild\old_detector_error_summary.json

python tools\tune_ordered_chorus_detector.py `
  --annotations data\chorus_annotations_completed.csv `
  --old-results "C:\path\to\chorus_duration_results.csv" `
  --model tiny `
  --transcribe-limit 35 `
  --min-scores 0.50,0.56,0.62 `
  --min-query-tokens 4 `
  --max-ious 0.50 `
  --gap-tolerances 0.75 `
  --pads 0,2,4,6 `
  --grid-output outputs\chorus_rebuild\ordered_detector_tuning_grid.csv `
  --predictions-output outputs\chorus_rebuild\ordered_detector_predictions.csv `
  --summary-output outputs\chorus_rebuild\ordered_detector_tuning_summary.json

python tools\compare_chorus_metrics.py `
  --annotations data\chorus_annotations_completed.csv `
  --ordered-predictions outputs\chorus_rebuild\ordered_detector_predictions.csv `
  --ordered-summary outputs\chorus_rebuild\ordered_detector_tuning_summary.json `
  --output outputs\chorus_rebuild\old_vs_ordered_metrics.json
```

Run the tuned ordered matcher on the clean preview manifest:

```powershell
python tools\run_ordered_chorus_detector.py `
  --manifest data\chorus_preview_manifest_clean.csv `
  --old-results "C:\path\to\chorus_duration_results.csv" `
  --lyrics-mode cached `
  --model tiny `
  --resume `
  --output outputs\chorus_rebuild\ordered_chorus_results_clean_cached.csv
```

`--lyrics-mode cached` uses only the legacy run's cached Genius chorus text, so it is useful for no-network reproduction but can inherit the old run's section-text limitations. For the dissertation-final run, prefer `--lyrics-mode genius_or_cached` after installing `lyricsgenius` and providing `GENIUS_TOKEN` through `--env-file`; fresh Genius lyrics preserve the rebuilt parser's distinction between chorus/hook/refrain and pre/post-chorus.

Apple preview access and use remain subject to Apple's applicable terms and any permissions obtained by the researcher.

### 500-preview chorus sample with 100-preview validation set

Build a clean 500-song chorus sample directly from the current sampled Apple
preview matches:

```powershell
python tools\build_chorus_500_manifest.py `
  --sampled-tracks data\sampled_tracks.csv `
  --output-folder outputs\chorus_500\previews_clean `
  --output-manifest data\chorus_preview_manifest_500_clean.csv `
  --target-ok 500
```

Create a blind 100-preview manual validation sample from those 500 previews:

```powershell
python tools\make_chorus_validation_sample.py `
  --manifest data\chorus_preview_manifest_500_clean.csv `
  --output data\chorus_validation_sample_100.csv `
  --size 100
```

The 100-preview file is for manual evaluation only. It should not be used to
construct the final 500-preview chorus prevalence estimates.

## RYM / CLAP interpretability check

The supplementary validation analysis links CLAP audio embeddings to independent,
human-generated Rate Your Music descriptors. It does not replace the main
sound-diversity measure; it checks whether major CLAP embedding directions are
associated with recognizable musical descriptors.

Copy the cleaned RYM export to `data/rym_clean1.csv`, then run:

```powershell
python tools\run_rym_clap_interpretability.py `
  --rym-csv data\rym_clean1.csv `
  --n-songs 500 `
  --max-tracks-per-release 2 `
  --min-descriptor-count 15 `
  --n-pcs 5 `
  --cv-folds 5
```

The script uses strict normalized artist+release matching only, with
Apple artist+collection as a fallback. It deliberately avoids title-only
matching because common album titles can create false matches. The selected
sample is capped at two tracks per RYM release so album-level descriptors do not
overwhelm the song-level analysis. Elastic Net validation uses grouped
cross-validation by RYM artist+release, keeping songs from the same album in the
same fold to avoid album-level descriptor leakage.

Outputs are written to `outputs/rym_clap_interpretability/`:

- `rym_clap_matched_500.csv`: the 500 matched song/release observations
- `rym_clap_pca_scores_500.csv`: song-level PCA scores
- `rym_clap_descriptor_coefficients.csv`: Elastic Net descriptor coefficients
- `rym_clap_model_validation.csv`: PC-level grouped cross-validated R²
- `rym_clap_pc_descriptor_summary.csv`: top positive/negative tags by PC
- `rym_clap_pc1_pc2_wordclouds_and_bars.png`: PC1/PC2 descriptor figure
- `rym_clap_explained_variance.png`: explained-variance chart
