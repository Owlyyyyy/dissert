# Data directory

This GitHub export intentionally does **not** include the large generated CSVs,
Discogs/iTunes caches, CLAP embeddings, Apple preview audio, logs, or token
files.

Regenerate the main panel with:

```powershell
python -m sound_diversity.cli skeleton
python -m sound_diversity.cli resolve-labels --env-file "C:\path\to\token.env"
python -m sound_diversity.cli collect-releases --env-file "C:\path\to\token.env"
python -m sound_diversity.cli discover-tracks --env-file "C:\path\to\token.env"
python -m sound_diversity.cli match-sample
python -m sound_diversity.cli retry-403
python -m sound_diversity.cli embed
python -m sound_diversity.cli build-panel
```

Regenerate the benchmark tables using the scripts in `scripts/`.
