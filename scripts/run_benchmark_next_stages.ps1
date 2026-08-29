while (Get-Process -Id 29120 -ErrorAction SilentlyContinue) {
  Start-Sleep -Seconds 30
}
python scripts\benchmark_pipeline.py discover-tracks --env-file "C:\path\to\discogs.env" *> data\benchmark_discover_tracks.combined.log
python scripts\benchmark_pipeline.py match-year-sample --tracks-per-year 250 *> data\benchmark_match_year_sample.combined.log
