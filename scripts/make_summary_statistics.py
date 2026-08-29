from __future__ import annotations

from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"


def describe_numeric(df: pd.DataFrame, variables: list[str]) -> pd.DataFrame:
    rows = []
    for variable in variables:
        if variable not in df.columns:
            continue
        s = pd.to_numeric(df[variable], errors="coerce")
        clean = s.dropna()
        rows.append(
            {
                "variable": variable,
                "n": int(clean.shape[0]),
                "missing": int(s.isna().sum()),
                "mean": clean.mean() if not clean.empty else pd.NA,
                "sd": clean.std(ddof=1) if clean.shape[0] > 1 else pd.NA,
                "min": clean.min() if not clean.empty else pd.NA,
                "p25": clean.quantile(0.25) if not clean.empty else pd.NA,
                "median": clean.median() if not clean.empty else pd.NA,
                "p75": clean.quantile(0.75) if not clean.empty else pd.NA,
                "max": clean.max() if not clean.empty else pd.NA,
            }
        )
    return pd.DataFrame(rows)


def main() -> None:
    panel = pd.read_csv(DATA / "full_panel.csv")
    songs = pd.read_csv(DATA / "label_year_song_data.csv")

    panel_numeric = [
        "year",
        "release_count",
        "sound_diversity",
        "n_clap_tracks",
        "ever_treated",
        "treated",
        "event_time",
    ]
    panel_stats = describe_numeric(panel, panel_numeric)
    panel_stats.to_csv(DATA / "summary_main_panel_stats.csv", index=False)

    song_numeric = [
        "year",
        "discogs_release_id",
        "artist_usable",
        "sample_order",
        "apple_track_id",
        "match_score",
    ]
    song_stats = describe_numeric(songs, song_numeric)
    song_stats.to_csv(DATA / "summary_song_level_stats.csv", index=False)

    panel_year = pd.to_numeric(panel["year"], errors="coerce")
    song_year = pd.to_numeric(songs["year"], errors="coerce")
    release_count = pd.to_numeric(panel["release_count"], errors="coerce")
    n_clap_tracks = pd.to_numeric(panel["n_clap_tracks"], errors="coerce")
    diversity = pd.to_numeric(panel["sound_diversity"], errors="coerce")

    overview_rows = [
        ("panel_rows_label_years", len(panel)),
        ("panel_unique_labels", panel["label"].nunique()),
        ("panel_donor_labels", int((panel["ever_treated"].astype(str) == "0").groupby(panel["label"]).first().sum())),
        ("panel_treated_labels", int((panel["ever_treated"].astype(str) == "1").groupby(panel["label"]).first().sum())),
        ("panel_min_year", int(panel_year.min())),
        ("panel_max_year", int(panel_year.max())),
        ("panel_rows_with_diversity", int(diversity.notna().sum())),
        ("panel_rows_missing_diversity", int(diversity.isna().sum())),
        ("panel_rows_with_release_count_zero", int((release_count == 0).sum())),
        ("panel_rows_with_n_clap_tracks_zero", int((n_clap_tracks == 0).sum())),
        ("panel_rows_with_full_12_clap_tracks", int((n_clap_tracks == 12).sum())),
        ("song_rows", len(songs)),
        ("song_unique_label_years", songs[["label", "year"]].drop_duplicates().shape[0]),
        ("song_unique_labels", songs["label"].nunique()),
        ("song_min_year", int(song_year.min())),
        ("song_max_year", int(song_year.max())),
        ("song_unique_discogs_releases", songs["discogs_release_id"].nunique()),
        ("song_unique_apple_tracks", songs["apple_track_id"].nunique()),
        ("song_rows_artist_from_track", int((songs["artist_source"] == "track").sum())),
        ("song_rows_artist_from_release", int((songs["artist_source"] == "release").sum())),
        ("song_rows_release_artist_various", int(songs["release_artist"].astype(str).str.lower().eq("various").sum())),
        ("song_rows_final_artist_various", int(songs["artist"].astype(str).str.lower().isin(["various", "various artists"]).sum())),
        ("song_rows_missing_preview_url", int(songs["preview_url"].isna().sum())),
    ]
    overview = pd.DataFrame(overview_rows, columns=["metric", "value"])
    overview.to_csv(DATA / "summary_dataset_overview.csv", index=False)

    with pd.ExcelWriter(DATA / "summary_statistics.xlsx") as writer:
        overview.to_excel(writer, sheet_name="overview", index=False)
        panel_stats.to_excel(writer, sheet_name="main_panel_stats", index=False)
        song_stats.to_excel(writer, sheet_name="song_level_stats", index=False)

    print("Wrote:")
    print(DATA / "summary_dataset_overview.csv")
    print(DATA / "summary_main_panel_stats.csv")
    print(DATA / "summary_song_level_stats.csv")
    print(DATA / "summary_statistics.xlsx")


if __name__ == "__main__":
    main()
