from __future__ import annotations

from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"


DISPLAY_SEGMENTS = {
    "country": "Country",
    "electronic_dance": "Electronic/dance",
    "hip_hop_rnb": "Hip hop/R&B",
    "indie_alternative": "Indie/alternative",
    "jazz_classical_experimental": "Jazz/classical/experimental",
    "latin_global": "Latin/global",
    "major_pop": "Major pop/mainstream",
    "metal_punk_hardcore": "Metal/punk/hardcore",
    "rock_pop_punk": "Rock/pop-punk",
}


def latex_escape(value: object) -> str:
    text = str(value)
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


def main() -> None:
    labels = pd.read_csv(DATA / "benchmark_resolved_labels_draft.csv")
    songs = pd.read_csv(DATA / "benchmark_sampled_tracks.csv")

    counts = (
        songs.groupby("label", as_index=False)
        .size()
        .rename(columns={"size": "benchmark_song_rows"})
    )
    total = int(counts["benchmark_song_rows"].sum())
    table = labels.merge(counts, on="label", how="left")
    table["benchmark_song_rows"] = table["benchmark_song_rows"].fillna(0).astype(int)
    table["standout_genre"] = table["market_segment"].map(DISPLAY_SEGMENTS).fillna(table["market_segment"])
    table["share"] = table["benchmark_song_rows"] / total
    table["share_percent"] = table["share"].mul(100).round(1).map(lambda x: f"{x:.1f}%")

    output = table[
        [
            "label",
            "standout_genre",
            "share_percent",
            "benchmark_song_rows",
            "discogs_label_id",
            "start_year",
            "end_year",
            "market_segment",
        ]
    ].sort_values(["standout_genre", "label"])

    csv_path = DATA / "benchmark_label_genre_share_table.csv"
    output.to_csv(csv_path, index=False)

    latex_path = DATA / "benchmark_label_genre_share_table.tex"
    lines = [
        r"\begin{tabular}{lll}",
        r"\toprule",
        r"Benchmark label & Standout genre & Share \\",
        r"\midrule",
    ]
    for row in output.itertuples(index=False):
        lines.append(
            f"{latex_escape(row.label)} & "
            f"{latex_escape(row.standout_genre)} & "
            f"{latex_escape(row.share_percent)} \\\\"
        )
    lines.extend([r"\bottomrule", r"\end{tabular}"])
    latex_path.write_text("\n".join(lines) + "\n", encoding="utf-8")

    print(csv_path)
    print(latex_path)
    print(f"rows={len(output)} total_benchmark_song_rows={total}")


if __name__ == "__main__":
    main()
