from __future__ import annotations

import argparse
import csv
import json
from collections import Counter
from pathlib import Path


def read_rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8-sig") as handle:
        return list(csv.DictReader(handle))


def write_csv(path: Path, rows: list[dict[str, object]], fields: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def chorus_percent(row: dict[str, str]) -> float:
    try:
        return float(row.get("chorus_percent", "") or 0.0)
    except ValueError:
        return 0.0


def pct_bin(value: float) -> str:
    if value <= 25:
        return "0–25%"
    if value <= 50:
        return "25–50%"
    if value <= 75:
        return "50–75%"
    return "75–100%"


def main() -> int:
    parser = argparse.ArgumentParser(description="Create chorus presence/share summary charts.")
    parser.add_argument("--results", type=Path, default=Path("outputs/chorus_rebuild/ordered_chorus_results_clean_fresh_api.csv"))
    parser.add_argument("--chart-output", type=Path, default=Path("outputs/chorus_rebuild/chorus_presence_and_share.png"))
    parser.add_argument("--summary-output", type=Path, default=Path("outputs/chorus_rebuild/chorus_presence_and_share_summary.json"))
    parser.add_argument("--table-output", type=Path, default=Path("outputs/chorus_rebuild/chorus_presence_and_share_table.csv"))
    args = parser.parse_args()

    rows = read_rows(args.results)
    ok = [row for row in rows if row.get("status") == "ok"]
    positive = [row for row in ok if chorus_percent(row) > 0]
    zero = [row for row in ok if chorus_percent(row) == 0]
    unscored = [row for row in rows if row.get("status") != "ok"]

    presence_counts = {
        "Detected chorus": len(positive),
        "No detected chorus": len(zero),
        "Unscored": len(unscored),
    }
    bin_counts = Counter(pct_bin(chorus_percent(row)) for row in positive)
    bin_order = ["0–25%", "25–50%", "50–75%", "75–100%"]
    positive_pcts = [chorus_percent(row) for row in positive]
    mean_positive_pct = sum(positive_pcts) / len(positive_pcts) if positive_pcts else 0.0
    sorted_pcts = sorted(positive_pcts)
    median_positive_pct = (
        sorted_pcts[len(sorted_pcts) // 2]
        if len(sorted_pcts) % 2
        else (sorted_pcts[len(sorted_pcts) // 2 - 1] + sorted_pcts[len(sorted_pcts) // 2]) / 2
    ) if sorted_pcts else 0.0

    summary = {
        "results": str(args.results),
        "total_previews": len(rows),
        "successfully_scored": len(ok),
        "detected_chorus": len(positive),
        "no_detected_chorus": len(zero),
        "unscored": len(unscored),
        "presence_counts": presence_counts,
        "positive_chorus_share_bins": {label: bin_counts.get(label, 0) for label in bin_order},
        "mean_positive_chorus_percent": mean_positive_pct,
        "median_positive_chorus_percent": median_positive_pct,
    }

    table_rows = (
        [{"section": "presence", "category": key, "count": value} for key, value in presence_counts.items()]
        + [{"section": "positive_chorus_share", "category": label, "count": bin_counts.get(label, 0)} for label in bin_order]
    )
    write_csv(args.table_output, table_rows, ["section", "category", "count"])
    args.summary_output.parent.mkdir(parents=True, exist_ok=True)
    args.summary_output.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")

    import matplotlib.pyplot as plt

    plt.rcParams.update({
        "figure.facecolor": "white",
        "axes.facecolor": "white",
        "font.size": 11,
        "axes.titlesize": 13,
        "axes.labelsize": 11,
    })
    fig, axes = plt.subplots(1, 2, figsize=(12, 5), gridspec_kw={"width_ratios": [1, 1.25]})

    left_labels = list(presence_counts.keys())
    left_values = list(presence_counts.values())
    left_colors = ["#2E86AB", "#A3A3A3", "#E0A458"]
    axes[0].bar(left_labels, left_values, color=left_colors)
    axes[0].set_title("Preview-level chorus detection")
    axes[0].set_ylabel("Number of previews")
    axes[0].bar_label(axes[0].containers[0], padding=3)
    axes[0].set_ylim(0, max(left_values) + 10)
    axes[0].tick_params(axis="x", rotation=15)

    right_values = [bin_counts.get(label, 0) for label in bin_order]
    axes[1].bar(bin_order, right_values, color="#6A4C93")
    axes[1].set_title("Among previews with chorus: chorus share of preview")
    axes[1].set_xlabel("Chorus duration as % of preview")
    axes[1].set_ylabel("Number of previews")
    axes[1].bar_label(axes[1].containers[0], padding=3)
    axes[1].set_ylim(0, max(right_values + [1]) + 10)
    axes[1].text(
        0.02,
        0.95,
        f"Mean: {mean_positive_pct:.1f}%\nMedian: {median_positive_pct:.1f}%",
        transform=axes[1].transAxes,
        va="top",
        ha="left",
        bbox={"boxstyle": "round,pad=0.35", "facecolor": "#F7F7F7", "edgecolor": "#CCCCCC"},
    )

    fig.suptitle("Chorus presence and duration share in clean preview set", fontsize=15, y=1.03)
    fig.text(
        0.5,
        -0.02,
        "Unscored = no counted Genius chorus/hook/refrain section or Whisper produced no word timestamps.",
        ha="center",
        va="top",
        fontsize=9,
        color="#555555",
    )
    fig.tight_layout()
    args.chart_output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.chart_output, dpi=200, bbox_inches="tight")
    print(args.chart_output)
    print(args.summary_output)
    print(args.table_output)
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
