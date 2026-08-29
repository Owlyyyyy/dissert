from __future__ import annotations

import argparse
import csv
import json
from collections import Counter
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def read_rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8-sig") as handle:
        return list(csv.DictReader(handle))


def chorus_percent(row: dict[str, str]) -> float:
    try:
        return float(row.get("chorus_percent", "") or 0.0)
    except ValueError:
        return 0.0


def has_detected_chorus(row: dict[str, str]) -> bool:
    return chorus_percent(row) > 0.0


def pct_bin(value: float) -> str:
    if value < 25:
        return "0–25%"
    if value < 50:
        return "25–50%"
    if value < 75:
        return "50–75%"
    return "75–100%"


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = ["section", "category", "count"]
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def run(args: argparse.Namespace) -> dict[str, object]:
    rows = read_rows(args.results)
    positive = [row for row in rows if has_detected_chorus(row)]
    negative = [row for row in rows if not has_detected_chorus(row)]

    bin_order = ["0–25%", "25–50%", "50–75%", "75–100%"]
    bin_counts = Counter(pct_bin(chorus_percent(row)) for row in positive)
    positive_pcts = [chorus_percent(row) for row in positive]

    mean_positive_pct = float(np.mean(positive_pcts)) if positive_pcts else 0.0
    median_positive_pct = float(np.median(positive_pcts)) if positive_pcts else 0.0
    summary = {
        "results": str(args.results),
        "total_previews": len(rows),
        "detected_chorus": len(positive),
        "no_detected_chorus": len(negative),
        "presence_counts": {
            "Detected chorus": len(positive),
            "No detected chorus": len(negative),
        },
        "positive_chorus_share_bins": {label: bin_counts.get(label, 0) for label in bin_order},
        "mean_positive_chorus_percent": mean_positive_pct,
        "median_positive_chorus_percent": median_positive_pct,
        "note": "Rows with no detected chorus interval are counted as No detected chorus.",
    }

    args.summary_output.parent.mkdir(parents=True, exist_ok=True)
    args.summary_output.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    write_csv(
        args.table_output,
        [
            {"section": "presence", "category": "Detected chorus", "count": len(positive)},
            {"section": "presence", "category": "No detected chorus", "count": len(negative)},
            *[
                {"section": "positive_chorus_share", "category": label, "count": bin_counts.get(label, 0)}
                for label in bin_order
            ],
        ],
    )

    fig, axes = plt.subplots(1, 2, figsize=(12, 5.8))

    presence_labels = ["Detected chorus", "No detected chorus"]
    presence_values = [len(positive), len(negative)]
    presence_colors = ["#2E8BB1", "#A9A9A9"]
    axes[0].bar(presence_labels, presence_values, color=presence_colors)
    axes[0].set_title("Preview-level chorus detection")
    axes[0].set_ylabel("Number of previews")
    axes[0].set_ylim(0, max(presence_values) * 1.15 if presence_values else 1)
    axes[0].tick_params(axis="x", rotation=15)
    for index, value in enumerate(presence_values):
        axes[0].text(index, value + max(presence_values) * 0.02, str(value), ha="center", va="bottom", fontsize=11)

    share_values = [bin_counts.get(label, 0) for label in bin_order]
    axes[1].bar(bin_order, share_values, color="#6E4C9A")
    axes[1].set_title("Among previews with chorus: chorus share of preview")
    axes[1].set_xlabel("Chorus duration as % of preview")
    axes[1].set_ylabel("Number of previews")
    axes[1].set_ylim(0, max(share_values + [1]) * 1.35)
    for index, value in enumerate(share_values):
        axes[1].text(index, value + max(share_values + [1]) * 0.04, str(value), ha="center", va="bottom", fontsize=11)
    axes[1].text(
        0.02,
        0.95,
        f"Mean: {mean_positive_pct:.1f}%\nMedian: {median_positive_pct:.1f}%",
        transform=axes[1].transAxes,
        va="top",
        ha="left",
        fontsize=10,
        bbox={"boxstyle": "round,pad=0.35", "facecolor": "white", "edgecolor": "#CCCCCC"},
    )

    fig.suptitle("Chorus presence and duration share in 500-preview set", fontsize=15, y=1.02)
    fig.tight_layout()
    args.chart_output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.chart_output, dpi=220, bbox_inches="tight")
    plt.close(fig)
    return summary


def parser() -> argparse.ArgumentParser:
    root = argparse.ArgumentParser(description="Create binary chorus detected/not-detected summary charts.")
    root.add_argument("--results", type=Path, default=Path("outputs/chorus_500/ordered_chorus_results_500.csv"))
    root.add_argument("--chart-output", type=Path, default=Path("outputs/chorus_500/chorus_binary_presence_and_share_500.png"))
    root.add_argument("--summary-output", type=Path, default=Path("outputs/chorus_500/chorus_binary_presence_and_share_500_summary.json"))
    root.add_argument("--table-output", type=Path, default=Path("outputs/chorus_500/chorus_binary_presence_and_share_500_table.csv"))
    return root


def main() -> None:
    args = parser().parse_args()
    summary = run(args)
    print(args.chart_output)
    print(args.summary_output)
    print(args.table_output)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
