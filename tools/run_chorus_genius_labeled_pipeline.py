from __future__ import annotations

import argparse
import csv
import json
import subprocess
import sys
from collections import Counter
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def run_command(command: list[str]) -> None:
    print("\n$ " + " ".join(command), flush=True)
    subprocess.run(command, cwd=ROOT, check=True)


def read_csv(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        return []
    with path.open(newline="", encoding="utf-8-sig") as handle:
        return list(csv.DictReader(handle))


def detected_count(rows: list[dict[str, str]]) -> int:
    total = 0
    for row in rows:
        try:
            if float(row.get("chorus_percent", "") or 0) > 0:
                total += 1
        except ValueError:
            pass
    return total


def summarize(args: argparse.Namespace) -> None:
    manifest_rows = read_csv(args.output_manifest)
    result_rows = read_csv(args.results_output)
    summary = json.loads(args.summary_output.read_text(encoding="utf-8")) if args.summary_output.exists() else {}

    print("\nFinal summary")
    print("-------------")
    print(f"Manifest rows checked: {len(manifest_rows)}")
    print(f"Manifest status: {dict(Counter(row.get('status', '') for row in manifest_rows))}")
    print(f"Detector rows: {len(result_rows)}")
    print(f"Detector status: {dict(Counter(row.get('status', '') for row in result_rows))}")
    print(f"Detected chorus: {detected_count(result_rows)}")
    print(f"No detected chorus: {len(result_rows) - detected_count(result_rows)}")
    if summary:
        print(f"Mean chorus share when detected: {summary.get('mean_positive_chorus_percent', 0):.1f}%")
        print(f"Median chorus share when detected: {summary.get('median_positive_chorus_percent', 0):.1f}%")
    print(f"\nGraph: {args.chart_output}")
    print(f"Results CSV: {args.results_output}")
    print(f"Manifest CSV: {args.output_manifest}")


def parser() -> argparse.ArgumentParser:
    root = argparse.ArgumentParser(
        description=(
            "Rebuild the 500-song chorus analysis using only songs with usable "
            "Genius chorus-family sections, then run detection and make the graph."
        )
    )
    root.add_argument("--sampled-tracks", type=Path, default=Path("data/sampled_tracks.csv"))
    root.add_argument("--genius-token-file", type=Path, required=True)
    root.add_argument("--target-ok", type=int, default=500)
    root.add_argument("--model", default="tiny")
    root.add_argument("--output-folder", type=Path, default=Path("outputs/chorus_500_genius_labeled/previews_clean"))
    root.add_argument("--output-manifest", type=Path, default=Path("data/chorus_preview_manifest_500_genius_labeled.csv"))
    root.add_argument("--results-output", type=Path, default=Path("outputs/chorus_500_genius_labeled/ordered_chorus_results_500.csv"))
    root.add_argument("--chart-output", type=Path, default=Path("outputs/chorus_500_genius_labeled/chorus_binary_presence_and_share_500.png"))
    root.add_argument("--summary-output", type=Path, default=Path("outputs/chorus_500_genius_labeled/chorus_binary_presence_and_share_500_summary.json"))
    root.add_argument("--table-output", type=Path, default=Path("outputs/chorus_500_genius_labeled/chorus_binary_presence_and_share_500_table.csv"))
    root.add_argument("--skip-manifest", action="store_true", help="Use the existing clean manifest.")
    root.add_argument("--skip-detector", action="store_true", help="Use the existing detector results.")
    root.add_argument("--resume-detector", action="store_true", help="Resume detector output instead of rerunning existing rows.")
    return root


def main() -> None:
    args = parser().parse_args()
    python = sys.executable

    if not args.genius_token_file.exists():
        raise FileNotFoundError(f"Genius token file not found: {args.genius_token_file}")

    if not args.skip_manifest:
        run_command([
            python,
            "tools/build_chorus_genius_labeled_manifest.py",
            "--sampled-tracks",
            str(args.sampled_tracks),
            "--output-folder",
            str(args.output_folder),
            "--output-manifest",
            str(args.output_manifest),
            "--target-ok",
            str(args.target_ok),
            "--genius-token-file",
            str(args.genius_token_file),
        ])

    detector_command = [
        python,
        "tools/run_ordered_chorus_detector.py",
        "--manifest",
        str(args.output_manifest),
        "--lyrics-mode",
        "genius_or_cached",
        "--genius-token-file",
        str(args.genius_token_file),
        "--model",
        args.model,
        "--output",
        str(args.results_output),
        "--save-every",
        "5",
    ]
    if args.resume_detector:
        detector_command.append("--resume")

    if not args.skip_detector:
        run_command(detector_command)

    run_command([
        python,
        "tools/make_chorus_binary_summary_chart.py",
        "--results",
        str(args.results_output),
        "--chart-output",
        str(args.chart_output),
        "--summary-output",
        str(args.summary_output),
        "--table-output",
        str(args.table_output),
    ])

    summarize(args)


if __name__ == "__main__":
    main()
