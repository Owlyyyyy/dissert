from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from sound_diversity.chorus import evaluate_annotations


def read_rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8-sig") as handle:
        return list(csv.DictReader(handle))


def remap_prediction(rows: list[dict[str, str]], prediction_col: str) -> list[dict[str, str]]:
    output = []
    for row in rows:
        item = dict(row)
        item["predicted_intervals"] = row.get(prediction_col, "")
        output.append(item)
    return output


def subset(rows: list[dict[str, str]], ids: set[str] | None) -> list[dict[str, str]]:
    if ids is None:
        return rows
    return [row for row in rows if row.get("sample_id", "") in ids]


def main() -> int:
    parser = argparse.ArgumentParser(description="Compare old and ordered chorus detector metrics.")
    parser.add_argument("--annotations", type=Path, default=Path("data/chorus_annotations_completed.csv"))
    parser.add_argument("--ordered-predictions", type=Path, default=Path("outputs/chorus_rebuild/ordered_detector_predictions.csv"))
    parser.add_argument("--ordered-summary", type=Path, default=Path("outputs/chorus_rebuild/ordered_detector_tuning_summary.json"))
    parser.add_argument("--output", type=Path, default=Path("outputs/chorus_rebuild/old_vs_ordered_metrics.json"))
    args = parser.parse_args()

    annotations = read_rows(args.annotations)
    ordered = read_rows(args.ordered_predictions)
    summary = json.loads(args.ordered_summary.read_text(encoding="utf-8"))
    groups = {
        "all": None,
        "tune": set(summary["tune_ids"]),
        "holdout": set(summary["holdout_ids"]),
    }
    comparison = {}
    for name, ids in groups.items():
        old_rows = subset(annotations, ids)
        ordered_rows = remap_prediction(subset(ordered, ids), "ordered_predicted_intervals")
        comparison[name] = {
            "old": evaluate_annotations(old_rows),
            "ordered": evaluate_annotations(ordered_rows),
        }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(comparison, indent=2) + "\n", encoding="utf-8")
    print(args.output)
    print(json.dumps(comparison, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
