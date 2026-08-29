from __future__ import annotations

import argparse
import json
from pathlib import Path

from .chorus import (
    ANNOTATION_FIELDS, clean_annotation_source_rows, evaluate_annotations, read_dict_rows,
    stratified_annotation_sample, write_dict_rows,
)


def make_annotations(args: argparse.Namespace) -> None:
    rows = read_dict_rows(Path(args.results))
    if args.clean_manifest:
        rows = clean_annotation_source_rows(read_dict_rows(Path(args.clean_manifest)), rows)
    sample = stratified_annotation_sample(rows, size=args.size, seed=args.seed)
    output = Path(args.output)
    write_dict_rows(output, sample, ANNOTATION_FIELDS)
    print(output)


def evaluate(args: argparse.Namespace) -> None:
    metrics = evaluate_annotations(read_dict_rows(Path(args.annotations)))
    text = json.dumps(metrics, indent=2)
    if args.output:
        output = Path(args.output)
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(text + "\n", encoding="utf-8")
        print(output)
    else:
        print(text)


def parser() -> argparse.ArgumentParser:
    root = argparse.ArgumentParser(description="Build and validate the chorus detector")
    commands = root.add_subparsers(dest="command", required=True)
    make = commands.add_parser("make-annotations", help="Create a stratified manual-review sheet")
    make.add_argument("--results", required=True)
    make.add_argument("--output", default="data/chorus_annotations.csv")
    make.add_argument("--clean-manifest", help="Restrict annotations to validated clean preview files")
    make.add_argument("--size", type=int, default=40)
    make.add_argument("--seed", type=int, default=20260627)
    make.set_defaults(func=make_annotations)
    score = commands.add_parser("evaluate", help="Score completed human annotations")
    score.add_argument("--annotations", required=True)
    score.add_argument("--output")
    score.set_defaults(func=evaluate)
    return root


def main() -> None:
    args = parser().parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
