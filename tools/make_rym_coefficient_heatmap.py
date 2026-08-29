from __future__ import annotations

import argparse
import csv
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import TwoSlopeNorm


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8-sig") as handle:
        return list(csv.DictReader(handle))


def short_float(value: str, default: float = 0.0) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def choose_descriptors(rows: list[dict[str, str]], top_n: int) -> list[str]:
    by_descriptor: dict[str, float] = {}
    display_by_descriptor: dict[str, str] = {}
    for row in rows:
        descriptor = row["descriptor"]
        display_by_descriptor[descriptor] = row.get("display_descriptor", descriptor)
        by_descriptor[descriptor] = max(
            by_descriptor.get(descriptor, 0.0),
            abs(short_float(row.get("coefficient", ""))),
        )
    ordered = sorted(by_descriptor, key=lambda item: by_descriptor[item], reverse=True)
    return ordered[:top_n]


def make_matrix(
    rows: list[dict[str, str]],
    descriptors: list[str],
) -> tuple[np.ndarray, list[int], dict[str, str]]:
    pcs = sorted({int(row["PC"]) for row in rows})
    pc_to_col = {pc: index for index, pc in enumerate(pcs)}
    descriptor_to_row = {descriptor: index for index, descriptor in enumerate(descriptors)}
    display_by_descriptor = {
        row["descriptor"]: row.get("display_descriptor", row["descriptor"])
        for row in rows
    }
    matrix = np.zeros((len(descriptors), len(pcs)), dtype=float)
    for row in rows:
        descriptor = row["descriptor"]
        if descriptor not in descriptor_to_row:
            continue
        matrix[descriptor_to_row[descriptor], pc_to_col[int(row["PC"])]] = short_float(row.get("coefficient", ""))
    return matrix, pcs, display_by_descriptor


def pc_labels(pcs: list[int], validation_rows: list[dict[str, str]]) -> list[str]:
    by_pc = {int(row["PC"]): row for row in validation_rows if row.get("PC", "").isdigit()}
    labels = []
    for pc in pcs:
        row = by_pc.get(pc, {})
        explained = short_float(row.get("explained_variance_ratio", "")) * 100
        cv_r2 = short_float(row.get("grouped_cv_r2", ""))
        labels.append(f"PC{pc}\n{explained:.1f}% var.\nCV R²={cv_r2:.2f}")
    return labels


def draw_heatmap(
    coefficients_csv: Path,
    validation_csv: Path,
    output: Path,
    *,
    top_n: int,
) -> None:
    rows = read_csv(coefficients_csv)
    validation_rows = read_csv(validation_csv) if validation_csv.exists() else []
    descriptors = choose_descriptors(rows, top_n)
    matrix, pcs, display_by_descriptor = make_matrix(rows, descriptors)

    # Group descriptors by the PC where they have the strongest absolute loading,
    # then by sign/size within that PC. This keeps the heatmap readable without
    # imposing a clustering dependency.
    order = sorted(
        range(len(descriptors)),
        key=lambda i: (
            int(np.argmax(np.abs(matrix[i]))),
            -abs(float(matrix[i, int(np.argmax(np.abs(matrix[i])))])),
            display_by_descriptor.get(descriptors[i], descriptors[i]),
        ),
    )
    matrix = matrix[order, :]
    descriptors = [descriptors[i] for i in order]
    labels = [display_by_descriptor.get(descriptor, descriptor) for descriptor in descriptors]

    max_abs = float(np.max(np.abs(matrix))) or 1.0
    norm = TwoSlopeNorm(vmin=-max_abs, vcenter=0.0, vmax=max_abs)
    height = max(7.0, 0.28 * len(descriptors) + 2.2)
    fig, ax = plt.subplots(figsize=(8.8, height))
    image = ax.imshow(matrix, aspect="auto", cmap="RdBu", norm=norm)

    ax.set_xticks(np.arange(len(pcs)))
    ax.set_xticklabels(pc_labels(pcs, validation_rows), fontsize=9)
    ax.set_yticks(np.arange(len(labels)))
    ax.set_yticklabels(labels, fontsize=8)
    ax.set_xlabel("CLAP principal component")
    ax.set_ylabel("RYM descriptor")
    ax.set_title(f"Elastic Net coefficient heatmap: top {len(descriptors)} RYM descriptors")

    for i in range(matrix.shape[0]):
        for j in range(matrix.shape[1]):
            value = matrix[i, j]
            if abs(value) >= max_abs * 0.38:
                ax.text(j, i, f"{value:+.2f}", ha="center", va="center", fontsize=6.5, color="white")

    ax.set_xticks(np.arange(-0.5, len(pcs), 1), minor=True)
    ax.set_yticks(np.arange(-0.5, len(labels), 1), minor=True)
    ax.grid(which="minor", color="white", linewidth=0.8)
    ax.tick_params(which="minor", bottom=False, left=False)
    colorbar = fig.colorbar(image, ax=ax, shrink=0.8)
    colorbar.set_label("Elastic Net coefficient")
    fig.tight_layout()
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=220, bbox_inches="tight")
    plt.close(fig)


def parser() -> argparse.ArgumentParser:
    root = argparse.ArgumentParser(description="Create a heatmap of RYM descriptor coefficients across CLAP PCs.")
    root.add_argument("--coefficients", type=Path, default=Path("outputs/rym_clap_interpretability/rym_clap_descriptor_coefficients.csv"))
    root.add_argument("--validation", type=Path, default=Path("outputs/rym_clap_interpretability/rym_clap_model_validation.csv"))
    root.add_argument("--output", type=Path, default=Path("outputs/rym_clap_interpretability/rym_clap_coefficient_heatmap.png"))
    root.add_argument("--top-n", type=int, default=35)
    return root


def main() -> None:
    args = parser().parse_args()
    draw_heatmap(args.coefficients, args.validation, args.output, top_n=args.top_n)
    print(args.output)


if __name__ == "__main__":
    main()
