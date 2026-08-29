from __future__ import annotations

import argparse
import json
import math
import sys
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

import matplotlib.pyplot as plt
import numpy as np
from sklearn.decomposition import PCA
from sklearn.linear_model import ElasticNet
from sklearn.metrics import r2_score
from sklearn.model_selection import GridSearchCV, GroupKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from sound_diversity.core import normalize, read_csv, stable_seed, write_csv


MATCH_FIELDS = [
    "analysis_id",
    "apple_track_id",
    "label",
    "year",
    "artist",
    "track_title",
    "release_title",
    "apple_artist",
    "apple_track",
    "apple_collection",
    "rym_artist",
    "rym_release",
    "rym_release_type",
    "rym_descriptors",
    "match_method",
]

SCORE_FIELDS = [
    "analysis_id",
    "apple_track_id",
    "artist",
    "track_title",
    "release_title",
    "PC",
    "score",
    "explained_variance_ratio",
]

COEFFICIENT_FIELDS = [
    "PC",
    "descriptor",
    "display_descriptor",
    "coefficient",
    "abs_coefficient",
    "descriptor_count",
    "selected",
    "alpha",
    "l1_ratio",
    "cv_folds",
    "grouped_cv_r2",
    "grouped_cv_fold_r2_mean",
    "in_sample_r2",
]

MODEL_SUMMARY_FIELDS = [
    "PC",
    "explained_variance_ratio",
    "alpha",
    "l1_ratio",
    "cv_folds",
    "cv_grouping",
    "selected_descriptors",
    "grouped_cv_r2",
    "grouped_cv_fold_r2_mean",
    "grouped_cv_fold_r2_std",
    "final_model_grouped_cv_mean_r2",
    "in_sample_r2",
]


@dataclass(frozen=True)
class MatchedTrack:
    sampled_track: dict[str, str]
    rym_release: dict[str, str]
    match_method: str
    release_key: tuple[str, str]


def descriptor_list(value: str) -> list[str]:
    return [
        tag.strip().casefold()
        for tag in (value or "").split(",")
        if tag.strip() and tag.strip().casefold() != "na"
    ]


DISPLAY_OVERRIDES = {
    "malevocals": "male vocals",
    "femalevocals": "female vocals",
    "conceptalbum": "concept album",
    "wallof sound": "wall of sound",
    "lofi": "lo-fi",
    "hifi": "hi-fi",
    "uncommontimesignatures": "uncommon time",
    "singersongwriter": "singer-songwriter",
    "rockopera": "rock opera",
    "scifi": "sci-fi",
}


def display_descriptor(tag: str) -> str:
    normalized = tag.strip().casefold()
    return DISPLAY_OVERRIDES.get(normalized, normalized.replace("_", " "))


def rym_key(row: Mapping[str, str]) -> tuple[str, str]:
    return normalize(row.get("artist_name", "")), normalize(row.get("release_name", ""))


def build_rym_lookup(rows: Sequence[dict[str, str]]) -> dict[tuple[str, str], dict[str, str]]:
    lookup: dict[tuple[str, str], dict[str, str]] = {}
    for row in rows:
        key = rym_key(row)
        if not all(key):
            continue
        current = lookup.get(key)
        if current is None:
            lookup[key] = row
            continue
        # If an exact artist-release pair appears more than once, keep the row
        # with the larger RYM audience because descriptors should be more stable.
        try:
            new_count = int(float(row.get("rating_count", "") or 0))
            old_count = int(float(current.get("rating_count", "") or 0))
        except ValueError:
            new_count = old_count = 0
        if new_count > old_count:
            lookup[key] = row
    return lookup


def exact_match_track(
    row: dict[str, str],
    rym_lookup: Mapping[tuple[str, str], dict[str, str]],
) -> tuple[dict[str, str], str, tuple[str, str]] | None:
    candidates = [
        ("artist_release_title", (normalize(row.get("artist", "")), normalize(row.get("release_title", "")))),
        ("release_artist_release_title", (normalize(row.get("release_artist", "")), normalize(row.get("release_title", "")))),
        ("apple_artist_collection", (normalize(row.get("apple_artist", "")), normalize(row.get("apple_collection", "")))),
    ]
    for method, key in candidates:
        hit = rym_lookup.get(key)
        if hit is not None:
            return hit, method, key
    return None


def collect_strict_matches(
    sampled_tracks: Sequence[dict[str, str]],
    rym_rows: Sequence[dict[str, str]],
    embedding_dir: Path,
) -> list[MatchedTrack]:
    rym_lookup = build_rym_lookup(rym_rows)
    embedded_ids = {path.stem for path in embedding_dir.glob("*.npy")}
    seen_apple_ids: set[str] = set()
    matches: list[MatchedTrack] = []
    for row in sampled_tracks:
        apple_track_id = str(row.get("apple_track_id", "")).strip()
        if not apple_track_id or apple_track_id in seen_apple_ids:
            continue
        if apple_track_id not in embedded_ids:
            continue
        match = exact_match_track(row, rym_lookup)
        if match is None:
            continue
        rym_release, method, release_key = match
        if not descriptor_list(rym_release.get("descriptors", "")):
            continue
        seen_apple_ids.add(apple_track_id)
        matches.append(MatchedTrack(row, rym_release, method, release_key))
    return matches


def deterministic_sample_with_release_cap(
    matches: Sequence[MatchedTrack],
    *,
    n_songs: int,
    max_tracks_per_release: int,
    seed: int,
) -> list[MatchedTrack]:
    by_release: dict[tuple[str, str], list[MatchedTrack]] = defaultdict(list)
    for match in matches:
        by_release[match.release_key].append(match)

    capped: list[MatchedTrack] = []
    for release_key, release_matches in by_release.items():
        ordered = sorted(
            release_matches,
            key=lambda item: stable_seed(
                seed,
                "within-rym-release",
                release_key[0],
                release_key[1],
                item.sampled_track.get("apple_track_id", ""),
            ),
        )
        capped.extend(ordered[:max_tracks_per_release])

    capped.sort(
        key=lambda item: stable_seed(
            seed,
            "rym-clap-analysis-sample",
            item.sampled_track.get("apple_track_id", ""),
        )
    )
    if len(capped) < n_songs:
        raise ValueError(
            f"Only {len(capped)} matched tracks remain after max_tracks_per_release="
            f"{max_tracks_per_release}; need {n_songs}. Increase the cap or lower --n-songs."
        )
    return capped[:n_songs]


def load_embedding_matrix(matches: Sequence[MatchedTrack], embedding_dir: Path) -> np.ndarray:
    vectors = []
    for match in matches:
        apple_track_id = match.sampled_track["apple_track_id"]
        vector = np.load(embedding_dir / f"{apple_track_id}.npy")
        vectors.append(np.asarray(vector, dtype=np.float64))
    return np.vstack(vectors)


def build_descriptor_matrix(
    matches: Sequence[MatchedTrack],
    *,
    min_descriptor_count: int,
) -> tuple[np.ndarray, list[str], Counter[str], list[list[str]]]:
    row_tags = [descriptor_list(match.rym_release.get("descriptors", "")) for match in matches]
    counts: Counter[str] = Counter()
    for tags in row_tags:
        counts.update(set(tags))
    descriptors = sorted(tag for tag, count in counts.items() if count >= min_descriptor_count)
    if not descriptors:
        raise ValueError(f"No descriptors appear at least {min_descriptor_count} times")
    descriptor_to_col = {tag: index for index, tag in enumerate(descriptors)}
    x = np.zeros((len(matches), len(descriptors)), dtype=np.float64)
    for row_index, tags in enumerate(row_tags):
        for tag in set(tags):
            col = descriptor_to_col.get(tag)
            if col is not None:
                x[row_index, col] = 1.0
    return x, descriptors, counts, row_tags


def release_groups(matches: Sequence[MatchedTrack]) -> np.ndarray:
    """Use the matched RYM release as the CV group to prevent album leakage."""
    return np.asarray(
        [
            f"{normalize(match.rym_release.get('artist_name', ''))}|"
            f"{normalize(match.rym_release.get('release_name', ''))}"
            for match in matches
        ]
    )


def effective_group_folds(groups: Sequence[str], requested_folds: int) -> int:
    n_groups = len(set(groups))
    folds = min(requested_folds, n_groups)
    if folds < 2:
        raise ValueError("Grouped cross-validation requires at least two unique RYM releases")
    return folds


def alpha_grid(alpha_min: float, alpha_max: float, n_alphas: int) -> np.ndarray:
    if alpha_min <= 0 or alpha_max <= 0:
        raise ValueError("Elastic Net alpha bounds must be positive")
    if alpha_min >= alpha_max:
        raise ValueError("--alpha-min must be smaller than --alpha-max")
    return np.logspace(math.log10(alpha_min), math.log10(alpha_max), n_alphas)


def build_elastic_net_search(
    *,
    cv: GroupKFold,
    seed: int,
    alphas: Sequence[float],
    l1_ratios: Sequence[float],
) -> GridSearchCV:
    model = make_pipeline(
        StandardScaler(),
        ElasticNet(random_state=seed, max_iter=100_000, tol=1e-4),
    )
    return GridSearchCV(
        model,
        param_grid={
            "elasticnet__alpha": list(alphas),
            "elasticnet__l1_ratio": list(l1_ratios),
        },
        scoring="r2",
        cv=cv,
        refit=True,
    )


def grouped_out_of_fold_r2(
    descriptor_matrix: np.ndarray,
    y: np.ndarray,
    groups: np.ndarray,
    *,
    cv_folds: int,
    seed: int,
    alphas: Sequence[float],
    l1_ratios: Sequence[float],
) -> tuple[float, list[float], list[dict[str, Any]]]:
    """Nested grouped CV R².

    The outer grouped split estimates performance on unseen releases. The inner
    grouped split tunes alpha/l1_ratio on the training releases only.
    """
    folds = effective_group_folds(groups, cv_folds)
    outer_cv = GroupKFold(n_splits=folds)
    oof_predictions = np.full_like(y, fill_value=np.nan, dtype=np.float64)
    fold_scores: list[float] = []
    fold_summaries: list[dict[str, Any]] = []

    for fold_index, (train_idx, test_idx) in enumerate(
        outer_cv.split(descriptor_matrix, y, groups=groups),
        start=1,
    ):
        train_groups = groups[train_idx]
        inner_folds = effective_group_folds(train_groups, cv_folds)
        search = build_elastic_net_search(
            cv=GroupKFold(n_splits=inner_folds),
            seed=seed,
            alphas=alphas,
            l1_ratios=l1_ratios,
        )
        search.fit(descriptor_matrix[train_idx], y[train_idx], groups=train_groups)
        predictions = search.predict(descriptor_matrix[test_idx])
        oof_predictions[test_idx] = predictions
        fold_r2 = float(r2_score(y[test_idx], predictions))
        fold_scores.append(fold_r2)
        fold_summaries.append(
            {
                "fold": fold_index,
                "n_train": int(len(train_idx)),
                "n_test": int(len(test_idx)),
                "n_train_groups": int(len(set(train_groups))),
                "n_test_groups": int(len(set(groups[test_idx]))),
                "r2": fold_r2,
                "best_alpha": float(search.best_params_["elasticnet__alpha"]),
                "best_l1_ratio": float(search.best_params_["elasticnet__l1_ratio"]),
                "inner_grouped_cv_mean_r2": float(search.best_score_),
            }
        )

    if np.isnan(oof_predictions).any():
        raise ValueError("Some rows did not receive an out-of-fold prediction")
    return float(r2_score(y, oof_predictions)), fold_scores, fold_summaries


def fit_elastic_net_for_pcs(
    descriptor_matrix: np.ndarray,
    pc_scores: np.ndarray,
    descriptors: Sequence[str],
    descriptor_counts: Counter[str],
    explained_variance: Sequence[float],
    groups: np.ndarray,
    *,
    n_pcs: int,
    cv_folds: int,
    seed: int,
    alpha_min: float,
    alpha_max: float,
    n_alphas: int,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    coefficient_rows: list[dict[str, Any]] = []
    model_summaries: list[dict[str, Any]] = []
    folds = effective_group_folds(groups, cv_folds)
    cv = GroupKFold(n_splits=folds)
    l1_ratios = [0.1, 0.25, 0.5, 0.75, 0.9, 1.0]
    alphas = alpha_grid(alpha_min, alpha_max, n_alphas)

    for pc_index in range(n_pcs):
        y = pc_scores[:, pc_index]
        y_std = (y - y.mean()) / (y.std(ddof=0) or 1.0)
        grouped_cv_r2, fold_scores, fold_summaries = grouped_out_of_fold_r2(
            descriptor_matrix,
            y_std,
            groups,
            cv_folds=folds,
            seed=seed,
            alphas=alphas,
            l1_ratios=l1_ratios,
        )

        final_search = build_elastic_net_search(
            cv=cv,
            seed=seed,
            alphas=alphas,
            l1_ratios=l1_ratios,
        )
        final_search.fit(descriptor_matrix, y_std, groups=groups)
        final_model = final_search.best_estimator_
        elastic = final_model.named_steps["elasticnet"]
        coefficients = elastic.coef_
        predictions = final_model.predict(descriptor_matrix)
        in_sample_r2 = r2_score(y_std, predictions)
        selected_count = int(np.sum(np.abs(coefficients) > 1e-10))
        mean_fold_r2 = float(np.mean(fold_scores))
        std_fold_r2 = float(np.std(fold_scores, ddof=0))

        model_summaries.append(
            {
                "PC": pc_index + 1,
                "explained_variance_ratio": float(explained_variance[pc_index]),
                "alpha": float(final_search.best_params_["elasticnet__alpha"]),
                "l1_ratio": float(final_search.best_params_["elasticnet__l1_ratio"]),
                "cv_folds": folds,
                "cv_grouping": "RYM artist + release",
                "candidate_l1_ratios": l1_ratios,
                "candidate_alpha_min": float(min(alphas)),
                "candidate_alpha_max": float(max(alphas)),
                "candidate_n_alphas": len(alphas),
                "selected_descriptors": selected_count,
                "grouped_cv_r2": grouped_cv_r2,
                "grouped_cv_fold_r2_mean": mean_fold_r2,
                "grouped_cv_fold_r2_std": std_fold_r2,
                "grouped_cv_fold_summaries": fold_summaries,
                "final_model_grouped_cv_mean_r2": float(final_search.best_score_),
                "in_sample_r2": float(in_sample_r2),
            }
        )

        for descriptor, coefficient in zip(descriptors, coefficients):
            coefficient_rows.append(
                {
                    "PC": pc_index + 1,
                    "descriptor": descriptor,
                    "display_descriptor": display_descriptor(descriptor),
                    "coefficient": f"{float(coefficient):.8f}",
                    "abs_coefficient": f"{abs(float(coefficient)):.8f}",
                    "descriptor_count": descriptor_counts[descriptor],
                    "selected": int(abs(float(coefficient)) > 1e-10),
                    "alpha": f"{float(final_search.best_params_['elasticnet__alpha']):.8f}",
                    "l1_ratio": f"{float(final_search.best_params_['elasticnet__l1_ratio']):.4f}",
                    "cv_folds": folds,
                    "grouped_cv_r2": f"{grouped_cv_r2:.8f}",
                    "grouped_cv_fold_r2_mean": f"{mean_fold_r2:.8f}",
                    "in_sample_r2": f"{float(in_sample_r2):.8f}",
                }
            )

    return coefficient_rows, model_summaries


def match_output_rows(matches: Sequence[MatchedTrack]) -> list[dict[str, Any]]:
    rows = []
    for index, match in enumerate(matches, 1):
        source = match.sampled_track
        rym = match.rym_release
        rows.append(
            {
                "analysis_id": f"R{index:04d}",
                "apple_track_id": source.get("apple_track_id", ""),
                "label": source.get("label", ""),
                "year": source.get("year", ""),
                "artist": source.get("artist", ""),
                "track_title": source.get("track_title", ""),
                "release_title": source.get("release_title", ""),
                "apple_artist": source.get("apple_artist", ""),
                "apple_track": source.get("apple_track", ""),
                "apple_collection": source.get("apple_collection", ""),
                "rym_artist": rym.get("artist_name", ""),
                "rym_release": rym.get("release_name", ""),
                "rym_release_type": rym.get("release_type", ""),
                "rym_descriptors": rym.get("descriptors", ""),
                "match_method": match.match_method,
            }
        )
    return rows


def score_output_rows(
    matches: Sequence[MatchedTrack],
    pc_scores: np.ndarray,
    explained_variance: Sequence[float],
    *,
    n_pcs: int,
) -> list[dict[str, Any]]:
    rows = []
    for index, match in enumerate(matches, 1):
        source = match.sampled_track
        for pc_index in range(n_pcs):
            rows.append(
                {
                    "analysis_id": f"R{index:04d}",
                    "apple_track_id": source.get("apple_track_id", ""),
                    "artist": source.get("artist", ""),
                    "track_title": source.get("track_title", ""),
                    "release_title": source.get("release_title", ""),
                    "PC": pc_index + 1,
                    "score": f"{float(pc_scores[index - 1, pc_index]):.8f}",
                    "explained_variance_ratio": f"{float(explained_variance[pc_index]):.8f}",
                }
            )
    return rows


def selected_coefficients(
    coefficient_rows: Sequence[Mapping[str, Any]],
    pc: int,
) -> list[tuple[str, float, int]]:
    rows = []
    for row in coefficient_rows:
        if int(row["PC"]) != pc:
            continue
        coefficient = float(row["coefficient"])
        if math.isclose(coefficient, 0.0, abs_tol=1e-10):
            continue
        rows.append((str(row["display_descriptor"]), coefficient, int(row["descriptor_count"])))
    rows.sort(key=lambda item: abs(item[1]), reverse=True)
    return rows


def top_signed(
    coefficients: Sequence[tuple[str, float, int]],
    *,
    sign: int,
    top_n: int,
) -> list[tuple[str, float, int]]:
    values = [item for item in coefficients if item[1] * sign > 0]
    values.sort(key=lambda item: abs(item[1]), reverse=True)
    return values[:top_n]


def draw_coefficient_cloud(
    ax: plt.Axes,
    coefficients: Sequence[tuple[str, float, int]],
    *,
    pc: int,
    explained_variance: float,
    max_words_each_side: int = 13,
) -> None:
    positive = top_signed(coefficients, sign=1, top_n=max_words_each_side)
    negative = top_signed(coefficients, sign=-1, top_n=max_words_each_side)
    chosen = negative + positive
    if not chosen:
        ax.text(0.5, 0.5, "No selected descriptors", ha="center", va="center")
        ax.axis("off")
        return

    max_abs = max(abs(coef) for _, coef, _ in chosen) or 1.0
    y_slots = [0.52, 0.72, 0.32, 0.86, 0.18, 0.62, 0.42, 0.78, 0.26, 0.10, 0.92, 0.58, 0.38]
    x_slots = [0.23, 0.35, 0.12, 0.43, 0.28, 0.16, 0.39, 0.06, 0.32, 0.20, 0.46, 0.10, 0.41]

    def place(words: Sequence[tuple[str, float, int]], *, right: bool) -> None:
        for rank, (word, coefficient, count) in enumerate(words):
            size = 9 + 22 * (abs(coefficient) / max_abs) ** 0.75
            x = 0.52 + x_slots[rank % len(x_slots)] if right else 0.48 - x_slots[rank % len(x_slots)]
            y = y_slots[rank % len(y_slots)]
            ax.text(
                x,
                y,
                word,
                fontsize=size,
                color="#4F67B1" if right else "#B65C55",
                ha="center",
                va="center",
                alpha=0.94,
                weight="bold" if rank < 3 else "normal",
            )

    place(negative, right=False)
    place(positive, right=True)
    ax.axvline(0.5, color="0.75", linewidth=0.8)
    ax.text(0.08, 0.03, "(-)", fontsize=11, ha="center", va="bottom")
    ax.text(0.92, 0.03, "(+)", fontsize=11, ha="center", va="bottom")
    ax.set_title(f"Wordcloud-style descriptors (PC {pc}, {explained_variance:.1%} var.)", fontsize=11)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")


def draw_top_bars(
    ax: plt.Axes,
    coefficients: Sequence[tuple[str, float, int]],
    *,
    pc: int,
    top_n: int = 5,
) -> None:
    positive = list(reversed(top_signed(coefficients, sign=1, top_n=top_n)))
    negative = top_signed(coefficients, sign=-1, top_n=top_n)
    values = negative + positive
    if not values:
        ax.text(0.5, 0.5, "No selected descriptors", ha="center", va="center")
        ax.axis("off")
        return

    labels = [item[0] for item in values]
    coefs = [item[1] for item in values]
    colors = ["#C99691" if value < 0 else "#7C8FD0" for value in coefs]
    y = np.arange(len(values))
    ax.barh(y, coefs, color=colors, edgecolor="0.35", linewidth=0.5)
    ax.axvline(0, color="0.25", linewidth=0.8)
    ax.set_yticks(y)
    ax.set_yticklabels(labels, fontsize=8)
    ax.set_xlabel("Elastic Net coefficient", fontsize=8)
    ax.set_ylabel("RYM descriptor", fontsize=8)
    ax.tick_params(axis="x", labelsize=7)
    ax.tick_params(axis="y", labelsize=7)
    ax.grid(axis="x", color="0.88", linewidth=0.6)
    ax.set_axisbelow(True)
    ax.set_title(f"Top 5 descriptor coefficients (PC {pc})", fontsize=11)


def write_figure(
    coefficient_rows: Sequence[Mapping[str, Any]],
    explained_variance: Sequence[float],
    output_path: Path,
) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(2, 2, figsize=(12, 8.4))
    for pc, col in [(1, 0), (2, 1)]:
        coefficients = selected_coefficients(coefficient_rows, pc)
        draw_coefficient_cloud(
            axes[0, col],
            coefficients,
            pc=pc,
            explained_variance=float(explained_variance[pc - 1]),
        )
        draw_top_bars(axes[1, col], coefficients, pc=pc)
    fig.suptitle("RYM descriptors as interpretation of CLAP principal components", y=0.995, fontsize=15)
    fig.tight_layout(rect=[0, 0, 1, 0.965])
    fig.savefig(output_path, dpi=220, bbox_inches="tight")
    plt.close(fig)


def write_explained_variance_figure(explained_variance: Sequence[float], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    pcs = np.arange(1, len(explained_variance) + 1)
    fig, ax = plt.subplots(figsize=(8, 4.8))
    ax.bar(pcs, np.asarray(explained_variance) * 100, color="#7C8FD0", edgecolor="0.35", linewidth=0.6)
    ax.plot(pcs, np.cumsum(explained_variance) * 100, marker="o", color="#B65C55", linewidth=1.5, label="Cumulative")
    ax.set_xlabel("Principal component")
    ax.set_ylabel("Explained variance (%)")
    ax.set_title("Explained variance of CLAP PCA components")
    ax.set_xticks(pcs)
    ax.grid(axis="y", color="0.88", linewidth=0.6)
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(output_path, dpi=220, bbox_inches="tight")
    plt.close(fig)


def top_summary_table(
    coefficient_rows: Sequence[Mapping[str, Any]],
    *,
    n_pcs: int,
    top_n: int,
) -> list[dict[str, Any]]:
    output = []
    for pc in range(1, n_pcs + 1):
        coefficients = selected_coefficients(coefficient_rows, pc)
        output.append(
            {
                "PC": pc,
                "positive_descriptors": ", ".join(word for word, _, _ in top_signed(coefficients, sign=1, top_n=top_n)),
                "negative_descriptors": ", ".join(word for word, _, _ in top_signed(coefficients, sign=-1, top_n=top_n)),
            }
        )
    return output


def run(args: argparse.Namespace) -> dict[str, Any]:
    sampled_tracks = read_csv(Path(args.sampled_tracks))
    rym_rows = read_csv(Path(args.rym_csv))
    embedding_dir = Path(args.embedding_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    strict_matches = collect_strict_matches(sampled_tracks, rym_rows, embedding_dir)
    selected = deterministic_sample_with_release_cap(
        strict_matches,
        n_songs=args.n_songs,
        max_tracks_per_release=args.max_tracks_per_release,
        seed=args.seed,
    )

    embeddings = load_embedding_matrix(selected, embedding_dir)
    descriptor_matrix, descriptors, descriptor_counts, _ = build_descriptor_matrix(
        selected,
        min_descriptor_count=args.min_descriptor_count,
    )

    n_pcs = min(args.n_pcs, embeddings.shape[0], embeddings.shape[1])
    pca = PCA(n_components=n_pcs, random_state=args.seed)
    pc_scores = pca.fit_transform(embeddings)
    explained_variance = pca.explained_variance_ratio_
    groups = release_groups(selected)

    coefficient_rows, model_summaries = fit_elastic_net_for_pcs(
        descriptor_matrix,
        pc_scores,
        descriptors,
        descriptor_counts,
        explained_variance,
        groups,
        n_pcs=n_pcs,
        cv_folds=args.cv_folds,
        seed=args.seed,
        alpha_min=args.alpha_min,
        alpha_max=args.alpha_max,
        n_alphas=args.n_alphas,
    )

    match_rows = match_output_rows(selected)
    score_rows = score_output_rows(selected, pc_scores, explained_variance, n_pcs=n_pcs)
    top_rows = top_summary_table(coefficient_rows, n_pcs=n_pcs, top_n=args.summary_top_n)

    write_csv(output_dir / "rym_clap_matched_500.csv", match_rows, MATCH_FIELDS)
    write_csv(output_dir / "rym_clap_pca_scores_500.csv", score_rows, SCORE_FIELDS)
    write_csv(output_dir / "rym_clap_descriptor_coefficients.csv", coefficient_rows, COEFFICIENT_FIELDS)
    write_csv(output_dir / "rym_clap_model_validation.csv", model_summaries, MODEL_SUMMARY_FIELDS)
    write_csv(output_dir / "rym_clap_pc_descriptor_summary.csv", top_rows, ["PC", "positive_descriptors", "negative_descriptors"])

    figure_path = output_dir / "rym_clap_pc1_pc2_wordclouds_and_bars.png"
    variance_path = output_dir / "rym_clap_explained_variance.png"
    write_figure(coefficient_rows, explained_variance, figure_path)
    write_explained_variance_figure(explained_variance, variance_path)

    match_method_counts = Counter(match.match_method for match in selected)
    selected_release_keys = {match.release_key for match in selected}
    summary = {
        "method": (
            "PCA on CLAP audio embeddings; RYM descriptors converted to binary variables; "
            "each PC regressed on descriptors with Elastic Net. Reported validation R^2 "
            "uses grouped cross-validation by RYM release/album so songs from the same "
            "release remain in the same fold."
        ),
        "sampled_tracks_rows": len(sampled_tracks),
        "rym_rows": len(rym_rows),
        "strict_unique_embedded_matches": len(strict_matches),
        "n_songs": len(selected),
        "unique_rym_releases": len(selected_release_keys),
        "max_tracks_per_release": args.max_tracks_per_release,
        "cv_grouping": "RYM artist + release",
        "cv_folds": effective_group_folds(groups, args.cv_folds),
        "match_methods_in_selected_sample": dict(match_method_counts.most_common()),
        "embedding_dimensions": int(embeddings.shape[1]),
        "n_pcs": n_pcs,
        "explained_variance_ratio": [float(value) for value in explained_variance],
        "cumulative_explained_variance_ratio": [float(value) for value in np.cumsum(explained_variance)],
        "min_descriptor_count": args.min_descriptor_count,
        "n_descriptors_before_filter": len({tag for match in selected for tag in descriptor_list(match.rym_release.get("descriptors", ""))}),
        "n_descriptors_after_filter": len(descriptors),
        "elastic_net_models": model_summaries,
        "top_descriptor_summary": top_rows,
        "outputs": {
            "matched_tracks": str((output_dir / "rym_clap_matched_500.csv").resolve()),
            "pca_scores": str((output_dir / "rym_clap_pca_scores_500.csv").resolve()),
            "coefficients": str((output_dir / "rym_clap_descriptor_coefficients.csv").resolve()),
            "model_validation": str((output_dir / "rym_clap_model_validation.csv").resolve()),
            "pc_descriptor_summary": str((output_dir / "rym_clap_pc_descriptor_summary.csv").resolve()),
            "pc1_pc2_figure": str(figure_path.resolve()),
            "explained_variance_figure": str(variance_path.resolve()),
        },
    }
    (output_dir / "rym_clap_interpretability_summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    return summary


def parser() -> argparse.ArgumentParser:
    root = argparse.ArgumentParser(description="Interpret CLAP PCA components using RYM descriptors and Elastic Net.")
    root.add_argument("--rym-csv", default="data/rym_clean1.csv")
    root.add_argument("--sampled-tracks", default="data/sampled_tracks.csv")
    root.add_argument("--embedding-dir", default="data/embeddings_full30")
    root.add_argument("--output-dir", default="outputs/rym_clap_interpretability")
    root.add_argument("--n-songs", type=int, default=500)
    root.add_argument("--max-tracks-per-release", type=int, default=2)
    root.add_argument("--min-descriptor-count", type=int, default=15)
    root.add_argument("--n-pcs", type=int, default=5)
    root.add_argument("--cv-folds", type=int, default=5)
    root.add_argument("--alpha-min", type=float, default=0.001)
    root.add_argument("--alpha-max", type=float, default=1.0)
    root.add_argument("--n-alphas", type=int, default=40)
    root.add_argument("--summary-top-n", type=int, default=8)
    root.add_argument("--seed", type=int, default=20260627)
    return root


def main() -> None:
    args = parser().parse_args()
    summary = run(args)
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
