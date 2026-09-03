"""Decision-specific propensity-neighborhood experiment.

The method retrieves peers specifically for the preliminary model's top-two
programs, estimates a local residual correction to that pair's propensity
margin, and swaps the winner only when the corrected margin changes sign.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import sparse
from scipy.stats import t as student_t
from sklearn.decomposition import TruncatedSVD
from sklearn.metrics import pairwise_distances
from sklearn.preprocessing import StandardScaler

from experiment import (
    DEFAULT_SEEDS,
    decision_metrics,
    fit_base_model,
    load_chronological_data,
    make_transformer,
    nearest_indices,
    parse_seeds,
)


K_GRID = (10, 25, 50, 100)
SHRINKAGE_GRID = (0.0, 20.0, 50.0, 100.0, 200.0, 1.0e9)
GATE_GRID = (0.0, 0.5, 1.0, 1.5, 2.0)
GEOMETRIES = (
    "full_propensity",
    "pair_propensity",
    "pair_propensity_same_winner",
    "pair_propensity_embedding",
)


def normalize_distance(distance: np.ndarray) -> np.ndarray:
    finite = distance[np.isfinite(distance) & (distance > 0)]
    scale = float(np.median(finite)) if len(finite) else 1.0
    return distance / max(scale, 1.0e-12)


def decision_distances(
    query_base: np.ndarray,
    reference_base: np.ndarray,
    embedding_distance: np.ndarray,
    exclude_self: bool,
) -> dict[str, np.ndarray]:
    query_count = len(query_base)
    reference_count = len(reference_base)
    top_two = np.argsort(-query_base, axis=1)[:, :2]
    reference_winner = reference_base.argmax(axis=1)
    full = pairwise_distances(query_base, reference_base, metric="chebyshev")
    pair = np.empty((query_count, reference_count), dtype=np.float64)
    same_winner = np.empty_like(pair)
    for row, (first, second) in enumerate(top_two):
        coordinate_gap = np.maximum(
            np.abs(reference_base[:, first] - query_base[row, first]),
            np.abs(reference_base[:, second] - query_base[row, second]),
        )
        target_margin = query_base[row, first] - query_base[row, second]
        reference_margin = reference_base[:, first] - reference_base[:, second]
        pair[row] = coordinate_gap + np.abs(reference_margin - target_margin)
        penalty = (reference_winner != first).astype(np.float64)
        same_winner[row] = pair[row] + penalty * (pair[row].max() + 1.0)
    full = normalize_distance(full)
    pair = normalize_distance(pair)
    same_winner = normalize_distance(same_winner)
    embedding = normalize_distance(embedding_distance)
    combined = pair + embedding
    if exclude_self:
        for distance in (full, pair, same_winner, combined):
            np.fill_diagonal(distance, np.inf)
    return {
        "full_propensity": full,
        "pair_propensity": pair,
        "pair_propensity_same_winner": same_winner,
        "pair_propensity_embedding": combined,
    }


def margin_corrected_prediction(
    query_base: np.ndarray,
    reference_base: np.ndarray,
    reference_y: np.ndarray,
    distance: np.ndarray,
    k: int,
    shrinkage: float,
    gate_multiplier: float = 0.0,
) -> tuple[np.ndarray, np.ndarray]:
    k_effective = min(k, distance.shape[1])
    neighbors = nearest_indices(distance, k_effective)[:, :k_effective]
    top_two = np.argsort(-query_base, axis=1)[:, :2]
    prediction = query_base.copy()
    correction = np.empty(len(query_base), dtype=np.float64)
    reference_residual = reference_y - reference_base
    weight = k_effective / (k_effective + shrinkage)
    for row, (first, second) in enumerate(top_two):
        peer = neighbors[row]
        pair_residual = (
            reference_residual[peer, first] - reference_residual[peer, second]
        )
        correction[row] = weight * pair_residual.mean()
        corrected_margin = (
            query_base[row, first] - query_base[row, second] + correction[row]
        )
        standard_error = (
            weight * pair_residual.std(ddof=1) / np.sqrt(k_effective)
            if k_effective > 1
            else np.inf
        )
        # The preliminary winner is changed only when the locally corrected
        # margin is negative by more than the requested uncertainty multiple.
        if corrected_margin < -gate_multiplier * standard_error:
            midpoint = 0.5 * (query_base[row, first] + query_base[row, second])
            prediction[row, first] = midpoint + 0.5 * corrected_margin
            prediction[row, second] = midpoint - 0.5 * corrected_margin
    return prediction, correction


def top_k_diagnostic(base: np.ndarray, truth: np.ndarray, k: int) -> dict[str, float]:
    candidates = np.argsort(-base, axis=1)[:, :k]
    rows = np.arange(len(base))[:, None]
    candidate_truth = truth[rows, candidates]
    true_best = truth.argmax(axis=1)
    contained = (candidates == true_best[:, None]).any(axis=1)
    base_choice = base.argmax(axis=1)
    base_wrong = base_choice != true_best
    return {
        "recall": float(contained.mean()),
        "recoverable_error_fraction": float(np.mean(base_wrong & contained)),
        "conditional_recoverability_given_base_error": float(
            np.mean(contained[base_wrong]) if np.any(base_wrong) else 0.0
        ),
        "oracle_regret": float(np.mean(truth.max(axis=1) - candidate_truth.max(axis=1))),
    }


def override_diagnostic(
    base: np.ndarray, prediction: np.ndarray, truth: np.ndarray
) -> dict[str, float]:
    base_choice = base.argmax(axis=1)
    choice = prediction.argmax(axis=1)
    changed = choice != base_choice
    rows = np.arange(len(base))
    base_regret = truth.max(axis=1) - truth[rows, base_choice]
    new_regret = truth.max(axis=1) - truth[rows, choice]
    delta = new_regret - base_regret
    return {
        "override_rate": float(changed.mean()),
        "beneficial_override_rate": float(np.mean(delta[changed] < 0)) if np.any(changed) else 0.0,
        "harmful_override_rate": float(np.mean(delta[changed] > 0)) if np.any(changed) else 0.0,
        "mean_regret_change_on_overrides": float(np.mean(delta[changed])) if np.any(changed) else 0.0,
        "mean_regret_change_all_users": float(np.mean(delta)),
    }


def run_split(
    data: dict[str, object], seed: int, embedding_dimensions: int, folds: int
) -> dict[str, object]:
    started = time.perf_counter()
    users = data["users"]
    future_rates = data["future_rates"]
    history_matrix = data["history_matrix"]
    model_columns = data["model_columns"]
    assert isinstance(users, pd.DataFrame)
    assert isinstance(future_rates, pd.DataFrame)
    assert sparse.issparse(history_matrix)
    assert isinstance(model_columns, list)

    rng = np.random.default_rng(seed)
    shuffled = np.arange(len(users))
    rng.shuffle(shuffled)
    n_estimation = int(0.60 * len(shuffled))
    n_tuning = int(0.25 * len(shuffled))
    estimation = shuffled[:n_estimation]
    tuning = shuffled[n_estimation : n_estimation + n_tuning]
    test = shuffled[n_estimation + n_tuning :]
    y = future_rates.to_numpy()
    y_estimation, y_tuning, y_test = y[estimation], y[tuning], y[test]

    svd = TruncatedSVD(n_components=embedding_dimensions, random_state=seed)
    svd.fit(history_matrix[estimation])
    embedding = svd.transform(history_matrix)
    embedding = StandardScaler().fit(embedding[estimation]).transform(embedding)
    transformer = make_transformer(users.iloc[estimation], model_columns)
    transformer.fit(users.iloc[estimation][model_columns])
    static = np.asarray(transformer.transform(users[model_columns]))
    features = np.hstack([static, embedding])
    base_oof, base_tuning, base_test = fit_base_model(
        features, y, estimation, tuning, test, seed, folds
    )

    embedding_distance_tuning = pairwise_distances(
        embedding[tuning], embedding[estimation], metric="cosine"
    )
    embedding_distance_test = pairwise_distances(
        embedding[test], embedding[estimation], metric="cosine"
    )
    tuning_distances = decision_distances(
        base_tuning, base_oof, embedding_distance_tuning, exclude_self=False
    )
    test_distances = decision_distances(
        base_test, base_oof, embedding_distance_test, exclude_self=False
    )

    tuning_candidates = []
    for geometry in GEOMETRIES:
        for k in K_GRID:
            for shrinkage in SHRINKAGE_GRID:
                for gate_multiplier in GATE_GRID:
                    prediction, _ = margin_corrected_prediction(
                        base_tuning,
                        base_oof,
                        y_estimation,
                        tuning_distances[geometry],
                        k,
                        shrinkage,
                        gate_multiplier,
                    )
                    metric = decision_metrics(
                        "candidate", prediction, y_tuning, np.clip(prediction, 0.0, 1.0)
                    )
                    tuning_candidates.append(
                        (
                            float(metric["top1_regret"]),
                            -float(metric["top1_hit_rate"]),
                            geometry,
                            k,
                            shrinkage,
                            gate_multiplier,
                        )
                    )
    selected = min(tuning_candidates)
    _, _, geometry, selected_k, selected_shrinkage, selected_gate = selected
    test_prediction, test_correction = margin_corrected_prediction(
        base_test,
        base_oof,
        y_estimation,
        test_distances[geometry],
        selected_k,
        selected_shrinkage,
        selected_gate,
    )
    fixed_prediction, _ = margin_corrected_prediction(
        base_test,
        base_oof,
        y_estimation,
        test_distances["pair_propensity_embedding"],
        50,
        50.0,
        1.0,
    )
    metrics = [
        decision_metrics("base_propensity", base_test, y_test, base_test),
        decision_metrics(
            "tuned_decision_neighborhood",
            test_prediction,
            y_test,
            np.clip(test_prediction, 0.0, 1.0),
        ),
        decision_metrics(
            "fixed_decision_neighborhood",
            fixed_prediction,
            y_test,
            np.clip(fixed_prediction, 0.0, 1.0),
        ),
    ]
    return {
        "seed": seed,
        "runtime_seconds": time.perf_counter() - started,
        "users": len(users),
        "estimation_users": len(estimation),
        "tuning_users": len(tuning),
        "test_users": len(test),
        "top2_diagnostic": top_k_diagnostic(base_test, y_test, 2),
        "top3_diagnostic": top_k_diagnostic(base_test, y_test, 3),
        "selected_geometry": geometry,
        "selected_k": selected_k,
        "selected_shrinkage": selected_shrinkage,
        "selected_gate_multiplier": selected_gate,
        "mean_absolute_margin_correction": float(np.mean(np.abs(test_correction))),
        "tuned_override": override_diagnostic(base_test, test_prediction, y_test),
        "fixed_override": override_diagnostic(base_test, fixed_prediction, y_test),
        "tuning_candidates": [
            {
                "top1_regret": candidate[0],
                "top1_hit_rate": -candidate[1],
                "geometry": candidate[2],
                "k": candidate[3],
                "shrinkage": candidate[4],
                "gate_multiplier": candidate[5],
            }
            for candidate in tuning_candidates
        ],
        "metrics": metrics,
    }


def interval(values: np.ndarray) -> list[float]:
    if len(values) < 2:
        return [float("nan"), float("nan")]
    half = float(
        student_t.ppf(0.975, len(values) - 1)
        * values.std(ddof=1)
        / np.sqrt(len(values))
    )
    return [float(values.mean() - half), float(values.mean() + half)]


def aggregate(
    rows: pd.DataFrame, summaries: list[dict[str, object]]
) -> tuple[pd.DataFrame, dict[str, object]]:
    columns = ["rmse", "top1_regret", "top1_hit_rate", "ndcg", "pairwise_log_loss"]
    means = rows.groupby("method", sort=False)[columns].mean().reset_index()
    pivot = rows.pivot(index="seed", columns="method", values="top1_regret")
    comparisons = {}
    for method in ("tuned_decision_neighborhood", "fixed_decision_neighborhood"):
        delta = pivot[method] - pivot["base_propensity"]
        comparisons[method] = {
            "mean_regret_change_vs_base": float(delta.mean()),
            "paired_95pct_interval": interval(delta.to_numpy()),
            "regret_wins": int((delta < 0).sum()),
            "splits": int(len(delta)),
        }
    return means, {
        "splits": int(rows.seed.nunique()),
        "comparisons": comparisons,
        "mean_top2_recall": float(
            np.mean([summary["top2_diagnostic"]["recall"] for summary in summaries])
        ),
        "mean_top3_recall": float(
            np.mean([summary["top3_diagnostic"]["recall"] for summary in summaries])
        ),
        "mean_top2_oracle_regret": float(
            np.mean(
                [summary["top2_diagnostic"]["oracle_regret"] for summary in summaries]
            )
        ),
        "mean_top3_oracle_regret": float(
            np.mean(
                [summary["top3_diagnostic"]["oracle_regret"] for summary in summaries]
            )
        ),
        "mean_tuned_override_rate": float(
            np.mean([summary["tuned_override"]["override_rate"] for summary in summaries])
        ),
        "mean_tuned_beneficial_override_rate": float(
            np.mean(
                [
                    summary["tuned_override"]["beneficial_override_rate"]
                    for summary in summaries
                ]
            )
        ),
        "selected_geometries": [summary["selected_geometry"] for summary in summaries],
        "selected_k": [summary["selected_k"] for summary in summaries],
        "selected_shrinkage": [summary["selected_shrinkage"] for summary in summaries],
        "selected_gate_multiplier": [
            summary["selected_gate_multiplier"] for summary in summaries
        ],
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--data-root", type=Path, default=Path("data/kuairec/extracted")
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("propensity-neighborhoods-v2/decision-boundary-results"),
    )
    parser.add_argument("--seeds", type=parse_seeds, default=DEFAULT_SEEDS)
    parser.add_argument("--categories", type=int, default=12)
    parser.add_argument("--history-fraction", type=float, default=0.60)
    parser.add_argument("--embedding-dimensions", type=int, default=32)
    parser.add_argument("--folds", type=int, default=5)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)

    overall_started = time.perf_counter()
    print("Loading and constructing chronological KuaiRec data...", flush=True)
    data = load_chronological_data(
        args.data_root, args.categories, args.history_fraction
    )
    all_rows = []
    summaries = []
    for seed in args.seeds:
        print(f"Running decision-boundary seed {seed}...", flush=True)
        summary = run_split(data, seed, args.embedding_dimensions, args.folds)
        directory = args.output / f"seed_{seed}"
        directory.mkdir(parents=True, exist_ok=True)
        (directory / "summary.json").write_text(
            json.dumps(summary, indent=2), encoding="utf-8"
        )
        frame = pd.DataFrame(summary["metrics"])
        frame.insert(0, "seed", seed)
        frame.to_csv(directory / "metrics.csv", index=False)
        all_rows.append(frame)
        summaries.append(summary)
        print(
            f"seed={seed} geometry={summary['selected_geometry']} "
            f"k={summary['selected_k']} shrinkage={summary['selected_shrinkage']} "
            f"gate={summary['selected_gate_multiplier']} "
            f"runtime={summary['runtime_seconds']:.1f}s",
            flush=True,
        )

    rows = pd.concat(all_rows, ignore_index=True)
    rows.to_csv(args.output / "split_metrics.csv", index=False)
    means, aggregate_summary = aggregate(rows, summaries)
    means.to_csv(args.output / "aggregate_metrics.csv", index=False)
    aggregate_summary.update(
        {
            "seeds": list(args.seeds),
            "runtime_seconds": time.perf_counter() - overall_started,
        }
    )
    (args.output / "aggregate_summary.json").write_text(
        json.dumps(aggregate_summary, indent=2), encoding="utf-8"
    )
    print(means.to_string(index=False), flush=True)
    print(json.dumps(aggregate_summary, indent=2), flush=True)


if __name__ == "__main__":
    main()
