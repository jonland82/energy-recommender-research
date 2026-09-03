"""Quick test of neighborhood-informed behavioral embeddings.

Neighborhoods are constructed only from pre-decision SVD embeddings. The final
propensity model is still ridge regression and recommendations are still the
descending order of its fitted propensities.
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
from sklearn.linear_model import Ridge
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


K_GRID = (10, 25, 50, 100, 200)
MIX_GRID = (0.10, 0.25, 0.50, 0.75)


def neighbor_mean(
    reference: np.ndarray, distance: np.ndarray, k: int
) -> tuple[np.ndarray, np.ndarray]:
    k_effective = min(k, distance.shape[1])
    neighbors = nearest_indices(distance, k_effective)[:, :k_effective]
    local = reference[neighbors]
    mean = local.mean(axis=1)
    mean_distance = np.take_along_axis(distance, neighbors, axis=1).mean(axis=1)
    return mean, mean_distance


def fit_and_score(
    estimation_x: np.ndarray,
    tuning_x: np.ndarray,
    test_x: np.ndarray,
    y_estimation: np.ndarray,
    y_tuning: np.ndarray,
) -> tuple[np.ndarray, dict[str, float | str]]:
    model = Ridge(alpha=10.0).fit(estimation_x, y_estimation)
    tuning_prediction = np.clip(model.predict(tuning_x), 0.0, 1.0)
    test_prediction = np.clip(model.predict(test_x), 0.0, 1.0)
    metric = decision_metrics(
        "candidate", tuning_prediction, y_tuning, tuning_prediction
    )
    return test_prediction, metric


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

    base_features = np.hstack([static, embedding])
    _, _, base_test = fit_base_model(
        base_features, y, estimation, tuning, test, seed, folds
    )

    distance_estimation = pairwise_distances(
        embedding[estimation], embedding[estimation], metric="cosine"
    )
    np.fill_diagonal(distance_estimation, np.inf)
    distance_tuning = pairwise_distances(
        embedding[tuning], embedding[estimation], metric="cosine"
    )
    distance_test = pairwise_distances(
        embedding[test], embedding[estimation], metric="cosine"
    )

    candidates = {"smoothed": [], "augmented": []}
    diagnostics = []
    for k in K_GRID:
        mean_estimation, mean_distance_estimation = neighbor_mean(
            embedding[estimation], distance_estimation, k
        )
        mean_tuning, mean_distance_tuning = neighbor_mean(
            embedding[estimation], distance_tuning, k
        )
        mean_test, mean_distance_test = neighbor_mean(
            embedding[estimation], distance_test, k
        )
        for mix in MIX_GRID:
            smooth_estimation = (1.0 - mix) * embedding[estimation] + mix * mean_estimation
            smooth_tuning = (1.0 - mix) * embedding[tuning] + mix * mean_tuning
            smooth_test = (1.0 - mix) * embedding[test] + mix * mean_test
            smooth_prediction, smooth_metric = fit_and_score(
                np.hstack([static[estimation], smooth_estimation]),
                np.hstack([static[tuning], smooth_tuning]),
                np.hstack([static[test], smooth_test]),
                y_estimation,
                y_tuning,
            )
            candidates["smoothed"].append(
                (
                    float(smooth_metric["top1_regret"]),
                    float(smooth_metric["rmse"]),
                    -float(smooth_metric["top1_hit_rate"]),
                    k,
                    mix,
                    smooth_prediction,
                )
            )

            augmented_estimation = np.hstack(
                [
                    static[estimation],
                    embedding[estimation],
                    mix * mean_estimation,
                    mix * (embedding[estimation] - mean_estimation),
                    mean_distance_estimation[:, None],
                ]
            )
            augmented_tuning = np.hstack(
                [
                    static[tuning],
                    embedding[tuning],
                    mix * mean_tuning,
                    mix * (embedding[tuning] - mean_tuning),
                    mean_distance_tuning[:, None],
                ]
            )
            augmented_test = np.hstack(
                [
                    static[test],
                    embedding[test],
                    mix * mean_test,
                    mix * (embedding[test] - mean_test),
                    mean_distance_test[:, None],
                ]
            )
            augmented_prediction, augmented_metric = fit_and_score(
                augmented_estimation,
                augmented_tuning,
                augmented_test,
                y_estimation,
                y_tuning,
            )
            candidates["augmented"].append(
                (
                    float(augmented_metric["top1_regret"]),
                    float(augmented_metric["rmse"]),
                    -float(augmented_metric["top1_hit_rate"]),
                    k,
                    mix,
                    augmented_prediction,
                )
            )
            diagnostics.extend(
                [
                    {
                        "method": "smoothed",
                        "k": k,
                        "mix": mix,
                        "top1_regret": float(smooth_metric["top1_regret"]),
                        "rmse": float(smooth_metric["rmse"]),
                    },
                    {
                        "method": "augmented",
                        "k": k,
                        "mix": mix,
                        "top1_regret": float(augmented_metric["top1_regret"]),
                        "rmse": float(augmented_metric["rmse"]),
                    },
                ]
            )

    selected_smoothed = min(candidates["smoothed"], key=lambda item: item[:5])
    selected_augmented = min(candidates["augmented"], key=lambda item: item[:5])

    # Prespecified low-strength variants protect against noisy split-specific
    # hyperparameter selection.
    fixed_k = 50
    fixed_mix = 0.25
    fixed_mean_estimation, _ = neighbor_mean(
        embedding[estimation], distance_estimation, fixed_k
    )
    fixed_mean_tuning, fixed_distance_tuning = neighbor_mean(
        embedding[estimation], distance_tuning, fixed_k
    )
    fixed_mean_test, fixed_distance_test = neighbor_mean(
        embedding[estimation], distance_test, fixed_k
    )
    fixed_smooth_prediction, _ = fit_and_score(
        np.hstack(
            [
                static[estimation],
                (1.0 - fixed_mix) * embedding[estimation]
                + fixed_mix * fixed_mean_estimation,
            ]
        ),
        np.hstack(
            [
                static[tuning],
                (1.0 - fixed_mix) * embedding[tuning] + fixed_mix * fixed_mean_tuning,
            ]
        ),
        np.hstack(
            [
                static[test],
                (1.0 - fixed_mix) * embedding[test] + fixed_mix * fixed_mean_test,
            ]
        ),
        y_estimation,
        y_tuning,
    )
    fixed_augmented_prediction, _ = fit_and_score(
        np.hstack(
            [
                static[estimation],
                embedding[estimation],
                fixed_mix * fixed_mean_estimation,
                fixed_mix * (embedding[estimation] - fixed_mean_estimation),
                np.take_along_axis(
                    distance_estimation,
                    nearest_indices(distance_estimation, fixed_k)[:, :fixed_k],
                    axis=1,
                ).mean(axis=1)[:, None],
            ]
        ),
        np.hstack(
            [
                static[tuning],
                embedding[tuning],
                fixed_mix * fixed_mean_tuning,
                fixed_mix * (embedding[tuning] - fixed_mean_tuning),
                fixed_distance_tuning[:, None],
            ]
        ),
        np.hstack(
            [
                static[test],
                embedding[test],
                fixed_mix * fixed_mean_test,
                fixed_mix * (embedding[test] - fixed_mean_test),
                fixed_distance_test[:, None],
            ]
        ),
        y_estimation,
        y_tuning,
    )

    metrics = [
        decision_metrics("base_embedding", base_test, y_test, base_test),
        decision_metrics(
            "tuned_smoothed_embedding",
            selected_smoothed[5],
            y_test,
            selected_smoothed[5],
        ),
        decision_metrics(
            "tuned_augmented_embedding",
            selected_augmented[5],
            y_test,
            selected_augmented[5],
        ),
        decision_metrics(
            "fixed_smoothed_embedding",
            fixed_smooth_prediction,
            y_test,
            fixed_smooth_prediction,
        ),
        decision_metrics(
            "fixed_augmented_embedding",
            fixed_augmented_prediction,
            y_test,
            fixed_augmented_prediction,
        ),
    ]
    return {
        "seed": seed,
        "runtime_seconds": time.perf_counter() - started,
        "selected_smoothed": {"k": selected_smoothed[3], "mix": selected_smoothed[4]},
        "selected_augmented": {"k": selected_augmented[3], "mix": selected_augmented[4]},
        "fixed": {"k": fixed_k, "mix": fixed_mix},
        "tuning_diagnostics": diagnostics,
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


def aggregate(rows: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, object]]:
    columns = ["rmse", "top1_regret", "top1_hit_rate", "ndcg", "pairwise_log_loss"]
    means = rows.groupby("method", sort=False)[columns].mean().reset_index()
    comparisons = {}
    for metric in ("rmse", "top1_regret", "top1_hit_rate"):
        pivot = rows.pivot(index="seed", columns="method", values=metric)
        for method in pivot.columns:
            if method == "base_embedding":
                continue
            delta = pivot[method] - pivot["base_embedding"]
            comparisons[f"{metric}:{method}"] = {
                "mean_change_vs_base": float(delta.mean()),
                "paired_95pct_interval": interval(delta.to_numpy()),
                "wins": int((delta < 0).sum()) if metric != "top1_hit_rate" else int((delta > 0).sum()),
            }
    return means, {"splits": int(rows.seed.nunique()), "comparisons": comparisons}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--data-root", type=Path, default=Path("data/kuairec/extracted")
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("propensity-neighborhoods-v2/embedding-neighborhood-results"),
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
    frames = []
    summaries = []
    for seed in args.seeds:
        print(f"Running embedding-neighborhood seed {seed}...", flush=True)
        summary = run_split(data, seed, args.embedding_dimensions, args.folds)
        directory = args.output / f"seed_{seed}"
        directory.mkdir(parents=True, exist_ok=True)
        (directory / "summary.json").write_text(
            json.dumps(summary, indent=2), encoding="utf-8"
        )
        frame = pd.DataFrame(summary["metrics"])
        frame.insert(0, "seed", seed)
        frame.to_csv(directory / "metrics.csv", index=False)
        frames.append(frame)
        summaries.append(summary)
        print(
            f"seed={seed} smooth={summary['selected_smoothed']} "
            f"augment={summary['selected_augmented']} "
            f"runtime={summary['runtime_seconds']:.1f}s",
            flush=True,
        )

    rows = pd.concat(frames, ignore_index=True)
    rows.to_csv(args.output / "split_metrics.csv", index=False)
    means, aggregate_summary = aggregate(rows)
    means.to_csv(args.output / "aggregate_metrics.csv", index=False)
    aggregate_summary.update(
        {
            "seeds": list(args.seeds),
            "runtime_seconds": time.perf_counter() - overall_started,
            "selected_smoothed": [summary["selected_smoothed"] for summary in summaries],
            "selected_augmented": [summary["selected_augmented"] for summary in summaries],
        }
    )
    (args.output / "aggregate_summary.json").write_text(
        json.dumps(aggregate_summary, indent=2), encoding="utf-8"
    )
    print(means.to_string(index=False), flush=True)
    print(json.dumps(aggregate_summary, indent=2), flush=True)


if __name__ == "__main__":
    main()

