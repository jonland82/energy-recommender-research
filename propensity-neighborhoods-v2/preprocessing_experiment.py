"""Neighborhood preprocessing before propensity estimation and ranking.

A preliminary cross-fitted propensity model defines propensity-space
neighborhoods. Historical peer outcomes are summarized without using the focal
user's outcome, then supplied as features to a second propensity model. Ranking
remains ordinary descending sorting of the final fitted propensities.
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
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from experiment import (
    DEFAULT_SEEDS,
    decision_metrics,
    fit_base_model,
    load_chronological_data,
    make_transformer,
    neighborhood_feature_tensor,
    parse_seeds,
)


ALPHA_GRID = (0.1, 1.0, 10.0, 100.0, 1000.0, 10000.0, 1.0e6)


def fit_second_stage(
    common_estimation: np.ndarray,
    common_tuning: np.ndarray,
    common_test: np.ndarray,
    preliminary_estimation: np.ndarray,
    preliminary_tuning: np.ndarray,
    preliminary_test: np.ndarray,
    y_estimation: np.ndarray,
    y_tuning: np.ndarray,
    estimation_neighborhood: np.ndarray | None,
    tuning_neighborhood: np.ndarray | None,
    test_neighborhood: np.ndarray | None,
) -> tuple[np.ndarray, float, list[dict[str, float]]]:
    action_count = y_estimation.shape[1]
    candidates = []
    diagnostics = []
    for alpha in ALPHA_GRID:
        tuning_prediction = np.empty_like(y_tuning)
        test_prediction = np.empty((len(common_test), action_count), dtype=np.float64)
        for action in range(action_count):
            estimation_parts = [common_estimation, preliminary_estimation]
            tuning_parts = [common_tuning, preliminary_tuning]
            test_parts = [common_test, preliminary_test]
            if estimation_neighborhood is not None:
                estimation_parts.append(estimation_neighborhood[:, action, :])
                tuning_parts.append(tuning_neighborhood[:, action, :])
                test_parts.append(test_neighborhood[:, action, :])
            x_estimation = np.hstack(estimation_parts)
            x_tuning = np.hstack(tuning_parts)
            x_test = np.hstack(test_parts)
            # The second stage is an additive preprocessing correction. Strong
            # regularization converges to the preliminary model instead of
            # forcing an unnecessary complete refit.
            model = make_pipeline(
                StandardScaler(), Ridge(alpha=alpha, fit_intercept=False)
            )
            model.fit(
                x_estimation,
                y_estimation[:, action] - preliminary_estimation[:, action],
            )
            tuning_prediction[:, action] = (
                preliminary_tuning[:, action] + model.predict(x_tuning)
            )
            test_prediction[:, action] = (
                preliminary_test[:, action] + model.predict(x_test)
            )
        tuning_prediction = np.clip(tuning_prediction, 0.0, 1.0)
        test_prediction = np.clip(test_prediction, 0.0, 1.0)
        metric = decision_metrics(
            "candidate", tuning_prediction, y_tuning, tuning_prediction
        )
        diagnostics.append(
            {
                "alpha": alpha,
                "top1_regret": float(metric["top1_regret"]),
                "top1_hit_rate": float(metric["top1_hit_rate"]),
                "rmse": float(metric["rmse"]),
            }
        )
        candidates.append(
            (
                float(metric["top1_regret"]),
                float(metric["rmse"]),
                -float(metric["top1_hit_rate"]),
                alpha,
                test_prediction,
            )
        )
    selected = min(candidates, key=lambda item: item[:4])
    return selected[4], float(selected[3]), diagnostics


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
    outcomes = future_rates.to_numpy()
    y_estimation = outcomes[estimation]
    y_tuning = outcomes[tuning]
    y_test = outcomes[test]

    svd = TruncatedSVD(n_components=embedding_dimensions, random_state=seed)
    svd.fit(history_matrix[estimation])
    embedding = svd.transform(history_matrix)
    embedding = StandardScaler().fit(embedding[estimation]).transform(embedding)
    transformer = make_transformer(users.iloc[estimation], model_columns)
    transformer.fit(users.iloc[estimation][model_columns])
    static = np.asarray(transformer.transform(users[model_columns]))
    common = np.hstack([static, embedding])

    preliminary_oof, preliminary_tuning, preliminary_test = fit_base_model(
        common, outcomes, estimation, tuning, test, seed, folds
    )
    propensity_distance = {
        "estimation": pairwise_distances(
            preliminary_oof, preliminary_oof, metric="chebyshev"
        ),
        "tuning": pairwise_distances(
            preliminary_tuning, preliminary_oof, metric="chebyshev"
        ),
        "test": pairwise_distances(
            preliminary_test, preliminary_oof, metric="chebyshev"
        ),
    }
    embedding_distance = {
        "estimation": pairwise_distances(
            embedding[estimation], embedding[estimation], metric="cosine"
        ),
        "tuning": pairwise_distances(
            embedding[tuning], embedding[estimation], metric="cosine"
        ),
        "test": pairwise_distances(
            embedding[test], embedding[estimation], metric="cosine"
        ),
    }
    np.fill_diagonal(propensity_distance["estimation"], np.inf)
    np.fill_diagonal(embedding_distance["estimation"], np.inf)
    distance_sets = {
        split: {
            "propensity": propensity_distance[split],
            "embedding": embedding_distance[split],
        }
        for split in ("estimation", "tuning", "test")
    }
    neighborhood_estimation, feature_names = neighborhood_feature_tensor(
        preliminary_oof,
        preliminary_oof,
        y_estimation,
        distance_sets["estimation"],
    )
    neighborhood_tuning, _ = neighborhood_feature_tensor(
        preliminary_tuning,
        preliminary_oof,
        y_estimation,
        distance_sets["tuning"],
    )
    neighborhood_test, _ = neighborhood_feature_tensor(
        preliminary_test,
        preliminary_oof,
        y_estimation,
        distance_sets["test"],
    )
    propensity_columns = [
        index for index, name in enumerate(feature_names) if name.startswith("propensity_")
    ]
    embedding_columns = [
        index for index, name in enumerate(feature_names) if name.startswith("embedding_")
    ]
    combined_columns = propensity_columns + embedding_columns

    split_common = (common[estimation], common[tuning], common[test])
    split_preliminary = (preliminary_oof, preliminary_tuning, preliminary_test)
    configurations = {
        "stacked_base_preprocessor": None,
        "propensity_neighborhood_preprocessor": propensity_columns,
        "embedding_neighborhood_preprocessor": embedding_columns,
        "combined_neighborhood_preprocessor": combined_columns,
    }
    predictions = {}
    selected_alphas = {}
    tuning_diagnostics = {}
    for name, columns in configurations.items():
        if columns is None:
            neighborhood_parts = (None, None, None)
        else:
            neighborhood_parts = (
                neighborhood_estimation[:, :, columns],
                neighborhood_tuning[:, :, columns],
                neighborhood_test[:, :, columns],
            )
        prediction, alpha, diagnostics = fit_second_stage(
            *split_common,
            *split_preliminary,
            y_estimation,
            y_tuning,
            *neighborhood_parts,
        )
        predictions[name] = prediction
        selected_alphas[name] = alpha
        tuning_diagnostics[name] = diagnostics

    metrics = [
        decision_metrics(
            "preliminary_base", preliminary_test, y_test, preliminary_test
        )
    ]
    metrics.extend(
        decision_metrics(name, prediction, y_test, prediction)
        for name, prediction in predictions.items()
    )
    return {
        "seed": seed,
        "runtime_seconds": time.perf_counter() - started,
        "users": len(users),
        "estimation_users": len(estimation),
        "tuning_users": len(tuning),
        "test_users": len(test),
        "embedding_dimensions": embedding_dimensions,
        "alpha_grid": list(ALPHA_GRID),
        "propensity_neighborhood_feature_count_per_action": len(propensity_columns),
        "embedding_neighborhood_feature_count_per_action": len(embedding_columns),
        "selected_alphas": selected_alphas,
        "tuning_diagnostics": tuning_diagnostics,
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
    metrics = ["rmse", "top1_regret", "top1_hit_rate", "ndcg", "pairwise_log_loss"]
    means = rows.groupby("method", sort=False)[metrics].mean().reset_index()
    comparisons = {}
    for metric in ("rmse", "top1_regret", "top1_hit_rate"):
        pivot = rows.pivot(index="seed", columns="method", values=metric)
        for method in pivot.columns:
            if method in ("preliminary_base", "stacked_base_preprocessor"):
                continue
            for reference in ("preliminary_base", "stacked_base_preprocessor"):
                delta = pivot[method] - pivot[reference]
                key = f"{metric}:{method}_vs_{reference}"
                comparisons[key] = {
                    "mean_change": float(delta.mean()),
                    "paired_95pct_interval": interval(delta.to_numpy()),
                    "wins": int((delta < 0).sum()) if metric != "top1_hit_rate" else int((delta > 0).sum()),
                    "splits": int(len(delta)),
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
        default=Path("propensity-neighborhoods-v2/preprocessing-results"),
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
        print(f"Running preprocessing seed {seed}...", flush=True)
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
            f"seed={seed} runtime={summary['runtime_seconds']:.1f}s "
            f"alphas={summary['selected_alphas']}",
            flush=True,
        )

    rows = pd.concat(all_rows, ignore_index=True)
    rows.to_csv(args.output / "split_metrics.csv", index=False)
    means, aggregate_summary = aggregate(rows)
    means.to_csv(args.output / "aggregate_metrics.csv", index=False)
    aggregate_summary.update(
        {
            "seeds": list(args.seeds),
            "runtime_seconds": time.perf_counter() - overall_started,
            "selected_alphas": [summary["selected_alphas"] for summary in summaries],
        }
    )
    (args.output / "aggregate_summary.json").write_text(
        json.dumps(aggregate_summary, indent=2), encoding="utf-8"
    )
    print(means.to_string(index=False), flush=True)
    print(json.dumps(aggregate_summary, indent=2), flush=True)


if __name__ == "__main__":
    main()
