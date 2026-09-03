"""Leakage-safe test of behavioral embeddings for KuaiRec propensity ranking.

Each user's early interactions form the representation. Later interactions form
the category-level propensity targets. Test-user future outcomes are never used
to learn embeddings, fit models, select neighbors, or tune hyperparameters.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import sparse
from sklearn.linear_model import Ridge
from sklearn.metrics import pairwise_distances
from sklearn.model_selection import KFold, cross_val_predict
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import TruncatedSVD

from kuairec_propensity_neighborhood_pilot import (
    find_file,
    make_transformer,
    metric_row,
    parse_first_tag,
)


def neighbor_prediction(
    base: np.ndarray,
    reference_residual: np.ndarray,
    distance: np.ndarray,
    k: int,
    shrinkage: float,
) -> np.ndarray:
    k_eff = min(k, distance.shape[1])
    neighbors = np.argpartition(distance, k_eff - 1, axis=1)[:, :k_eff]
    correction = reference_residual[neighbors].mean(axis=1)
    correction *= k_eff / (k_eff + shrinkage)
    return np.clip(base + correction, 0.0, 1.0)


def action_neighbor_prediction(
    base: np.ndarray,
    reference_residual: np.ndarray,
    distance: np.ndarray,
    action: int,
    k: int,
    shrinkage: float,
) -> np.ndarray:
    k_eff = min(k, distance.shape[1])
    neighbors = np.argpartition(distance, k_eff - 1, axis=1)[:, :k_eff]
    correction = reference_residual[neighbors, action].mean(axis=1)
    correction *= k_eff / (k_eff + shrinkage)
    return np.clip(base[:, action] + correction, 0.0, 1.0)


def fit_cross_fitted_ridge(
    x_estimation: np.ndarray,
    y_estimation: np.ndarray,
    x_tuning: np.ndarray,
    x_test: np.ndarray,
    seed: int,
    folds: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    split = KFold(n_splits=folds, shuffle=True, random_state=seed + 101)
    oof = np.clip(
        cross_val_predict(
            Ridge(alpha=10.0), x_estimation, y_estimation, cv=split, method="predict"
        ),
        0.0,
        1.0,
    )
    model = Ridge(alpha=10.0).fit(x_estimation, y_estimation)
    tuning = np.clip(model.predict(x_tuning), 0.0, 1.0)
    test = np.clip(model.predict(x_test), 0.0, 1.0)
    return oof, tuning, test


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", type=Path, default=Path("data/kuairec/extracted"))
    parser.add_argument(
        "--output", type=Path, default=Path("results/kuairec_behavior_embeddings")
    )
    parser.add_argument("--seed", type=int, default=20260902)
    parser.add_argument("--categories", type=int, default=12)
    parser.add_argument("--history-fraction", type=float, default=0.60)
    parser.add_argument("--embedding-dimensions", type=int, default=32)
    parser.add_argument("--folds", type=int, default=5)
    args = parser.parse_args()

    started = time.perf_counter()
    args.output.mkdir(parents=True, exist_ok=True)

    category_path = find_file(args.data_root, "item_categories.csv")
    small_path = find_file(args.data_root, "small_matrix.csv")
    user_path = find_file(args.data_root, "user_features.csv")

    categories = pd.read_csv(category_path, usecols=["video_id", "feat"])
    categories["category"] = categories["feat"].map(parse_first_tag)
    categories = categories.dropna(subset=["category"])
    categories["category"] = categories["category"].astype(int)
    selected_categories = (
        categories.groupby("category")["video_id"]
        .nunique()
        .nlargest(args.categories)
        .index.to_list()
    )
    video_category = categories.loc[
        categories["category"].isin(selected_categories), ["video_id", "category"]
    ].drop_duplicates("video_id")

    interactions = pd.read_csv(
        small_path,
        usecols=["user_id", "video_id", "timestamp", "watch_ratio"],
        dtype={
            "user_id": "int32",
            "video_id": "int32",
            "timestamp": "float64",
            "watch_ratio": "float32",
        },
    ).merge(video_category, on="video_id", how="inner")
    interactions = interactions.sort_values(["user_id", "timestamp"])
    position = interactions.groupby("user_id").cumcount()
    group_size = interactions.groupby("user_id")["user_id"].transform("size")
    history_mask = position < np.floor(args.history_fraction * group_size)
    history = interactions.loc[history_mask].copy()
    future = interactions.loc[~history_mask].copy()
    history["complete"] = (history["watch_ratio"] >= 1.0).astype("float32")
    future["complete"] = (future["watch_ratio"] >= 1.0).astype("float32")

    future_rates = future.pivot_table(
        index="user_id", columns="category", values="complete", aggfunc="mean"
    ).dropna()
    users = pd.read_csv(user_path)
    users = users.loc[users["user_id"].isin(future_rates.index)].drop_duplicates("user_id")
    users = users.set_index("user_id").loc[future_rates.index]
    future_rates = future_rates.loc[users.index]
    user_ids = future_rates.index.to_numpy()
    category_order = list(future_rates.columns)

    history = history.loc[history["user_id"].isin(user_ids)]
    user_lookup = pd.Series(np.arange(len(user_ids)), index=user_ids)
    video_ids = np.sort(history["video_id"].unique())
    video_lookup = pd.Series(np.arange(len(video_ids)), index=video_ids)
    rows = history["user_id"].map(user_lookup).to_numpy()
    columns = history["video_id"].map(video_lookup).to_numpy()
    values = history["complete"].to_numpy(dtype=np.float64)
    history_matrix = sparse.csr_matrix(
        (values, (rows, columns)), shape=(len(user_ids), len(video_ids))
    )

    history_rates = history.pivot_table(
        index="user_id", columns="category", values="complete", aggfunc="mean"
    ).reindex(index=user_ids, columns=category_order)

    rng = np.random.default_rng(args.seed)
    shuffled = np.arange(len(user_ids))
    rng.shuffle(shuffled)
    n_estimation = int(0.60 * len(shuffled))
    n_tuning = int(0.25 * len(shuffled))
    estimation = shuffled[:n_estimation]
    tuning = shuffled[n_estimation : n_estimation + n_tuning]
    test = shuffled[n_estimation + n_tuning :]

    y = future_rates.to_numpy()
    y_estimation, y_tuning, y_test = y[estimation], y[tuning], y[test]

    svd = TruncatedSVD(n_components=args.embedding_dimensions, random_state=args.seed)
    svd.fit(history_matrix[estimation])
    embedding = svd.transform(history_matrix)
    embedding_scaler = StandardScaler().fit(embedding[estimation])
    embedding = embedding_scaler.transform(embedding)

    preferred_blocks = [
        ["user_active_degree", "is_lowactive_period", "register_days", "register_days_range"],
        ["is_live_streamer", "is_video_author"],
        [
            "follow_user_num",
            "follow_user_num_range",
            "fans_user_num",
            "fans_user_num_range",
            "friend_user_num",
            "friend_user_num_range",
        ],
        [c for c in users.columns if c.startswith("onehot_feat")],
    ]
    model_columns = list(
        dict.fromkeys(
            c for block in preferred_blocks for c in block if c in users.columns
        )
    )
    transformer = make_transformer(users.iloc[estimation], model_columns)
    transformer.fit(users.iloc[estimation][model_columns])
    static_features = transformer.transform(users[model_columns])

    history_rate_array = history_rates.to_numpy(dtype=np.float64)
    history_means = np.nanmean(history_rate_array[estimation], axis=0)
    missing_rows, missing_columns = np.where(np.isnan(history_rate_array))
    history_rate_array[missing_rows, missing_columns] = history_means[missing_columns]
    history_scaler = StandardScaler().fit(history_rate_array[estimation])
    history_rate_array = history_scaler.transform(history_rate_array)

    static_x = np.asarray(static_features)
    embedding_x = np.hstack([static_x, embedding])
    behavior_x = np.hstack([static_x, embedding, history_rate_array])

    static_oof, static_tuning, static_test = fit_cross_fitted_ridge(
        static_x[estimation], y_estimation, static_x[tuning], static_x[test], args.seed, args.folds
    )
    embedding_oof, embedding_tuning, embedding_test = fit_cross_fitted_ridge(
        embedding_x[estimation],
        y_estimation,
        embedding_x[tuning],
        embedding_x[test],
        args.seed,
        args.folds,
    )
    behavior_oof, behavior_tuning, behavior_test = fit_cross_fitted_ridge(
        behavior_x[estimation],
        y_estimation,
        behavior_x[tuning],
        behavior_x[test],
        args.seed,
        args.folds,
    )

    reference_residual = embedding_oof - y_estimation
    reference_residual *= -1.0
    distance_sets = {}
    for distance_name in ["euclidean", "cosine"]:
        distance_sets[distance_name] = {
            "estimation": pairwise_distances(
                embedding[estimation], embedding[estimation], metric=distance_name
            ),
            "tuning": pairwise_distances(
                embedding[tuning], embedding[estimation], metric=distance_name
            ),
            "test": pairwise_distances(
                embedding[test], embedding[estimation], metric=distance_name
            ),
        }

    k_grid = [25, 50, 100, 200]
    shrinkage_grid = [0.0, 20.0, 50.0, 100.0, 200.0, 1.0e9]
    shared_candidates = []
    for distance_name, distances in distance_sets.items():
        for k in k_grid:
            for shrinkage in shrinkage_grid:
                prediction = neighbor_prediction(
                    embedding_tuning,
                    reference_residual,
                    distances["tuning"],
                    k,
                    shrinkage,
                )
                score = metric_row("candidate", prediction, y_tuning)
                shared_candidates.append(
                    (score["top1_regret"], score["rmse"], distance_name, k, shrinkage)
                )
    _, _, shared_distance, shared_k, shared_shrinkage = min(shared_candidates)
    shared_test = neighbor_prediction(
        embedding_test,
        reference_residual,
        distance_sets[shared_distance]["test"],
        shared_k,
        shared_shrinkage,
    )

    program_test = np.empty_like(embedding_test)
    program_selections = []
    for action, category in enumerate(category_order):
        candidates = []
        for distance_name, distances in distance_sets.items():
            for k in k_grid:
                for shrinkage in shrinkage_grid:
                    prediction = action_neighbor_prediction(
                        embedding_tuning,
                        reference_residual,
                        distances["tuning"],
                        action,
                        k,
                        shrinkage,
                    )
                    mse = float(np.mean((prediction - y_tuning[:, action]) ** 2))
                    candidates.append((mse, distance_name, k, shrinkage))
        tuning_mse, distance_name, k, shrinkage = min(candidates)
        program_test[:, action] = action_neighbor_prediction(
            embedding_test,
            reference_residual,
            distance_sets[distance_name]["test"],
            action,
            k,
            shrinkage,
        )
        program_selections.append(
            {
                "category": int(category),
                "distance": distance_name,
                "k": k,
                "shrinkage": shrinkage,
                "tuning_mse": tuning_mse,
            }
        )

    # Propensity-space neighborhoods use fitted score rows rather than the
    # original customer representation. Reference rows are strictly out of fold.
    propensity_distance_sets = {}
    for distance_name in ["euclidean", "chebyshev"]:
        propensity_distance_sets[distance_name] = {
            "tuning": pairwise_distances(
                embedding_tuning, embedding_oof, metric=distance_name
            ),
            "test": pairwise_distances(
                embedding_test, embedding_oof, metric=distance_name
            ),
        }

    propensity_shared_candidates = []
    for distance_name, distances in propensity_distance_sets.items():
        for k in k_grid:
            for shrinkage in shrinkage_grid:
                prediction = neighbor_prediction(
                    embedding_tuning,
                    reference_residual,
                    distances["tuning"],
                    k,
                    shrinkage,
                )
                score = metric_row("candidate", prediction, y_tuning)
                propensity_shared_candidates.append(
                    (score["top1_regret"], score["rmse"], distance_name, k, shrinkage)
                )
    (
        _,
        _,
        propensity_shared_distance,
        propensity_shared_k,
        propensity_shared_shrinkage,
    ) = min(propensity_shared_candidates)
    propensity_shared_test = neighbor_prediction(
        embedding_test,
        reference_residual,
        propensity_distance_sets[propensity_shared_distance]["test"],
        propensity_shared_k,
        propensity_shared_shrinkage,
    )
    propensity_fixed_test = neighbor_prediction(
        embedding_test,
        reference_residual,
        propensity_distance_sets["chebyshev"]["test"],
        50,
        20.0,
    )

    # When calibrating program a, exclude its own fitted propensity from the
    # distance. This tests whether the other program scores predict its residual.
    propensity_loo_test = np.empty_like(embedding_test)
    propensity_loo_selections = []
    action_count = len(category_order)
    for action, category in enumerate(category_order):
        other_actions = [index for index in range(action_count) if index != action]
        candidates = []
        action_distances = {}
        for distance_name in ["euclidean", "chebyshev"]:
            action_distances[distance_name] = {
                "tuning": pairwise_distances(
                    embedding_tuning[:, other_actions],
                    embedding_oof[:, other_actions],
                    metric=distance_name,
                ),
                "test": pairwise_distances(
                    embedding_test[:, other_actions],
                    embedding_oof[:, other_actions],
                    metric=distance_name,
                ),
            }
            for k in k_grid:
                for shrinkage in shrinkage_grid:
                    prediction = action_neighbor_prediction(
                        embedding_tuning,
                        reference_residual,
                        action_distances[distance_name]["tuning"],
                        action,
                        k,
                        shrinkage,
                    )
                    mse = float(np.mean((prediction - y_tuning[:, action]) ** 2))
                    candidates.append((mse, distance_name, k, shrinkage))
        tuning_mse, distance_name, k, shrinkage = min(candidates)
        propensity_loo_test[:, action] = action_neighbor_prediction(
            embedding_test,
            reference_residual,
            action_distances[distance_name]["test"],
            action,
            k,
            shrinkage,
        )
        propensity_loo_selections.append(
            {
                "category": int(category),
                "distance": distance_name,
                "k": k,
                "shrinkage": shrinkage,
                "tuning_mse": tuning_mse,
            }
        )

    residual_diagnostic = []
    for distance_name, distances in distance_sets.items():
        leave_one_out = distances["estimation"].copy()
        np.fill_diagonal(leave_one_out, np.inf)
        corrected = neighbor_prediction(
            embedding_oof, reference_residual, leave_one_out, 100, 50.0
        )
        residual_diagnostic.append(
            {
                "distance": distance_name,
                "base_rmse": metric_row("base", embedding_oof, y_estimation)["rmse"],
                "corrected_rmse": metric_row("corrected", corrected, y_estimation)["rmse"],
            }
        )

    metrics = [
        metric_row("static_base", static_test, y_test),
        metric_row("static_plus_svd", embedding_test, y_test),
        metric_row("static_svd_plus_history_rates", behavior_test, y_test),
        metric_row("embedding_shared_knn", shared_test, y_test),
        metric_row("embedding_program_specific_knn", program_test, y_test),
        metric_row("propensity_full_shared_knn", propensity_shared_test, y_test),
        metric_row("propensity_full_fixed_chebyshev_k50", propensity_fixed_test, y_test),
        metric_row("propensity_leave_one_program_out_knn", propensity_loo_test, y_test),
    ]
    summary = {
        "seed": args.seed,
        "runtime_seconds": time.perf_counter() - started,
        "users": len(user_ids),
        "estimation_users": len(estimation),
        "tuning_users": len(tuning),
        "test_users": len(test),
        "history_fraction": args.history_fraction,
        "history_interactions": len(history),
        "future_interactions": len(future),
        "embedding_dimensions": args.embedding_dimensions,
        "embedding_explained_variance": float(svd.explained_variance_ratio_.sum()),
        "shared_selection": {
            "distance": shared_distance,
            "k": shared_k,
            "shrinkage": shared_shrinkage,
        },
        "program_selections": program_selections,
        "propensity_shared_selection": {
            "distance": propensity_shared_distance,
            "k": propensity_shared_k,
            "shrinkage": propensity_shared_shrinkage,
        },
        "propensity_leave_one_out_selections": propensity_loo_selections,
        "residual_diagnostic": residual_diagnostic,
        "metrics": metrics,
    }
    pd.DataFrame(metrics).to_csv(args.output / "metrics.csv", index=False)
    (args.output / "summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
