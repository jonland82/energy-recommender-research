"""Leakage-safe KuaiRec test of program-specific propensity neighborhoods.

The base propensity model is cross-fitted on an estimation pool. Leave-one-out
residual similarity inside that pool learns one nonnegative set of feature-block
weights per program. A separate tuning split selects neighbor count and residual
shrinkage, and an untouched test split supplies the reported comparison.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.metrics import pairwise_distances
from sklearn.model_selection import KFold, cross_val_predict

from kuairec_propensity_neighborhood_pilot import (
    find_file,
    make_transformer,
    metric_row,
    parse_first_tag,
)
from sklearn.linear_model import Ridge
from sklearn.pipeline import make_pipeline


def nearest_residual_prediction(
    base_prediction: np.ndarray,
    reference_residual: np.ndarray,
    distance: np.ndarray,
    k: int,
    shrinkage: float,
) -> np.ndarray:
    """Apply a shared k-neighbor residual correction to all programs."""
    k_eff = min(k, distance.shape[1])
    neighbors = np.argpartition(distance, k_eff - 1, axis=1)[:, :k_eff]
    correction = reference_residual[neighbors].mean(axis=1)
    correction *= k_eff / (k_eff + shrinkage)
    return np.clip(base_prediction + correction, 0.0, 1.0)


def action_prediction(
    base_prediction: np.ndarray,
    reference_residual: np.ndarray,
    distance: np.ndarray,
    action: int,
    k: int,
    shrinkage: float,
) -> np.ndarray:
    """Apply a k-neighbor residual correction for one program."""
    k_eff = min(k, distance.shape[1])
    neighbors = np.argpartition(distance, k_eff - 1, axis=1)[:, :k_eff]
    correction = reference_residual[neighbors, action].mean(axis=1)
    correction *= k_eff / (k_eff + shrinkage)
    return np.clip(base_prediction[:, action] + correction, 0.0, 1.0)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", type=Path, default=Path("data/kuairec/extracted"))
    parser.add_argument(
        "--output", type=Path, default=Path("results/kuairec_program_specific_knn")
    )
    parser.add_argument("--seed", type=int, default=20260902)
    parser.add_argument("--categories", type=int, default=12)
    parser.add_argument("--completion-threshold", type=float, default=1.0)
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
    selected = (
        categories.groupby("category")["video_id"]
        .nunique()
        .nlargest(args.categories)
        .index.to_list()
    )
    video_category = categories.loc[
        categories["category"].isin(selected), ["video_id", "category"]
    ].drop_duplicates("video_id")

    interactions = pd.read_csv(
        small_path,
        usecols=["user_id", "video_id", "watch_ratio"],
        dtype={"user_id": "int32", "video_id": "int32", "watch_ratio": "float32"},
    ).merge(video_category, on="video_id", how="inner")
    interactions["complete"] = (
        interactions["watch_ratio"] >= args.completion_threshold
    ).astype("float32")
    rates = interactions.pivot_table(
        index="user_id", columns="category", values="complete", aggfunc="mean"
    ).dropna()

    users = pd.read_csv(user_path)
    users = users.loc[users["user_id"].isin(rates.index)].drop_duplicates("user_id")
    users = users.set_index("user_id").loc[rates.index]
    rates = rates.loc[users.index]

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
    blocks = [[c for c in block if c in users.columns] for block in preferred_blocks]
    blocks = [block for block in blocks if block]
    model_columns = list(dict.fromkeys(c for block in blocks for c in block))

    rng = np.random.default_rng(args.seed)
    ids = rates.index.to_numpy().copy()
    rng.shuffle(ids)
    n_users = len(ids)
    n_estimation = int(0.60 * n_users)
    n_tuning = int(0.25 * n_users)
    estimation_ids = ids[:n_estimation]
    tuning_ids = ids[n_estimation : n_estimation + n_tuning]
    test_ids = ids[n_estimation + n_tuning :]

    y_estimation = rates.loc[estimation_ids].to_numpy()
    y_tuning = rates.loc[tuning_ids].to_numpy()
    y_test = rates.loc[test_ids].to_numpy()

    transformer = make_transformer(users.loc[estimation_ids], model_columns)
    base_model = make_pipeline(transformer, Ridge(alpha=10.0))
    folds = KFold(n_splits=args.folds, shuffle=True, random_state=args.seed + 101)
    pred_estimation_oof = np.clip(
        cross_val_predict(
            clone(base_model),
            users.loc[estimation_ids, model_columns],
            y_estimation,
            cv=folds,
            method="predict",
        ),
        0.0,
        1.0,
    )
    reference_residual = y_estimation - pred_estimation_oof

    base_model.fit(users.loc[estimation_ids, model_columns], y_estimation)
    pred_tuning = np.clip(
        base_model.predict(users.loc[tuning_ids, model_columns]), 0.0, 1.0
    )
    pred_test = np.clip(
        base_model.predict(users.loc[test_ids, model_columns]), 0.0, 1.0
    )

    group_ids = {
        "estimation": estimation_ids,
        "tuning": tuning_ids,
        "test": test_ids,
    }
    block_distances: dict[str, list[np.ndarray]] = {name: [] for name in group_ids}
    for block in blocks:
        block_transformer = make_transformer(users.loc[estimation_ids], block)
        block_transformer.fit(users.loc[estimation_ids, block])
        reference_matrix = block_transformer.transform(users.loc[estimation_ids, block])
        raw_distances = {}
        for name, group in group_ids.items():
            matrix = block_transformer.transform(users.loc[group, block])
            raw_distances[name] = pairwise_distances(
                matrix, reference_matrix, metric="euclidean"
            )
        positive = raw_distances["estimation"][raw_distances["estimation"] > 0]
        scale = float(np.median(positive)) if len(positive) else 1.0
        for name in group_ids:
            block_distances[name].append(raw_distances[name] / max(scale, 1e-12))

    # Learn one interpretable, nonnegative block-weight vector per program.
    k_probe = min(100, len(estimation_ids))
    action_count = y_estimation.shape[1]
    block_count = len(blocks)
    block_mse = np.zeros((action_count, block_count))
    weights = np.zeros_like(block_mse)
    for action in range(action_count):
        zero_mse = float(np.mean(reference_residual[:, action] ** 2))
        for block_index, raw_distance in enumerate(block_distances["estimation"]):
            distance = raw_distance.copy()
            np.fill_diagonal(distance, np.inf)
            estimate = action_prediction(
                pred_estimation_oof,
                reference_residual,
                distance,
                action,
                k_probe,
                0.0,
            )
            block_mse[action, block_index] = np.mean(
                (estimate - y_estimation[:, action]) ** 2
            )
        gain = np.maximum(0.0, 1.0 - block_mse[action] / max(zero_mse, 1e-12))
        learned = (
            gain / gain.sum() if gain.sum() else np.full(block_count, 1 / block_count)
        )
        # A small uniform component stabilizes weights learned from 10% of users.
        weights[action] = 0.90 * learned + 0.10 / block_count

    k_grid = [25, 50, 100, 200]
    # The final value is effectively "no neighborhood correction" and lets
    # validation decline calibration for programs where peers add no signal.
    shrinkage_grid = [0.0, 20.0, 50.0, 100.0, 200.0, 1.0e9]

    # Strong shared baseline: validation chooses one geometry, k, and shrinkage
    # for all programs by the actual top-1 recommendation regret.
    shared_geometries: dict[str, dict[str, np.ndarray]] = {}
    for block_index in range(block_count):
        shared_geometries[f"block_{block_index + 1}"] = {
            name: block_distances[name][block_index] for name in group_ids
        }
    for depth in range(2, block_count + 1):
        shared_geometries[f"prefix_{depth}"] = {
            name: np.mean(block_distances[name][:depth], axis=0) for name in group_ids
        }

    shared_candidates = []
    for geometry_name, distances in shared_geometries.items():
        for k in k_grid:
            for shrinkage in shrinkage_grid:
                prediction = nearest_residual_prediction(
                    pred_tuning,
                    reference_residual,
                    distances["tuning"],
                    k,
                    shrinkage,
                )
                score = metric_row("candidate", prediction, y_tuning)
                shared_candidates.append(
                    (score["top1_regret"], score["rmse"], geometry_name, k, shrinkage)
                )
    _, _, shared_geometry, shared_k, shared_shrinkage = min(shared_candidates)
    shared_test_prediction = nearest_residual_prediction(
        pred_test,
        reference_residual,
        shared_geometries[shared_geometry]["test"],
        shared_k,
        shared_shrinkage,
    )
    prespecified_test_prediction = nearest_residual_prediction(
        pred_test,
        reference_residual,
        shared_geometries[f"prefix_{block_count}"]["test"],
        200,
        50.0,
    )

    # Program-specific weighted geometry and independently validated k/shrinkage.
    program_test_prediction = np.empty_like(pred_test)
    selected_hyperparameters = []
    for action in range(action_count):
        tuning_distance = sum(
            weights[action, block_index] * block_distances["tuning"][block_index]
            for block_index in range(block_count)
        )
        test_distance = sum(
            weights[action, block_index] * block_distances["test"][block_index]
            for block_index in range(block_count)
        )
        candidates = []
        for k in k_grid:
            for shrinkage in shrinkage_grid:
                prediction = action_prediction(
                    pred_tuning,
                    reference_residual,
                    tuning_distance,
                    action,
                    k,
                    shrinkage,
                )
                mse = float(np.mean((prediction - y_tuning[:, action]) ** 2))
                candidates.append((mse, k, shrinkage))
        validation_mse, selected_k, selected_shrinkage = min(candidates)
        program_test_prediction[:, action] = action_prediction(
            pred_test,
            reference_residual,
            test_distance,
            action,
            selected_k,
            selected_shrinkage,
        )
        selected_hyperparameters.append(
            {
                "category": int(rates.columns[action]),
                "k": selected_k,
                "shrinkage": selected_shrinkage,
                "tuning_mse": validation_mse,
                "block_weights": [float(x) for x in weights[action]],
                "block_metric_mse": [float(x) for x in block_mse[action]],
            }
        )

    metrics = [
        metric_row("base_ridge", pred_test, y_test),
        metric_row("prespecified_broad_knn", prespecified_test_prediction, y_test),
        metric_row("globally_tuned_shared_knn", shared_test_prediction, y_test),
        metric_row("program_specific_knn", program_test_prediction, y_test),
    ]
    summary = {
        "seed": args.seed,
        "runtime_seconds": time.perf_counter() - started,
        "users": n_users,
        "estimation_users": len(estimation_ids),
        "tuning_users": len(tuning_ids),
        "test_users": len(test_ids),
        "completion_threshold": args.completion_threshold,
        "completion_rate": float(interactions["complete"].mean()),
        "categories": [int(x) for x in rates.columns],
        "feature_blocks": blocks,
        "k_grid": k_grid,
        "shrinkage_grid": shrinkage_grid,
        "shared_selection": {
            "geometry": shared_geometry,
            "k": shared_k,
            "shrinkage": shared_shrinkage,
        },
        "program_selections": selected_hyperparameters,
        "metrics": metrics,
    }
    pd.DataFrame(metrics).to_csv(args.output / "metrics.csv", index=False)
    (args.output / "summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
