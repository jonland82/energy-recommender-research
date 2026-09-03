"""Fast KuaiRec pilot for adaptive propensity neighborhoods.

Uses disjoint train, reference-peer, and target-test users. Videos are grouped by
their first category tag, and a user's category propensity is the fraction of
videos watched to completion in that category.
"""

from __future__ import annotations

import argparse
import ast
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge
from sklearn.metrics import pairwise_distances
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


SEED = 20260902


def find_file(root: Path, name: str) -> Path:
    matches = list(root.rglob(name))
    if not matches:
        raise FileNotFoundError(f"Could not find {name} below {root}")
    return matches[0]


def parse_first_tag(value: object) -> int | None:
    try:
        tags = ast.literal_eval(str(value))
        return int(tags[0]) if tags else None
    except (ValueError, SyntaxError, TypeError):
        return None


def make_transformer(frame: pd.DataFrame, columns: list[str]) -> ColumnTransformer:
    numeric = [
        c
        for c in columns
        if pd.api.types.is_numeric_dtype(frame[c]) and not c.startswith("onehot_feat")
    ]
    categorical = [c for c in columns if c not in numeric]
    return ColumnTransformer(
        [
            (
                "num",
                make_pipeline(SimpleImputer(strategy="median"), StandardScaler()),
                numeric,
            ),
            (
                "cat",
                make_pipeline(
                    SimpleImputer(strategy="most_frequent"),
                    OneHotEncoder(handle_unknown="ignore", sparse_output=False),
                ),
                categorical,
            ),
        ],
        remainder="drop",
    )


def metric_row(name: str, prediction: np.ndarray, truth: np.ndarray) -> dict[str, float | str]:
    chosen = prediction.argmax(axis=1)
    rows = np.arange(len(truth))
    regret = truth.max(axis=1) - truth[rows, chosen]
    true_best = truth.argmax(axis=1)
    return {
        "method": name,
        "rmse": float(np.sqrt(np.mean((prediction - truth) ** 2))),
        "top1_regret": float(regret.mean()),
        "top1_hit_rate": float(np.mean(chosen == true_best)),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", type=Path, default=Path("data/kuairec/extracted"))
    parser.add_argument("--output", type=Path, default=Path("results/kuairec_propensity_pilot"))
    parser.add_argument("--categories", type=int, default=12)
    parser.add_argument("--radius-quantile", type=float, default=0.12)
    parser.add_argument("--alpha", type=float, default=0.10)
    parser.add_argument("--seed", type=int, default=SEED)
    parser.add_argument(
        "--calibration",
        choices=["peer_mean", "residual", "shrunk_residual"],
        default="peer_mean",
    )
    parser.add_argument("--shrinkage", type=float, default=20.0)
    parser.add_argument("--mismatch-quantile", type=float, default=1.0)
    parser.add_argument("--completion-threshold", type=float, default=1.0)
    parser.add_argument(
        "--block-order",
        choices=["standard", "reverse"],
        default="standard",
        help="Order in which feature blocks refine the neighborhood.",
    )
    args = parser.parse_args()
    started = time.perf_counter()
    args.output.mkdir(parents=True, exist_ok=True)

    small_path = find_file(args.data_root, "small_matrix.csv")
    category_path = find_file(args.data_root, "item_categories.csv")
    user_path = find_file(args.data_root, "user_features.csv")

    categories = pd.read_csv(category_path, usecols=["video_id", "feat"])
    categories["category"] = categories["feat"].map(parse_first_tag)
    categories = categories.dropna(subset=["category"])
    categories["category"] = categories["category"].astype(int)
    top_categories = (
        categories.groupby("category")["video_id"]
        .nunique()
        .nlargest(args.categories)
        .index.to_list()
    )
    video_category = categories.loc[
        categories["category"].isin(top_categories), ["video_id", "category"]
    ].drop_duplicates("video_id")
    catalog_video_count = int(video_category["video_id"].nunique())

    interactions = pd.read_csv(
        small_path,
        usecols=["user_id", "video_id", "watch_ratio"],
        dtype={"user_id": "int32", "video_id": "int32", "watch_ratio": "float32"},
    )
    input_interaction_count = len(interactions)
    interactions = interactions.merge(video_category, on="video_id", how="inner")
    analyzed_video_count = int(interactions["video_id"].nunique())
    interactions["complete"] = (
        interactions["watch_ratio"] >= args.completion_threshold
    ).astype("float32")
    analyzed_interaction_count = len(interactions)
    completion_rate = float(interactions["complete"].mean())
    rates = interactions.pivot_table(
        index="user_id", columns="category", values="complete", aggfunc="mean"
    ).dropna()
    category_order = list(rates.columns)

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
    if args.block_order == "reverse":
        blocks = list(reversed(blocks))
    if len(blocks) < 2:
        raise RuntimeError(f"Too few usable feature blocks: {blocks}")
    model_columns = list(dict.fromkeys(c for block in blocks for c in block))

    rng = np.random.default_rng(args.seed)
    ids = rates.index.to_numpy().copy()
    rng.shuffle(ids)
    n_users = len(ids)
    n_train = int(0.60 * n_users)
    n_reference = int(0.20 * n_users)
    train_ids = ids[:n_train]
    reference_ids = ids[n_train : n_train + n_reference]
    test_ids = ids[n_train + n_reference :]

    transformer = make_transformer(users, model_columns)
    model = make_pipeline(transformer, Ridge(alpha=10.0))
    model.fit(users.loc[train_ids, model_columns], rates.loc[train_ids].to_numpy())
    pred_reference = np.clip(model.predict(users.loc[reference_ids, model_columns]), 0.0, 1.0)
    pred_test = np.clip(model.predict(users.loc[test_ids, model_columns]), 0.0, 1.0)
    y_reference = rates.loc[reference_ids].to_numpy()
    y_test = rates.loc[test_ids].to_numpy()

    cumulative_distances: list[np.ndarray] = []
    cumulative = np.zeros((len(test_ids), len(reference_ids)), dtype=np.float64)
    for block in blocks:
        block_transformer = make_transformer(users.loc[train_ids], block)
        block_transformer.fit(users.loc[train_ids, block])
        ref_matrix = block_transformer.transform(users.loc[reference_ids, block])
        test_matrix = block_transformer.transform(users.loc[test_ids, block])
        distance = pairwise_distances(test_matrix, ref_matrix, metric="euclidean")
        positive = distance[distance > 0]
        scale = float(np.median(positive)) if len(positive) else 1.0
        cumulative = cumulative + distance / max(scale, 1e-12)
        cumulative_distances.append(cumulative.copy())

    radius = float(np.quantile(cumulative_distances[-1], args.radius_quantile))
    k_actions = len(category_order)
    m_levels = len(cumulative_distances)
    complexity_log = np.log(2.0 * k_actions * m_levels / args.alpha)

    fixed_predictions = [np.full_like(y_test, np.nan) for _ in range(m_levels)]
    adaptive_prediction = np.full_like(y_test, np.nan)
    chosen_level = np.full(len(test_ids), -1, dtype=int)
    chosen_support = np.zeros(len(test_ids), dtype=int)
    estimated_objective = np.full((len(test_ids), m_levels), np.inf)
    actual_certificate = np.full((len(test_ids), m_levels), np.nan)
    actual_regret = np.full((len(test_ids), m_levels), np.nan)

    for row in range(len(test_ids)):
        for level, distance in enumerate(cumulative_distances):
            mask = distance[row] <= radius
            support = int(mask.sum())
            if support == 0:
                continue
            if args.calibration == "peer_mean":
                peer_prediction = y_reference[mask].mean(axis=0)
            else:
                residual_adjustment = (y_reference[mask] - pred_reference[mask]).mean(axis=0)
                if args.calibration == "shrunk_residual":
                    residual_adjustment *= support / (support + args.shrinkage)
                peer_prediction = np.clip(pred_test[row] + residual_adjustment, 0.0, 1.0)
            fixed_predictions[level][row] = peer_prediction
            peer_mismatch = np.max(np.abs(pred_reference[mask] - pred_test[row]), axis=1)
            mismatch_hat = float(np.quantile(peer_mismatch, args.mismatch_quantile))
            uncertainty = np.sqrt(complexity_log / (2.0 * support))
            estimated_objective[row, level] = mismatch_hat + uncertainty

            true_mismatch = np.max(np.abs(y_reference[mask] - y_test[row]))
            actual_certificate[row, level] = 2.0 * true_mismatch + 2.0 * uncertainty
            action = int(peer_prediction.argmax())
            actual_regret[row, level] = y_test[row].max() - y_test[row, action]

        available = np.flatnonzero(np.isfinite(estimated_objective[row]))
        if len(available):
            level = int(available[np.argmin(estimated_objective[row, available])])
            adaptive_prediction[row] = fixed_predictions[level][row]
            chosen_level[row] = level
            chosen_support[row] = int((cumulative_distances[level][row] <= radius).sum())

    metrics = [metric_row("base_ridge", pred_test, y_test)]
    for level, prediction in enumerate(fixed_predictions, start=1):
        valid = np.isfinite(prediction).all(axis=1)
        prediction_with_fallback = prediction.copy()
        prediction_with_fallback[~valid] = pred_test[~valid]
        row = metric_row(
            f"fixed_level_{level}_base_fallback", prediction_with_fallback, y_test
        )
        row["coverage"] = float(valid.mean())
        row["mean_support"] = float(
            np.mean((cumulative_distances[level - 1][valid] <= radius).sum(axis=1))
        )
        metrics.append(row)
    valid_adaptive = chosen_level >= 0
    adaptive_metrics = metric_row(
        "adaptive_theorem_rule", adaptive_prediction[valid_adaptive], y_test[valid_adaptive]
    )
    adaptive_metrics["coverage"] = float(valid_adaptive.mean())
    adaptive_metrics["mean_support"] = float(chosen_support[valid_adaptive].mean())
    adaptive_metrics["mean_level"] = float((chosen_level[valid_adaptive] + 1).mean())
    metrics.append(adaptive_metrics)

    chosen_cert = actual_certificate[np.arange(len(test_ids)), np.maximum(chosen_level, 0)]
    chosen_regret = actual_regret[np.arange(len(test_ids)), np.maximum(chosen_level, 0)]
    certificate_coverage = float(
        np.mean(chosen_regret[valid_adaptive] <= chosen_cert[valid_adaptive])
    )
    mean_certificate = float(np.mean(chosen_cert[valid_adaptive]))
    nonvacuous_certificate_fraction = float(np.mean(chosen_cert[valid_adaptive] < 1.0))
    oracle_regret = np.nanmin(actual_regret, axis=1)
    adaptive_oracle_match = float(
        np.mean(chosen_regret[valid_adaptive] <= oracle_regret[valid_adaptive] + 1e-12)
    )
    level_counts = {
        str(level + 1): int(np.sum(chosen_level == level)) for level in range(m_levels)
    }
    summary = {
        "seed": args.seed,
        "runtime_seconds": time.perf_counter() - started,
        "users": n_users,
        "train_users": len(train_ids),
        "reference_users": len(reference_ids),
        "test_users": len(test_ids),
        "input_interactions": input_interaction_count,
        "analyzed_interactions": analyzed_interaction_count,
        "catalog_videos_in_selected_categories": catalog_video_count,
        "analyzed_videos": analyzed_video_count,
        "completion_rate": completion_rate,
        "selected_categories": [int(x) for x in category_order],
        "category_count": k_actions,
        "feature_blocks": blocks,
        "radius_quantile": args.radius_quantile,
        "radius": radius,
        "alpha": args.alpha,
        "calibration": args.calibration,
        "shrinkage": args.shrinkage,
        "mismatch_quantile": args.mismatch_quantile,
        "completion_threshold": args.completion_threshold,
        "block_order": args.block_order,
        "adaptive_level_counts": level_counts,
        "adaptive_certificate_coverage": certificate_coverage,
        "adaptive_mean_certificate": mean_certificate,
        "adaptive_nonvacuous_certificate_fraction": nonvacuous_certificate_fraction,
        "adaptive_oracle_level_match": adaptive_oracle_match,
        "metrics": metrics,
    }
    pd.DataFrame(metrics).to_csv(args.output / "metrics.csv", index=False)
    (args.output / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
