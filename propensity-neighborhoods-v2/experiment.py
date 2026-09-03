"""Chronological KuaiRec experiment for neighborhood-conditioned ranking.

The script keeps test-user future outcomes untouched. Estimation users fit the
behavioral propensity model and the pairwise rankers, tuning users select ranker
regularization and the safe-reranking threshold, and test users are evaluated
once. Multiple seeds are loaded and run in one process to avoid rereading the
large interaction file for every split.
"""

from __future__ import annotations

import argparse
import ast
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import sparse
from scipy.stats import t as student_t
from sklearn.compose import ColumnTransformer
from sklearn.decomposition import TruncatedSVD
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.metrics import ndcg_score, pairwise_distances
from sklearn.model_selection import KFold, cross_val_predict
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


DEFAULT_SEEDS = list(range(20260902, 20260912))
K_GRID = (25, 50, 100, 200)
C_GRID = (0.01, 0.1, 1.0, 10.0)
TAU_GRID = (0.0, 0.0025, 0.005, 0.01, 0.02, 0.05, 0.10)


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
        column
        for column in columns
        if pd.api.types.is_numeric_dtype(frame[column])
        and not column.startswith("onehot_feat")
    ]
    categorical = [column for column in columns if column not in numeric]
    return ColumnTransformer(
        [
            (
                "numeric",
                make_pipeline(SimpleImputer(strategy="median"), StandardScaler()),
                numeric,
            ),
            (
                "categorical",
                make_pipeline(
                    SimpleImputer(strategy="most_frequent"),
                    OneHotEncoder(handle_unknown="ignore", sparse_output=False),
                ),
                categorical,
            ),
        ]
    )


def parse_seeds(value: str) -> list[int]:
    if ":" in value:
        first, last = (int(part) for part in value.split(":", maxsplit=1))
        return list(range(first, last + 1))
    return [int(part) for part in value.split(",") if part.strip()]


def load_chronological_data(
    data_root: Path, categories_count: int, history_fraction: float
) -> dict[str, object]:
    category_path = find_file(data_root, "item_categories.csv")
    small_path = find_file(data_root, "small_matrix.csv")
    user_path = find_file(data_root, "user_features.csv")

    categories = pd.read_csv(category_path, usecols=["video_id", "feat"])
    categories["category"] = categories["feat"].map(parse_first_tag)
    categories = categories.dropna(subset=["category"])
    categories["category"] = categories["category"].astype(int)
    selected_categories = (
        categories.groupby("category")["video_id"]
        .nunique()
        .nlargest(categories_count)
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
    history_mask = position < np.floor(history_fraction * group_size)
    history = interactions.loc[history_mask].copy()
    future = interactions.loc[~history_mask].copy()
    history["complete"] = (history["watch_ratio"] >= 1.0).astype("float32")
    future["complete"] = (future["watch_ratio"] >= 1.0).astype("float32")

    future_rates = future.pivot_table(
        index="user_id", columns="category", values="complete", aggfunc="mean"
    ).dropna()
    users = pd.read_csv(user_path)
    users = users.loc[users["user_id"].isin(future_rates.index)].drop_duplicates(
        "user_id"
    )
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

    preferred_blocks = [
        [
            "user_active_degree",
            "is_lowactive_period",
            "register_days",
            "register_days_range",
        ],
        ["is_live_streamer", "is_video_author"],
        [
            "follow_user_num",
            "follow_user_num_range",
            "fans_user_num",
            "fans_user_num_range",
            "friend_user_num",
            "friend_user_num_range",
        ],
        [column for column in users.columns if column.startswith("onehot_feat")],
    ]
    model_columns = list(
        dict.fromkeys(
            column
            for block in preferred_blocks
            for column in block
            if column in users.columns
        )
    )
    return {
        "users": users,
        "future_rates": future_rates,
        "history_matrix": history_matrix,
        "model_columns": model_columns,
        "categories": [int(value) for value in category_order],
        "history_interactions": int(len(history)),
        "future_interactions": int(len(future)),
    }


def fit_base_model(
    features: np.ndarray,
    outcomes: np.ndarray,
    estimation: np.ndarray,
    tuning: np.ndarray,
    test: np.ndarray,
    seed: int,
    folds: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    splitter = KFold(n_splits=folds, shuffle=True, random_state=seed + 101)
    oof = np.clip(
        cross_val_predict(
            Ridge(alpha=10.0),
            features[estimation],
            outcomes[estimation],
            cv=splitter,
            method="predict",
        ),
        0.0,
        1.0,
    )
    model = Ridge(alpha=10.0).fit(features[estimation], outcomes[estimation])
    tuning_prediction = np.clip(model.predict(features[tuning]), 0.0, 1.0)
    test_prediction = np.clip(model.predict(features[test]), 0.0, 1.0)
    return oof, tuning_prediction, test_prediction


def nearest_indices(distance: np.ndarray, maximum_k: int) -> np.ndarray:
    maximum_k = min(maximum_k, distance.shape[1])
    partial = np.argpartition(distance, maximum_k - 1, axis=1)[:, :maximum_k]
    partial_distance = np.take_along_axis(distance, partial, axis=1)
    order = np.argsort(partial_distance, axis=1)
    return np.take_along_axis(partial, order, axis=1)


def base_feature_tensor(base: np.ndarray) -> tuple[np.ndarray, list[str]]:
    users, actions = base.shape
    order = np.argsort(-base, axis=1)
    ranks = np.empty_like(order)
    ranks[np.arange(users)[:, None], order] = np.arange(actions)
    best = base.max(axis=1, keepdims=True)
    one_hot = np.broadcast_to(np.eye(actions)[None, :, :], (users, actions, actions))
    tensor = np.concatenate(
        [
            base[:, :, None],
            np.square(base)[:, :, None],
            (-ranks / max(actions - 1, 1))[:, :, None],
            (base - best)[:, :, None],
            one_hot,
        ],
        axis=2,
    )
    names = ["base", "base_squared", "negative_rank", "gap_from_base_best"]
    names.extend(f"category_{index}" for index in range(actions))
    return tensor, names


def neighborhood_feature_tensor(
    query_base: np.ndarray,
    reference_base: np.ndarray,
    reference_y: np.ndarray,
    distances: dict[str, np.ndarray],
    k_grid: tuple[int, ...] = K_GRID,
) -> tuple[np.ndarray, list[str]]:
    base_tensor, names = base_feature_tensor(query_base)
    tensors = [base_tensor]
    reference_residual = reference_y - reference_base
    actions = query_base.shape[1]

    for geometry, distance in distances.items():
        neighbor_order = nearest_indices(distance, max(k_grid))
        for requested_k in k_grid:
            k = min(requested_k, neighbor_order.shape[1])
            neighbor = neighbor_order[:, :k]
            local_y = reference_y[neighbor]
            local_residual = reference_residual[neighbor]
            residual_mean = local_residual.mean(axis=1)
            residual_sd = local_residual.std(axis=1)
            outcome_mean = local_y.mean(axis=1)
            local_best = local_y.argmax(axis=2)
            best_share = np.stack(
                [(local_best == action).mean(axis=1) for action in range(actions)],
                axis=1,
            )
            local_gap = np.empty_like(outcome_mean)
            for action in range(actions):
                others = [index for index in range(actions) if index != action]
                local_gap[:, action] = (
                    local_y[:, :, action] - local_y[:, :, others].max(axis=2)
                ).mean(axis=1)
            correction_weight = k / (k + 20.0)
            corrected = np.clip(
                query_base + correction_weight * residual_mean, 0.0, 1.0
            )
            local_distance = np.take_along_axis(distance, neighbor, axis=1)
            distance_mean = local_distance.mean(axis=1, keepdims=True)
            distance_max = local_distance.max(axis=1, keepdims=True)
            block = np.stack(
                [
                    residual_mean,
                    residual_sd,
                    outcome_mean,
                    best_share,
                    local_gap,
                    corrected,
                    corrected * distance_mean,
                    corrected * distance_max,
                ],
                axis=2,
            )
            tensors.append(block)
            names.extend(
                f"{geometry}_k{requested_k}_{statistic}"
                for statistic in (
                    "residual_mean",
                    "residual_sd",
                    "outcome_mean",
                    "best_share",
                    "local_gap",
                    "corrected",
                    "corrected_x_mean_distance",
                    "corrected_x_max_distance",
                )
            )
    return np.concatenate(tensors, axis=2), names


def pairwise_training_data(
    feature_tensor: np.ndarray,
    outcomes: np.ndarray,
    top_only: bool = False,
    candidate_mask: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    if top_only:
        best = outcomes.argmax(axis=1)
        differences = []
        outcome_differences = []
        for row, best_action in enumerate(best):
            other = np.array(
                [action for action in range(outcomes.shape[1]) if action != best_action]
            )
            differences.append(
                feature_tensor[row, best_action][None, :]
                - feature_tensor[row, other, :]
            )
            outcome_differences.append(
                outcomes[row, best_action] - outcomes[row, other]
            )
        difference = np.concatenate(differences, axis=0)
        outcome_difference = np.concatenate(outcome_differences, axis=0)
    else:
        left, right = np.triu_indices(outcomes.shape[1], k=1)
        difference = feature_tensor[:, left, :] - feature_tensor[:, right, :]
        outcome_difference = outcomes[:, left] - outcomes[:, right]
    keep = np.abs(outcome_difference) > 1.0e-12
    if candidate_mask is not None:
        if top_only:
            raise ValueError("candidate_mask and top_only are not used together")
        keep &= candidate_mask[:, left] & candidate_mask[:, right]
    x = difference[keep]
    y = (outcome_difference[keep] > 0).astype(int)
    weight = np.abs(outcome_difference[keep])
    return (
        np.concatenate([x, -x], axis=0),
        np.concatenate([y, 1 - y], axis=0),
        np.concatenate([weight, weight], axis=0),
    )


def ranking_scores(
    model: LogisticRegression, scaler: StandardScaler, features: np.ndarray
) -> np.ndarray:
    # Pairwise differences remove any common centering constant, so only the
    # scale is needed to recover decomposable per-action scores.
    return (features / scaler.scale_[None, None, :]) @ model.coef_[0]


def pairwise_log_loss(scores: np.ndarray, outcomes: np.ndarray) -> float:
    left, right = np.triu_indices(outcomes.shape[1], k=1)
    outcome_difference = outcomes[:, left] - outcomes[:, right]
    score_difference = scores[:, left] - scores[:, right]
    keep = np.abs(outcome_difference) > 1.0e-12
    signs = np.sign(outcome_difference[keep])
    weights = np.abs(outcome_difference[keep])
    losses = np.logaddexp(0.0, -signs * score_difference[keep])
    return float(np.average(losses, weights=weights))


def decision_metrics(
    method: str,
    decision_scores: np.ndarray,
    truth: np.ndarray,
    probability: np.ndarray | None = None,
) -> dict[str, float | str]:
    chosen = decision_scores.argmax(axis=1)
    rows = np.arange(len(truth))
    best = truth.argmax(axis=1)
    return {
        "method": method,
        "rmse": (
            float(np.sqrt(np.mean(np.square(probability - truth))))
            if probability is not None
            else float("nan")
        ),
        "top1_regret": float(np.mean(truth.max(axis=1) - truth[rows, chosen])),
        "top1_hit_rate": float(np.mean(chosen == best)),
        "ndcg": float(ndcg_score(truth, decision_scores)),
        "pairwise_log_loss": float(pairwise_log_loss(decision_scores, truth)),
    }


def tune_ranker(
    train_features: np.ndarray,
    train_y: np.ndarray,
    tuning_features: np.ndarray,
    tuning_y: np.ndarray,
    top_only: bool = False,
) -> tuple[LogisticRegression, StandardScaler, float, list[dict[str, float]]]:
    x_pair, y_pair, weights = pairwise_training_data(
        train_features, train_y, top_only=top_only
    )
    scaler = StandardScaler().fit(x_pair)
    x_scaled = scaler.transform(x_pair)
    candidates: list[tuple[float, float, float, LogisticRegression]] = []
    diagnostics = []
    for c_value in C_GRID:
        model = LogisticRegression(
            C=c_value,
            penalty="l2",
            solver="lbfgs",
            fit_intercept=False,
            max_iter=1000,
            random_state=0,
        ).fit(x_scaled, y_pair, sample_weight=weights)
        scores = ranking_scores(model, scaler, tuning_features)
        metrics = decision_metrics("candidate", scores, tuning_y)
        diagnostics.append(
            {
                "C": c_value,
                "top1_regret": float(metrics["top1_regret"]),
                "top1_hit_rate": float(metrics["top1_hit_rate"]),
                "pairwise_log_loss": float(metrics["pairwise_log_loss"]),
            }
        )
        candidates.append(
            (
                float(metrics["top1_regret"]),
                -float(metrics["top1_hit_rate"]),
                float(metrics["pairwise_log_loss"]),
                model,
            )
        )
    selected = min(candidates, key=lambda item: item[:3])[3]
    return selected, scaler, float(selected.C), diagnostics


def corrected_prediction(
    base: np.ndarray,
    reference_base: np.ndarray,
    reference_y: np.ndarray,
    distance: np.ndarray,
    k: int = 50,
    shrinkage: float = 20.0,
) -> np.ndarray:
    neighbors = nearest_indices(distance, k)[:, : min(k, distance.shape[1])]
    residual = reference_y - reference_base
    correction = residual[neighbors].mean(axis=1)
    k_effective = neighbors.shape[1]
    correction *= k_effective / (k_effective + shrinkage)
    return np.clip(base + correction, 0.0, 1.0)


def restrict_scores(
    ranker_scores: np.ndarray, anchor: np.ndarray, tau: float
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    gap = anchor.max(axis=1, keepdims=True) - anchor
    candidate = gap <= tau + 1.0e-15
    # Keep excluded actions finite so full-ranking diagnostics such as NDCG can
    # still be computed. Their order is inherited from the anchor gap, below
    # every eligible ranker score.
    floor = ranker_scores.min(axis=1, keepdims=True) - 1.0
    restricted = np.where(candidate, ranker_scores, floor - gap)
    chosen = restricted.argmax(axis=1)
    departure = anchor.max(axis=1) - anchor[np.arange(len(anchor)), chosen]
    return restricted, candidate.sum(axis=1), departure


def tune_restriction(
    ranker_scores: np.ndarray, anchor: np.ndarray, truth: np.ndarray
) -> tuple[float, list[tuple[float, float, float, float, float]]]:
    candidates = []
    for tau in TAU_GRID:
        restricted, sizes, departure = restrict_scores(ranker_scores, anchor, tau)
        metric = decision_metrics("candidate", restricted, truth)
        candidates.append(
            (
                float(metric["top1_regret"]),
                -float(metric["top1_hit_rate"]),
                tau,
                float(np.mean(sizes)),
                float(np.mean(departure)),
            )
        )
    selected_tau = float(min(candidates, key=lambda item: item[:3])[2])
    return selected_tau, candidates


def tune_candidate_conditioned_ranker(
    train_features: np.ndarray,
    train_y: np.ndarray,
    train_anchor: np.ndarray,
    tuning_features: np.ndarray,
    tuning_y: np.ndarray,
    tuning_anchor: np.ndarray,
) -> tuple[
    LogisticRegression | None,
    StandardScaler | None,
    float,
    float | None,
    list[dict[str, float]],
]:
    # Tau zero is the unchanged anchor and is retained as a safe fallback.
    candidates: list[
        tuple[float, float, float, float | None, LogisticRegression | None, StandardScaler | None]
    ] = []
    diagnostics = []
    anchor_metric = decision_metrics("anchor", tuning_anchor, tuning_y)
    candidates.append(
        (
            float(anchor_metric["top1_regret"]),
            -float(anchor_metric["top1_hit_rate"]),
            0.0,
            None,
            None,
            None,
        )
    )
    diagnostics.append(
        {
            "tau": 0.0,
            "C": float("nan"),
            "top1_regret": float(anchor_metric["top1_regret"]),
            "top1_hit_rate": float(anchor_metric["top1_hit_rate"]),
        }
    )
    for tau in TAU_GRID[1:]:
        train_mask = train_anchor.max(axis=1, keepdims=True) - train_anchor <= tau
        x_pair, y_pair, weights = pairwise_training_data(
            train_features, train_y, candidate_mask=train_mask
        )
        if len(x_pair) == 0 or len(np.unique(y_pair)) < 2:
            continue
        scaler = StandardScaler().fit(x_pair)
        x_scaled = scaler.transform(x_pair)
        for c_value in C_GRID:
            model = LogisticRegression(
                C=c_value,
                penalty="l2",
                solver="lbfgs",
                fit_intercept=False,
                max_iter=1000,
                random_state=0,
            ).fit(x_scaled, y_pair, sample_weight=weights)
            raw_scores = ranking_scores(model, scaler, tuning_features)
            scores, _, _ = restrict_scores(raw_scores, tuning_anchor, tau)
            metric = decision_metrics("candidate", scores, tuning_y)
            candidates.append(
                (
                    float(metric["top1_regret"]),
                    -float(metric["top1_hit_rate"]),
                    tau,
                    c_value,
                    model,
                    scaler,
                )
            )
            diagnostics.append(
                {
                    "tau": tau,
                    "C": c_value,
                    "top1_regret": float(metric["top1_regret"]),
                    "top1_hit_rate": float(metric["top1_hit_rate"]),
                }
            )
    selected = min(candidates, key=lambda item: item[:4])
    return selected[4], selected[5], float(selected[2]), selected[3], diagnostics


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
    y_estimation, y_tuning, y_test = (
        outcomes[estimation],
        outcomes[tuning],
        outcomes[test],
    )

    svd = TruncatedSVD(n_components=embedding_dimensions, random_state=seed)
    svd.fit(history_matrix[estimation])
    embedding = svd.transform(history_matrix)
    embedding = StandardScaler().fit(embedding[estimation]).transform(embedding)

    transformer = make_transformer(users.iloc[estimation], model_columns)
    transformer.fit(users.iloc[estimation][model_columns])
    static = np.asarray(transformer.transform(users[model_columns]))
    model_features = np.hstack([static, embedding])
    base_oof, base_tuning, base_test = fit_base_model(
        model_features,
        outcomes,
        estimation,
        tuning,
        test,
        seed,
        folds,
    )

    propensity_distances = {
        "estimation": pairwise_distances(base_oof, base_oof, metric="chebyshev"),
        "tuning": pairwise_distances(base_tuning, base_oof, metric="chebyshev"),
        "test": pairwise_distances(base_test, base_oof, metric="chebyshev"),
    }
    np.fill_diagonal(propensity_distances["estimation"], np.inf)
    embedding_distances = {
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
    np.fill_diagonal(embedding_distances["estimation"], np.inf)

    distances_by_split = {
        name: {
            "propensity": propensity_distances[name],
            "embedding": embedding_distances[name],
        }
        for name in ("estimation", "tuning", "test")
    }
    full_estimation, full_feature_names = neighborhood_feature_tensor(
        base_oof,
        base_oof,
        y_estimation,
        distances_by_split["estimation"],
    )
    full_tuning, _ = neighborhood_feature_tensor(
        base_tuning,
        base_oof,
        y_estimation,
        distances_by_split["tuning"],
    )
    full_test, _ = neighborhood_feature_tensor(
        base_test,
        base_oof,
        y_estimation,
        distances_by_split["test"],
    )
    base_estimation, base_feature_names = base_feature_tensor(base_oof)
    base_tuning_features, _ = base_feature_tensor(base_tuning)
    base_test_features, _ = base_feature_tensor(base_test)

    base_ranker, base_scaler, base_c, base_diagnostics = tune_ranker(
        base_estimation,
        y_estimation,
        base_tuning_features,
        y_tuning,
    )
    neighborhood_ranker, neighborhood_scaler, neighborhood_c, neighborhood_diagnostics = (
        tune_ranker(full_estimation, y_estimation, full_tuning, y_tuning)
    )
    base_feature_count = len(base_feature_names)
    propensity_columns = list(range(base_feature_count)) + [
        index
        for index, name in enumerate(full_feature_names)
        if name.startswith("propensity_")
    ]
    embedding_columns = list(range(base_feature_count)) + [
        index
        for index, name in enumerate(full_feature_names)
        if name.startswith("embedding_")
    ]
    propensity_ranker, propensity_scaler, propensity_c, propensity_diagnostics = (
        tune_ranker(
            full_estimation[:, :, propensity_columns],
            y_estimation,
            full_tuning[:, :, propensity_columns],
            y_tuning,
        )
    )
    embedding_ranker, embedding_scaler, embedding_c, embedding_diagnostics = tune_ranker(
        full_estimation[:, :, embedding_columns],
        y_estimation,
        full_tuning[:, :, embedding_columns],
        y_tuning,
    )
    base_top_ranker, base_top_scaler, base_top_c, base_top_diagnostics = tune_ranker(
        base_estimation,
        y_estimation,
        base_tuning_features,
        y_tuning,
        top_only=True,
    )
    propensity_top_ranker, propensity_top_scaler, propensity_top_c, propensity_top_diagnostics = (
        tune_ranker(
            full_estimation[:, :, propensity_columns],
            y_estimation,
            full_tuning[:, :, propensity_columns],
            y_tuning,
            top_only=True,
        )
    )
    base_rank_scores = ranking_scores(base_ranker, base_scaler, base_test_features)
    tuning_base_rank_scores = ranking_scores(
        base_ranker, base_scaler, base_tuning_features
    )
    tuning_neighborhood_scores = ranking_scores(
        neighborhood_ranker, neighborhood_scaler, full_tuning
    )
    test_neighborhood_scores = ranking_scores(
        neighborhood_ranker, neighborhood_scaler, full_test
    )
    test_propensity_rank_scores = ranking_scores(
        propensity_ranker,
        propensity_scaler,
        full_test[:, :, propensity_columns],
    )
    test_embedding_rank_scores = ranking_scores(
        embedding_ranker,
        embedding_scaler,
        full_test[:, :, embedding_columns],
    )
    test_base_top_scores = ranking_scores(
        base_top_ranker, base_top_scaler, base_test_features
    )
    tuning_propensity_top_scores = ranking_scores(
        propensity_top_ranker,
        propensity_top_scaler,
        full_tuning[:, :, propensity_columns],
    )
    test_propensity_top_scores = ranking_scores(
        propensity_top_ranker,
        propensity_top_scaler,
        full_test[:, :, propensity_columns],
    )

    corrected_tuning = corrected_prediction(
        base_tuning,
        base_oof,
        y_estimation,
        propensity_distances["tuning"],
    )
    corrected_test = corrected_prediction(
        base_test,
        base_oof,
        y_estimation,
        propensity_distances["test"],
    )
    corrected_estimation = corrected_prediction(
        base_oof,
        base_oof,
        y_estimation,
        propensity_distances["estimation"],
    )
    (
        candidate_ranker,
        candidate_scaler,
        selected_candidate_tau,
        selected_candidate_c,
        candidate_ranker_diagnostics,
    ) = tune_candidate_conditioned_ranker(
        full_estimation[:, :, propensity_columns],
        y_estimation,
        corrected_estimation,
        full_tuning[:, :, propensity_columns],
        y_tuning,
        corrected_tuning,
    )
    if candidate_ranker is None or candidate_scaler is None:
        candidate_conditioned_test_scores = corrected_test
    else:
        candidate_raw_test_scores = ranking_scores(
            candidate_ranker,
            candidate_scaler,
            full_test[:, :, propensity_columns],
        )
        candidate_conditioned_test_scores, _, _ = restrict_scores(
            candidate_raw_test_scores, corrected_test, selected_candidate_tau
        )
    selected_base_tau, base_tau_candidates = tune_restriction(
        tuning_base_rank_scores, corrected_tuning, y_tuning
    )
    selected_tau, tau_candidates = tune_restriction(
        tuning_neighborhood_scores, corrected_tuning, y_tuning
    )
    selected_top_tau, top_tau_candidates = tune_restriction(
        tuning_propensity_top_scores, corrected_tuning, y_tuning
    )
    restricted_base_test_scores, base_candidate_sizes, base_departure = restrict_scores(
        base_rank_scores, corrected_test, selected_base_tau
    )
    restricted_test_scores, candidate_sizes, departure = restrict_scores(
        test_neighborhood_scores, corrected_test, selected_tau
    )
    restricted_top_test_scores, top_candidate_sizes, top_departure = restrict_scores(
        test_propensity_top_scores, corrected_test, selected_top_tau
    )

    metrics = [
        decision_metrics("base_propensity", base_test, y_test, base_test),
        decision_metrics(
            "fixed_propensity_peer_mean", corrected_test, y_test, corrected_test
        ),
        decision_metrics("base_pairwise_ranker", base_rank_scores, y_test),
        decision_metrics("base_top_pairwise_ranker", test_base_top_scores, y_test),
        decision_metrics(
            "restricted_base_pairwise_ranker", restricted_base_test_scores, y_test
        ),
        decision_metrics(
            "propensity_neighborhood_ranker", test_propensity_rank_scores, y_test
        ),
        decision_metrics(
            "propensity_top_pairwise_ranker", test_propensity_top_scores, y_test
        ),
        decision_metrics(
            "embedding_neighborhood_ranker", test_embedding_rank_scores, y_test
        ),
        decision_metrics(
            "neighborhood_pairwise_ranker", test_neighborhood_scores, y_test
        ),
        decision_metrics(
            "restricted_neighborhood_ranker", restricted_test_scores, y_test
        ),
        decision_metrics(
            "restricted_propensity_top_ranker", restricted_top_test_scores, y_test
        ),
        decision_metrics(
            "candidate_conditioned_propensity_ranker",
            candidate_conditioned_test_scores,
            y_test,
        ),
    ]
    anchor_choice = corrected_test.argmax(axis=1)
    neighborhood_choice = test_neighborhood_scores.argmax(axis=1)
    restricted_choice = restricted_test_scores.argmax(axis=1)
    unrestricted_departure = corrected_test.max(axis=1) - corrected_test[
        np.arange(len(test)), neighborhood_choice
    ]
    summary: dict[str, object] = {
        "seed": seed,
        "runtime_seconds": time.perf_counter() - started,
        "users": len(users),
        "estimation_users": len(estimation),
        "tuning_users": len(tuning),
        "test_users": len(test),
        "categories": data["categories"],
        "embedding_dimensions": embedding_dimensions,
        "k_grid": list(K_GRID),
        "ranker_c_grid": list(C_GRID),
        "tau_grid": list(TAU_GRID),
        "selected_base_ranker_c": base_c,
        "selected_neighborhood_ranker_c": neighborhood_c,
        "selected_propensity_ranker_c": propensity_c,
        "selected_embedding_ranker_c": embedding_c,
        "selected_base_top_ranker_c": base_top_c,
        "selected_propensity_top_ranker_c": propensity_top_c,
        "selected_base_tau": selected_base_tau,
        "selected_tau": selected_tau,
        "selected_top_tau": selected_top_tau,
        "selected_candidate_conditioned_tau": selected_candidate_tau,
        "selected_candidate_conditioned_c": selected_candidate_c,
        "mean_base_candidate_set_size": float(np.mean(base_candidate_sizes)),
        "mean_candidate_set_size": float(np.mean(candidate_sizes)),
        "mean_top_candidate_set_size": float(np.mean(top_candidate_sizes)),
        "restricted_base_override_rate": float(
            np.mean(restricted_base_test_scores.argmax(axis=1) != anchor_choice)
        ),
        "restricted_override_rate": float(np.mean(restricted_choice != anchor_choice)),
        "unrestricted_override_rate": float(
            np.mean(neighborhood_choice != anchor_choice)
        ),
        "mean_restricted_departure_xi": float(np.mean(departure)),
        "max_restricted_departure_xi": float(np.max(departure)),
        "mean_restricted_top_departure_xi": float(np.mean(top_departure)),
        "max_restricted_top_departure_xi": float(np.max(top_departure)),
        "mean_restricted_base_departure_xi": float(np.mean(base_departure)),
        "max_restricted_base_departure_xi": float(np.max(base_departure)),
        "mean_unrestricted_departure_xi": float(np.mean(unrestricted_departure)),
        "max_unrestricted_departure_xi": float(np.max(unrestricted_departure)),
        "base_feature_count": len(base_feature_names),
        "neighborhood_feature_count": len(full_feature_names),
        "base_ranker_tuning": base_diagnostics,
        "neighborhood_ranker_tuning": neighborhood_diagnostics,
        "propensity_ranker_tuning": propensity_diagnostics,
        "embedding_ranker_tuning": embedding_diagnostics,
        "base_top_ranker_tuning": base_top_diagnostics,
        "propensity_top_ranker_tuning": propensity_top_diagnostics,
        "candidate_conditioned_ranker_tuning": candidate_ranker_diagnostics,
        "base_tau_tuning": [
            {
                "tau": float(candidate[2]),
                "top1_regret": float(candidate[0]),
                "top1_hit_rate": float(-candidate[1]),
                "mean_candidate_set_size": float(candidate[3]),
                "mean_departure_xi": float(candidate[4]),
            }
            for candidate in base_tau_candidates
        ],
        "tau_tuning": [
            {
                "tau": float(candidate[2]),
                "top1_regret": float(candidate[0]),
                "top1_hit_rate": float(-candidate[1]),
                "mean_candidate_set_size": float(candidate[3]),
                "mean_departure_xi": float(candidate[4]),
            }
            for candidate in tau_candidates
        ],
        "top_tau_tuning": [
            {
                "tau": float(candidate[2]),
                "top1_regret": float(candidate[0]),
                "top1_hit_rate": float(-candidate[1]),
                "mean_candidate_set_size": float(candidate[3]),
                "mean_departure_xi": float(candidate[4]),
            }
            for candidate in top_tau_candidates
        ],
        "metrics": metrics,
    }
    return summary


def confidence_interval(values: np.ndarray) -> tuple[float, float]:
    if len(values) < 2:
        return float("nan"), float("nan")
    half = float(
        student_t.ppf(0.975, df=len(values) - 1)
        * values.std(ddof=1)
        / np.sqrt(len(values))
    )
    return float(values.mean() - half), float(values.mean() + half)


def aggregate_results(split_metrics: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, object]]:
    numeric = [
        "rmse",
        "top1_regret",
        "top1_hit_rate",
        "ndcg",
        "pairwise_log_loss",
    ]
    aggregate = split_metrics.groupby("method", sort=False)[numeric].mean().reset_index()
    pivot = split_metrics.pivot(index="seed", columns="method", values="top1_regret")
    base = pivot["base_propensity"]
    comparisons = {}
    for method in pivot.columns:
        if method == "base_propensity":
            continue
        delta = pivot[method] - base
        lower, upper = confidence_interval(delta.to_numpy())
        comparisons[method] = {
            "mean_regret_change_vs_base": float(delta.mean()),
            "relative_regret_change_vs_base": float(delta.mean() / base.mean()),
            "paired_95pct_interval": [lower, upper],
            "regret_wins_vs_base": int((delta < 0).sum()),
            "splits": int(len(delta)),
        }
    incremental_pairs = (
        ("propensity_top_pairwise_ranker", "base_top_pairwise_ranker"),
        ("propensity_neighborhood_ranker", "base_pairwise_ranker"),
        ("embedding_neighborhood_ranker", "base_pairwise_ranker"),
        ("neighborhood_pairwise_ranker", "base_pairwise_ranker"),
        ("restricted_neighborhood_ranker", "restricted_base_pairwise_ranker"),
        ("restricted_neighborhood_ranker", "fixed_propensity_peer_mean"),
        ("restricted_propensity_top_ranker", "fixed_propensity_peer_mean"),
        ("candidate_conditioned_propensity_ranker", "fixed_propensity_peer_mean"),
    )
    incremental_comparisons = {}
    for method, reference in incremental_pairs:
        delta = pivot[method] - pivot[reference]
        lower, upper = confidence_interval(delta.to_numpy())
        incremental_comparisons[f"{method}_vs_{reference}"] = {
            "mean_regret_change": float(delta.mean()),
            "paired_95pct_interval": [lower, upper],
            "regret_wins": int((delta < 0).sum()),
            "splits": int(len(delta)),
        }
    information = split_metrics.pivot(
        index="seed", columns="method", values="pairwise_log_loss"
    )
    logloss_gain = (
        information["base_pairwise_ranker"]
        - information["neighborhood_pairwise_ranker"]
    )
    lower, upper = confidence_interval(logloss_gain.to_numpy())
    summary = {
        "splits": int(split_metrics["seed"].nunique()),
        "comparisons": comparisons,
        "incremental_comparisons": incremental_comparisons,
        "neighborhood_pairwise_logloss_reduction_vs_base_ranker": {
            "mean": float(logloss_gain.mean()),
            "paired_95pct_interval": [lower, upper],
            "wins": int((logloss_gain > 0).sum()),
        },
    }
    return aggregate, summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--data-root", type=Path, default=Path("data/kuairec/extracted")
    )
    parser.add_argument(
        "--output", type=Path, default=Path("propensity-neighborhoods-v2/results")
    )
    parser.add_argument(
        "--seeds",
        type=parse_seeds,
        default=DEFAULT_SEEDS,
        help="Comma-separated seeds or an inclusive range such as 20260902:20260911.",
    )
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
    split_summaries = []
    for seed in args.seeds:
        print(f"Running seed {seed}...", flush=True)
        summary = run_split(data, seed, args.embedding_dimensions, args.folds)
        split_directory = args.output / f"seed_{seed}"
        split_directory.mkdir(parents=True, exist_ok=True)
        (split_directory / "summary.json").write_text(
            json.dumps(summary, indent=2), encoding="utf-8"
        )
        rows = pd.DataFrame(summary["metrics"])
        rows.insert(0, "seed", seed)
        rows.to_csv(split_directory / "metrics.csv", index=False)
        all_rows.append(rows)
        split_summaries.append(summary)
        print(
            f"seed={seed} runtime={summary['runtime_seconds']:.1f}s "
            f"tau={summary['selected_tau']}",
            flush=True,
        )

    split_metrics = pd.concat(all_rows, ignore_index=True)
    split_metrics.to_csv(args.output / "split_metrics.csv", index=False)
    aggregate, aggregate_summary = aggregate_results(split_metrics)
    aggregate.to_csv(args.output / "aggregate_metrics.csv", index=False)
    aggregate_summary.update(
        {
            "seeds": list(args.seeds),
            "overall_runtime_seconds": time.perf_counter() - overall_started,
            "history_interactions": data["history_interactions"],
            "future_interactions": data["future_interactions"],
            "selected_ranker_c": [
                summary["selected_neighborhood_ranker_c"]
                for summary in split_summaries
            ],
            "selected_tau": [summary["selected_tau"] for summary in split_summaries],
            "selected_base_tau": [
                summary["selected_base_tau"] for summary in split_summaries
            ],
            "mean_candidate_set_size": float(
                np.mean(
                    [summary["mean_candidate_set_size"] for summary in split_summaries]
                )
            ),
            "mean_restricted_override_rate": float(
                np.mean(
                    [summary["restricted_override_rate"] for summary in split_summaries]
                )
            ),
        }
    )
    (args.output / "aggregate_summary.json").write_text(
        json.dumps(aggregate_summary, indent=2), encoding="utf-8"
    )
    print(aggregate.to_string(index=False), flush=True)
    print(json.dumps(aggregate_summary, indent=2), flush=True)


if __name__ == "__main__":
    main()
