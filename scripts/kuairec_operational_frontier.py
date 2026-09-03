"""Operational split-conformal test for propensity-neighborhood refinement.

The experiment uses five disjoint user sets:

1. fit the base propensity model;
2. fit a model of the base model's residual vector;
3. supply observed reference-peer residuals;
4. calibrate a simultaneous residual-prediction radius; and
5. evaluate recommendations and certificates.

The conformal certificate targets the observed outcome vector for a held-out KuaiRec user,
not the latent conditional propensity vector.  The target user's outcome is never
used to choose a neighborhood or construct its certificate.
"""

from __future__ import annotations

import argparse
import ast
import json
import math
import time
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge
from sklearn.metrics import pairwise_distances
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


DEFAULT_SEEDS = list(range(20260902, 20260922))


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


def conformal_quantile(scores: np.ndarray, alpha: float) -> float:
    """Finite-sample split-conformal quantile with the standard +1 correction."""
    clean = np.asarray(scores, dtype=float)
    if clean.ndim != 1 or len(clean) == 0 or not np.isfinite(clean).all():
        raise ValueError("Conformal scores must be a nonempty finite vector")
    rank = int(math.ceil((len(clean) + 1) * (1.0 - alpha)))
    rank = min(max(rank, 1), len(clean))
    return float(np.partition(clean, rank - 1)[rank - 1])


def metric_row(name: str, prediction: np.ndarray, truth: np.ndarray) -> dict[str, float | str]:
    chosen = prediction.argmax(axis=1)
    rows = np.arange(len(truth))
    regret = truth.max(axis=1) - truth[rows, chosen]
    return {
        "method": name,
        "rmse": float(np.sqrt(np.mean((prediction - truth) ** 2))),
        "top1_regret": float(regret.mean()),
        "top1_hit_rate": float(np.mean(chosen == truth.argmax(axis=1))),
    }


def load_data(
    data_root: Path, categories_count: int, completion_threshold: float
) -> tuple[pd.DataFrame, pd.DataFrame, list[list[str]], dict[str, object]]:
    small_path = find_file(data_root, "small_matrix.csv")
    category_path = find_file(data_root, "item_categories.csv")
    user_path = find_file(data_root, "user_features.csv")

    categories = pd.read_csv(category_path, usecols=["video_id", "feat"])
    categories["category"] = categories["feat"].map(parse_first_tag)
    categories = categories.dropna(subset=["category"])
    categories["category"] = categories["category"].astype(int)
    top_categories = (
        categories.groupby("category")["video_id"]
        .nunique()
        .nlargest(categories_count)
        .index.to_list()
    )
    video_category = categories.loc[
        categories["category"].isin(top_categories), ["video_id", "category"]
    ].drop_duplicates("video_id")

    interactions = pd.read_csv(
        small_path,
        usecols=["user_id", "video_id", "watch_ratio"],
        dtype={"user_id": "int32", "video_id": "int32", "watch_ratio": "float32"},
    )
    input_interactions = len(interactions)
    interactions = interactions.merge(video_category, on="video_id", how="inner")
    interactions["complete"] = (
        interactions["watch_ratio"] >= completion_threshold
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
    metadata = {
        "users": len(users),
        "input_interactions": input_interactions,
        "analyzed_interactions": len(interactions),
        "categories": [int(category) for category in rates.columns],
        "feature_blocks": blocks,
    }
    return users, rates, blocks, metadata


def cumulative_distances(
    users: pd.DataFrame,
    fit_ids: np.ndarray,
    query_ids: np.ndarray,
    reference_ids: np.ndarray,
    blocks: list[list[str]],
) -> list[np.ndarray]:
    cumulative = np.zeros((len(query_ids), len(reference_ids)), dtype=np.float64)
    result: list[np.ndarray] = []
    for block in blocks:
        transformer = make_transformer(users.loc[fit_ids], block)
        transformer.fit(users.loc[fit_ids, block])
        reference = transformer.transform(users.loc[reference_ids, block])
        query = transformer.transform(users.loc[query_ids, block])
        distance = pairwise_distances(query, reference, metric="euclidean")
        positive = distance[distance > 0]
        scale = float(np.median(positive)) if len(positive) else 1.0
        cumulative = cumulative + distance / max(scale, 1e-12)
        result.append(cumulative.copy())
    return result


def build_candidates(
    base_prediction: np.ndarray,
    residual_prediction: np.ndarray,
    reference_prediction: np.ndarray,
    reference_residual: np.ndarray,
    distances: list[np.ndarray],
    radius: float,
    complexity_log: float,
    shrinkage: float,
) -> dict[str, np.ndarray]:
    n_query, actions = base_prediction.shape
    levels = len(distances)
    predictions = np.full((n_query, levels + 1, actions), np.nan)
    objectives = np.full((n_query, levels + 1), np.inf)
    bias_proxy = np.full((n_query, levels + 1), np.inf)
    supports = np.zeros((n_query, levels + 1), dtype=int)
    old_objectives = np.full((n_query, levels), np.inf)

    predictions[:, 0] = base_prediction
    bias_proxy[:, 0] = np.max(np.abs(residual_prediction), axis=1)
    objectives[:, 0] = bias_proxy[:, 0]

    for row in range(n_query):
        for level, distance in enumerate(distances):
            mask = distance[row] <= radius
            support = int(mask.sum())
            if support == 0:
                continue
            weight = support / (support + shrinkage)
            correction = weight * reference_residual[mask].mean(axis=0)
            prediction = np.clip(base_prediction[row] + correction, 0.0, 1.0)
            proxy = float(np.max(np.abs(residual_prediction[row] - correction)))
            neighbor_envelope = float(
                np.max(np.abs(reference_residual[mask] - residual_prediction[row]))
            )
            uncertainty = weight * math.sqrt(complexity_log / (2.0 * support))
            predictions[row, level + 1] = prediction
            bias_proxy[row, level + 1] = proxy
            # The theorem-faithful observable envelope.  If the target residual
            # lies within eta of rhohat(x), then shrinkage gives
            # ||R_x-w mean(R_i)|| <= eta + this objective.
            objectives[row, level + 1] = (
                weight * neighbor_envelope
                + (1.0 - weight) * np.max(np.abs(residual_prediction[row]))
            )
            supports[row, level + 1] = support

            score_distance = np.max(
                np.abs(reference_prediction[mask] - base_prediction[row]), axis=1
            )
            old_objectives[row, level] = float(score_distance.max()) + uncertainty

    selected = objectives.argmin(axis=1)
    selected_prediction = predictions[np.arange(n_query), selected]
    selected_proxy = bias_proxy[np.arange(n_query), selected]
    selected_objective = objectives[np.arange(n_query), selected]
    selected_support = supports[np.arange(n_query), selected]

    # For a peer-mean correction, linearity gives the tighter observable term
    # ||rhohat(x)-w mean(R_i)|| directly; no max-neighbor relaxation is needed.
    mean_selected = bias_proxy.argmin(axis=1)
    mean_selected_prediction = predictions[np.arange(n_query), mean_selected]
    mean_selected_proxy = bias_proxy[np.arange(n_query), mean_selected]
    mean_selected_support = supports[np.arange(n_query), mean_selected]

    old_selected_zero = old_objectives.argmin(axis=1)
    old_selected = old_selected_zero + 1
    old_prediction = predictions[np.arange(n_query), old_selected]

    return {
        "predictions": predictions,
        "objectives": objectives,
        "bias_proxy": bias_proxy,
        "supports": supports,
        "selected": selected,
        "selected_prediction": selected_prediction,
        "selected_proxy": selected_proxy,
        "selected_objective": selected_objective,
        "selected_support": selected_support,
        "mean_selected": mean_selected,
        "mean_selected_prediction": mean_selected_prediction,
        "mean_selected_proxy": mean_selected_proxy,
        "mean_selected_support": mean_selected_support,
        "old_selected": old_selected,
        "old_prediction": old_prediction,
    }


def run_seed(
    users: pd.DataFrame,
    rates: pd.DataFrame,
    blocks: list[list[str]],
    seed: int,
    alpha: float,
    radius_quantile: float,
    shrinkage: float,
    forest_trees: int,
) -> tuple[dict[str, object], list[dict[str, object]]]:
    rng = np.random.default_rng(seed)
    ids = rates.index.to_numpy().copy()
    rng.shuffle(ids)
    n = len(ids)
    cuts = np.cumsum(
        [int(0.45 * n), int(0.15 * n), int(0.15 * n), int(0.10 * n)]
    )
    base_ids = ids[: cuts[0]]
    residual_ids = ids[cuts[0] : cuts[1]]
    reference_ids = ids[cuts[1] : cuts[2]]
    conformal_ids = ids[cuts[2] : cuts[3]]
    test_ids = ids[cuts[3] :]
    model_columns = list(dict.fromkeys(c for block in blocks for c in block))

    base_model = make_pipeline(
        make_transformer(users.loc[base_ids], model_columns), Ridge(alpha=10.0)
    )
    base_model.fit(users.loc[base_ids, model_columns], rates.loc[base_ids].to_numpy())

    all_secondary_ids = np.concatenate(
        [residual_ids, reference_ids, conformal_ids, test_ids]
    )
    all_base_prediction = np.clip(
        base_model.predict(users.loc[all_secondary_ids, model_columns]), 0.0, 1.0
    )
    offsets: dict[str, tuple[int, int]] = {}
    cursor = 0
    for name, split_ids in [
        ("residual", residual_ids),
        ("reference", reference_ids),
        ("conformal", conformal_ids),
        ("test", test_ids),
    ]:
        offsets[name] = (cursor, cursor + len(split_ids))
        cursor += len(split_ids)

    def base_for(name: str) -> np.ndarray:
        lo, hi = offsets[name]
        return all_base_prediction[lo:hi]

    residual_transformer = make_transformer(users.loc[residual_ids], model_columns)
    residual_features = residual_transformer.fit_transform(
        users.loc[residual_ids, model_columns]
    )
    residual_features = np.column_stack([residual_features, base_for("residual")])
    residual_truth = rates.loc[residual_ids].to_numpy() - base_for("residual")
    residual_model = RandomForestRegressor(
        n_estimators=forest_trees,
        min_samples_leaf=8,
        max_features=0.7,
        random_state=seed,
        n_jobs=-1,
    )
    residual_model.fit(residual_features, residual_truth)

    def residual_prediction(split_ids: np.ndarray, name: str) -> np.ndarray:
        transformed = residual_transformer.transform(users.loc[split_ids, model_columns])
        features = np.column_stack([transformed, base_for(name)])
        return residual_model.predict(features)

    rhohat_conformal = residual_prediction(conformal_ids, "conformal")
    rhohat_test = residual_prediction(test_ids, "test")
    y_conformal = rates.loc[conformal_ids].to_numpy()
    y_test = rates.loc[test_ids].to_numpy()
    residual_conformal = y_conformal - base_for("conformal")
    residual_test = y_test - base_for("test")
    residual_scores = np.max(np.abs(residual_conformal - rhohat_conformal), axis=1)
    eta = conformal_quantile(residual_scores, alpha)

    distance_cal = cumulative_distances(
        users, base_ids, conformal_ids, reference_ids, blocks
    )
    distance_test = cumulative_distances(users, base_ids, test_ids, reference_ids, blocks)
    radius_source = cumulative_distances(
        users, base_ids, residual_ids, reference_ids, blocks
    )[-1]
    radius = float(np.quantile(radius_source, radius_quantile))

    y_reference = rates.loc[reference_ids].to_numpy()
    pred_reference = base_for("reference")
    reference_residual = y_reference - pred_reference
    complexity_log = math.log(2.0 * rates.shape[1] * len(blocks) / alpha)

    conformal_candidates = build_candidates(
        base_for("conformal"),
        rhohat_conformal,
        pred_reference,
        reference_residual,
        distance_cal,
        radius,
        complexity_log,
        shrinkage,
    )
    test_candidates = build_candidates(
        base_for("test"),
        rhohat_test,
        pred_reference,
        reference_residual,
        distance_test,
        radius,
        complexity_log,
        shrinkage,
    )

    # The theorem-faithful direct certificate uses the max-neighbor envelope.
    width = eta + test_candidates["selected_objective"]
    selected_prediction = test_candidates["selected_prediction"]
    max_error = np.max(np.abs(selected_prediction - y_test), axis=1)
    chosen = selected_prediction.argmax(axis=1)
    rows = np.arange(len(test_ids))
    regret = y_test.max(axis=1) - y_test[rows, chosen]
    lead = np.partition(selected_prediction, -2, axis=1)[:, -1] - np.partition(
        selected_prediction, -2, axis=1
    )[:, -2]
    certified = lead > 2.0 * width

    # Peer averaging permits a tighter operational certificate:
    # ||R_x-c_m|| <= ||R_x-rhohat(x)|| + ||rhohat(x)-c_m||.
    mean_selected_prediction = test_candidates["mean_selected_prediction"]
    mean_width = eta + test_candidates["mean_selected_proxy"]
    mean_max_error = np.max(np.abs(mean_selected_prediction - y_test), axis=1)
    mean_chosen = mean_selected_prediction.argmax(axis=1)
    mean_regret = y_test.max(axis=1) - y_test[rows, mean_chosen]

    # A second, end-to-end split-conformal correction calibrates any remaining
    # slack for the complete peer-mean selection algorithm.
    conformal_selected = conformal_candidates["mean_selected_prediction"]
    conformal_error = np.max(np.abs(conformal_selected - y_conformal), axis=1)
    slack_scores = conformal_error - conformal_candidates["mean_selected_proxy"]
    slack = conformal_quantile(slack_scores, alpha)
    e2e_width = np.maximum(0.0, test_candidates["mean_selected_proxy"] + slack)

    base_scores = np.max(np.abs(y_conformal - base_for("conformal")), axis=1)
    base_width = conformal_quantile(base_scores, alpha)
    base_prediction = base_for("test")
    base_error = np.max(np.abs(base_prediction - y_test), axis=1)
    base_chosen = base_prediction.argmax(axis=1)
    base_regret = y_test.max(axis=1) - y_test[rows, base_chosen]

    old_prediction = test_candidates["old_prediction"]
    old_selected = test_candidates["old_selected"]
    old_regret = y_test.max(axis=1) - y_test[rows, old_prediction.argmax(axis=1)]
    old_cert = np.empty(len(test_ids))
    for row, selected_level in enumerate(old_selected):
        level = int(selected_level - 1)
        mask = distance_test[level][row] <= radius
        support = int(mask.sum())
        weight = support / (support + shrinkage)
        realized_mismatch = float(
            np.max(np.abs(y_reference[mask] - y_test[row]))
        )
        uncertainty = weight * math.sqrt(complexity_log / (2.0 * support))
        old_cert[row] = 2.0 * (realized_mismatch + uncertainty)

    level_metrics: list[dict[str, object]] = []
    for level in range(len(blocks) + 1):
        prediction = test_candidates["predictions"][:, level]
        valid = np.isfinite(prediction).all(axis=1)
        if not valid.any():
            continue
        row_metric = metric_row(
            "base" if level == 0 else f"fixed_level_{level}",
            prediction[valid],
            y_test[valid],
        )
        row_metric.update(
            {
                "seed": seed,
                "coverage": float(valid.mean()),
                "mean_support": 0.0
                if level == 0
                else float(test_candidates["supports"][valid, level].mean()),
            }
        )
        level_metrics.append(row_metric)
    operational_metric = metric_row(
        "operational_theorem_frontier", selected_prediction, y_test
    )
    operational_metric.update(
        {
            "seed": seed,
            "coverage": 1.0,
            "mean_support": float(test_candidates["selected_support"].mean()),
        }
    )
    level_metrics.append(operational_metric)
    mean_metric = metric_row(
        "operational_mean_frontier", mean_selected_prediction, y_test
    )
    mean_metric.update(
        {
            "seed": seed,
            "coverage": 1.0,
            "mean_support": float(test_candidates["mean_selected_support"].mean()),
        }
    )
    level_metrics.append(mean_metric)
    old_metric = metric_row("old_score_proxy", old_prediction, y_test)
    old_metric.update(
        {
            "seed": seed,
            "coverage": 1.0,
            "mean_support": float(
                test_candidates["supports"][np.arange(len(test_ids)), old_selected].mean()
            ),
        }
    )
    level_metrics.append(old_metric)

    result = {
        "seed": seed,
        "split_sizes": {
            "base": len(base_ids),
            "residual": len(residual_ids),
            "reference": len(reference_ids),
            "conformal": len(conformal_ids),
            "test": len(test_ids),
        },
        "radius": radius,
        "residual_band_radius": eta,
        "residual_band_coverage": float(
            np.mean(np.max(np.abs(residual_test - rhohat_test), axis=1) <= eta)
        ),
        "direct_max_error_coverage": float(np.mean(max_error <= width)),
        "direct_regret_coverage": float(np.mean(regret <= 2.0 * width)),
        "direct_mean_regret_bound": float(np.mean(2.0 * width)),
        "direct_nonvacuous_fraction": float(np.mean(2.0 * width < 1.0)),
        "direct_certified_fraction": float(certified.mean()),
        "direct_certified_correct_fraction": None
        if not certified.any()
        else float(np.mean(chosen[certified] == y_test[certified].argmax(axis=1))),
        "mean_direct_max_error_coverage": float(
            np.mean(mean_max_error <= mean_width)
        ),
        "mean_direct_regret_coverage": float(
            np.mean(mean_regret <= 2.0 * mean_width)
        ),
        "mean_direct_mean_regret_bound": float(np.mean(2.0 * mean_width)),
        "mean_direct_nonvacuous_fraction": float(np.mean(2.0 * mean_width < 1.0)),
        "e2e_slack": slack,
        "e2e_max_error_coverage": float(np.mean(max_error <= e2e_width)),
        "e2e_regret_coverage": float(np.mean(regret <= 2.0 * e2e_width)),
        "e2e_mean_regret_bound": float(np.mean(2.0 * e2e_width)),
        "e2e_nonvacuous_fraction": float(np.mean(2.0 * e2e_width < 1.0)),
        "base_conformal_width": base_width,
        "base_max_error_coverage": float(np.mean(base_error <= base_width)),
        "base_regret_coverage": float(np.mean(base_regret <= 2.0 * base_width)),
        "base_mean_regret_bound": float(2.0 * base_width),
        "old_retrospective_regret_coverage": float(np.mean(old_regret <= old_cert)),
        "old_retrospective_mean_regret_bound": float(old_cert.mean()),
        "old_retrospective_nonvacuous_fraction": float(np.mean(old_cert < 1.0)),
        "selected_counts": {
            int(level): int(count)
            for level, count in sorted(
                Counter(test_candidates["selected"].astype(int)).items()
            )
        },
        "mean_selected_counts": {
            int(level): int(count)
            for level, count in sorted(
                Counter(test_candidates["mean_selected"].astype(int)).items()
            )
        },
        "old_selected_counts": {
            int(level): int(count)
            for level, count in sorted(
                Counter(test_candidates["old_selected"].astype(int)).items()
            )
        },
    }
    return result, level_metrics


def aggregate(results: list[dict[str, object]], metrics: pd.DataFrame) -> dict[str, object]:
    scalar_keys = [
        "residual_band_radius",
        "residual_band_coverage",
        "direct_max_error_coverage",
        "direct_regret_coverage",
        "direct_mean_regret_bound",
        "direct_nonvacuous_fraction",
        "direct_certified_fraction",
        "mean_direct_max_error_coverage",
        "mean_direct_regret_coverage",
        "mean_direct_mean_regret_bound",
        "mean_direct_nonvacuous_fraction",
        "e2e_slack",
        "e2e_max_error_coverage",
        "e2e_regret_coverage",
        "e2e_mean_regret_bound",
        "e2e_nonvacuous_fraction",
        "base_conformal_width",
        "base_max_error_coverage",
        "base_regret_coverage",
        "base_mean_regret_bound",
        "old_retrospective_regret_coverage",
        "old_retrospective_mean_regret_bound",
        "old_retrospective_nonvacuous_fraction",
    ]
    summary: dict[str, object] = {}
    for key in scalar_keys:
        values = np.array([float(result[key]) for result in results], dtype=float)
        summary[key] = {
            "mean": float(values.mean()),
            "std": float(values.std(ddof=1)),
            "min": float(values.min()),
            "max": float(values.max()),
        }
    grouped = (
        metrics.groupby("method")
        .agg(
            rmse_mean=("rmse", "mean"),
            rmse_std=("rmse", "std"),
            regret_mean=("top1_regret", "mean"),
            regret_std=("top1_regret", "std"),
            hit_rate_mean=("top1_hit_rate", "mean"),
            hit_rate_std=("top1_hit_rate", "std"),
            coverage_mean=("coverage", "mean"),
            mean_support=("mean_support", "mean"),
        )
        .reset_index()
    )
    summary["metrics"] = grouped.to_dict(orient="records")
    selected_total: Counter[int] = Counter()
    mean_selected_total: Counter[int] = Counter()
    old_selected_total: Counter[int] = Counter()
    for result in results:
        selected_total.update({int(k): int(v) for k, v in result["selected_counts"].items()})
        mean_selected_total.update(
            {int(k): int(v) for k, v in result["mean_selected_counts"].items()}
        )
        old_selected_total.update(
            {int(k): int(v) for k, v in result["old_selected_counts"].items()}
        )
    summary["selected_counts"] = dict(sorted(selected_total.items()))
    summary["mean_selected_counts"] = dict(sorted(mean_selected_total.items()))
    summary["old_selected_counts"] = dict(sorted(old_selected_total.items()))
    return summary


def parse_seeds(value: str) -> list[int]:
    if ":" in value:
        start, stop = (int(part) for part in value.split(":", maxsplit=1))
        return list(range(start, stop + 1))
    return [int(part) for part in value.split(",")]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", type=Path, default=Path("data/kuairec/extracted"))
    parser.add_argument(
        "--output", type=Path, default=Path("results/kuairec_operational_frontier")
    )
    parser.add_argument("--categories", type=int, default=12)
    parser.add_argument("--completion-threshold", type=float, default=1.0)
    parser.add_argument("--radius-quantile", type=float, default=0.12)
    parser.add_argument("--alpha", type=float, default=0.10)
    parser.add_argument("--shrinkage", type=float, default=20.0)
    parser.add_argument("--forest-trees", type=int, default=200)
    parser.add_argument(
        "--seeds",
        type=parse_seeds,
        default=DEFAULT_SEEDS,
        help="Comma-separated seeds or inclusive start:stop range.",
    )
    args = parser.parse_args()
    started = time.perf_counter()
    args.output.mkdir(parents=True, exist_ok=True)

    users, rates, blocks, metadata = load_data(
        args.data_root, args.categories, args.completion_threshold
    )
    results: list[dict[str, object]] = []
    metric_rows: list[dict[str, object]] = []
    for seed in args.seeds:
        result, seed_metrics = run_seed(
            users,
            rates,
            blocks,
            seed,
            args.alpha,
            args.radius_quantile,
            args.shrinkage,
            args.forest_trees,
        )
        results.append(result)
        metric_rows.extend(seed_metrics)
        print(
            f"seed={seed} band_cov={result['residual_band_coverage']:.3f} "
            f"direct_cov={result['direct_max_error_coverage']:.3f} "
            f"e2e_cov={result['e2e_max_error_coverage']:.3f}"
        )

    metrics = pd.DataFrame(metric_rows)
    summary = {
        "protocol": {
            "target": "realized category-rate vector for a held-out user",
            "guarantee": "split-conformal marginal coverage under exchangeable users",
            "does_not_guarantee": "pointwise coverage of latent conditional propensities",
            "alpha": args.alpha,
            "radius_quantile": args.radius_quantile,
            "shrinkage": args.shrinkage,
            "forest_trees": args.forest_trees,
            "seeds": args.seeds,
        },
        "data": metadata,
        "aggregate": aggregate(results, metrics),
        "runs": results,
        "runtime_seconds": time.perf_counter() - started,
    }
    metrics.to_csv(args.output / "metrics.csv", index=False)
    (args.output / "summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )
    print(json.dumps(summary["aggregate"], indent=2))


if __name__ == "__main__":
    main()
