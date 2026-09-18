"""Leakage-safe evaluation of a two-neural-model score ensemble."""

from __future__ import annotations

import json
from dataclasses import asdict
from pathlib import Path
from typing import Any

import numpy as np
import polars as pl

from ecup_matching.blending import (
    category_fold_percentile_ranks,
    cross_fitted_macro_blend,
    fit_final_macro_weights,
)
from ecup_matching.metrics import macro_average_precision

PAIR_KEYS = ("id1", "id2")
FOLDS = (0, 1, 2, 3, 4)


def _score(
    target: np.ndarray,
    score: np.ndarray,
    categories: np.ndarray,
) -> dict[str, Any]:
    report = macro_average_precision(target, score, categories)
    if len(report.per_category) != 20:
        raise ValueError("ensemble evaluation requires exactly 20 categories")
    return asdict(report)


def _fold_prediction_path(root: Path, fold: int) -> tuple[Path, dict[str, Any]]:
    manifest_path = root / f"fold_{fold}" / "training_manifest.json"
    if not manifest_path.is_file():
        raise FileNotFoundError(manifest_path)
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    path = manifest_path.parent / f"predictions_epoch_{int(manifest['best_epoch'])}.parquet"
    if not path.is_file():
        raise FileNotFoundError(path)
    return path, manifest


def load_ensemble_oof(base_root: Path, small_oof_path: Path) -> pl.DataFrame:
    """Load exact paired OOF logits and fold-local stress thresholds."""

    frames: list[pl.DataFrame] = []
    for fold in FOLDS:
        prediction_path, manifest = _fold_prediction_path(base_root, fold)
        thresholds = manifest["train"].get("hard_pair_thresholds")
        if not isinstance(thresholds, dict) or len(thresholds) != 20:
            raise ValueError(f"base fold {fold} has no 20-category stress thresholds")
        required = {
            *PAIR_KEYS,
            "target",
            "category",
            "score",
            "surface_similarity",
            "identity_conflicts",
        }
        schema = pl.read_parquet_schema(prediction_path)
        missing = sorted(required - set(schema))
        if missing:
            raise ValueError(f"base fold {fold} is missing columns: {missing}")
        threshold_frame = pl.DataFrame(
            {
                "category": list(thresholds),
                "hard_negative_min_similarity": [
                    thresholds[category]["hard_negative_min_similarity"]
                    for category in thresholds
                ],
                "hard_positive_max_similarity": [
                    thresholds[category]["hard_positive_max_similarity"] for category in thresholds
                ],
            }
        )
        frames.append(
            pl.read_parquet(prediction_path, columns=sorted(required))
            .rename({"score": "base_score"})
            .with_columns(pl.lit(fold, dtype=pl.Int8).alias("fold"))
            .join(threshold_frame, on="category", how="left", validate="m:1")
        )
    base = pl.concat(frames, how="vertical")
    if base.select(pl.struct(*PAIR_KEYS).n_unique()).item() != base.height:
        raise ValueError("base OOF contains duplicate pair keys")
    if base.select(pl.any_horizontal(pl.all().is_null()).any()).item():
        raise ValueError("base OOF contains nulls")

    if not small_oof_path.is_file():
        raise FileNotFoundError(small_oof_path)
    required_small = {*PAIR_KEYS, "target", "category", "fold", "predict"}
    small_schema = pl.read_parquet_schema(small_oof_path)
    missing_small = sorted(required_small - set(small_schema))
    if missing_small:
        raise ValueError(f"small OOF is missing columns: {missing_small}")
    small = pl.read_parquet(small_oof_path, columns=sorted(required_small)).rename(
        {
            "target": "small_target",
            "category": "small_category",
            "fold": "small_fold",
            "predict": "small_score",
        }
    )
    if small.select(pl.struct(*PAIR_KEYS).n_unique()).item() != small.height:
        raise ValueError("small OOF contains duplicate pair keys")
    if small.height != base.height:
        raise ValueError(f"base/small OOF row counts differ: {base.height} != {small.height}")

    joined = base.join(small, on=list(PAIR_KEYS), how="left", validate="1:1")
    if joined["small_score"].null_count():
        raise ValueError("small OOF does not cover every base OOF pair")
    mismatches = joined.filter(
        (pl.col("target") != pl.col("small_target"))
        | (pl.col("category") != pl.col("small_category"))
        | (pl.col("fold") != pl.col("small_fold"))
    ).height
    if mismatches:
        raise ValueError(f"base/small OOF contract mismatches: {mismatches}")
    scores = joined.select("base_score", "small_score").to_numpy()
    if not np.isfinite(scores).all():
        raise ValueError("ensemble OOF contains non-finite scores")
    return joined.drop("small_target", "small_category", "small_fold")


def _stress_report(
    frame: pl.DataFrame,
    target: np.ndarray,
    prediction: np.ndarray,
    categories: np.ndarray,
) -> dict[str, Any]:
    surface = frame["surface_similarity"].to_numpy()
    conflicts = frame["identity_conflicts"].to_numpy()
    hard_negative_min = frame["hard_negative_min_similarity"].to_numpy()
    hard_positive_max = frame["hard_positive_max_similarity"].to_numpy()
    positives = target == 1
    negatives = ~positives
    masks = {
        "hard_negative_challenge": positives | (negatives & (surface >= hard_negative_min)),
        "hard_positive_challenge": negatives | (positives & (surface <= hard_positive_max)),
        "identity_conflict_challenge": positives | (negatives & (conflicts > 0)),
    }
    return {
        name: {
            "rows": int(mask.sum()),
            "metric": _score(target[mask], prediction[mask], categories[mask]),
        }
        for name, mask in masks.items()
    }


def evaluate_feature_method(
    *,
    name: str,
    frame: pl.DataFrame,
    features: np.ndarray,
    step: float,
    minimum_uplift: float,
    public_correction: float,
    forecast_uncertainty: float,
    candidate_name: str,
) -> tuple[dict[str, Any], np.ndarray]:
    """Cross-fit one global blend method and evaluate its frozen gates."""

    target = frame["target"].to_numpy().astype(np.int8, copy=False)
    categories = frame["category"].to_numpy()
    folds = frame["fold"].to_numpy()
    base_score = frame["base_score"].to_numpy()
    prediction, records = cross_fitted_macro_blend(
        target,
        features,
        categories,
        folds,
        evaluation_folds=FOLDS,
        step=step,
    )
    final_weights = fit_final_macro_weights(
        target,
        features,
        categories,
        folds,
        evaluation_folds=FOLDS,
        step=step,
    )
    candidate = _score(target, prediction, categories)
    reference = _score(target, base_score, categories)
    per_fold: dict[str, dict[str, float]] = {}
    for fold in FOLDS:
        mask = folds == fold
        candidate_fold = _score(target[mask], prediction[mask], categories[mask])["score"]
        reference_fold = _score(target[mask], base_score[mask], categories[mask])["score"]
        per_fold[str(fold)] = {
            "candidate": candidate_fold,
            "reference": reference_fold,
            "delta": candidate_fold - reference_fold,
        }
    category_delta = {
        category: candidate["per_category"][category] - reference["per_category"][category]
        for category in sorted(candidate["per_category"])
    }
    candidate_stress = _stress_report(frame, target, prediction, categories)
    reference_stress = _stress_report(frame, target, base_score, categories)
    stress_delta = {
        cohort: candidate_stress[cohort]["metric"]["score"]
        - reference_stress[cohort]["metric"]["score"]
        for cohort in candidate_stress
    }
    candidate_weights = [record.weights[1] for record in records]
    overall_delta = candidate["score"] - reference["score"]
    gates = {
        "overall_uplift_at_least_minimum": overall_delta >= minimum_uplift,
        "positive_at_least_4_of_5_folds": sum(
            fold["delta"] > 0.0 for fold in per_fold.values()
        )
        >= 4,
        "nonnegative_at_least_16_of_20_categories": sum(
            value >= 0.0 for value in category_delta.values()
        )
        >= 16,
        "worst_category_delta_at_least_minus_0_005": min(category_delta.values()) >= -0.005,
        "all_stress_challenges_nonnegative": all(value >= 0.0 for value in stress_delta.values()),
        "cross_fitted_candidate_weight_range_at_most_0_25": (
            max(candidate_weights) - min(candidate_weights) <= 0.25
        ),
        "final_candidate_weight_positive": final_weights[1] > 0.0,
    }
    return (
        {
            "name": name,
            "protocol": (
                "For each held-out fold, one global weight is selected using only the other "
                "four folds. Final deployment weights are then fit once on all OOF rows."
            ),
            "feature_order": ["base", candidate_name],
            "step": step,
            "cross_fitted_weights": [asdict(record) for record in records],
            "final_deployment_weights": list(final_weights),
            "local_oof": candidate,
            "base_reference": reference,
            "candidate_minus_base": {
                "macro_pr_auc": overall_delta,
                "per_fold": per_fold,
                "positive_folds": sum(fold["delta"] > 0.0 for fold in per_fold.values()),
                "per_category": category_delta,
                "positive_categories": sum(value > 0.0 for value in category_delta.values()),
                "nonnegative_categories": sum(
                    value >= 0.0 for value in category_delta.values()
                ),
                "worst_category_delta": min(category_delta.values()),
                "stress": stress_delta,
            },
            "stress": {"candidate": candidate_stress, "reference": reference_stress},
            "public_forecast": {
                "point": candidate["score"] - public_correction,
                "lower": candidate["score"] - public_correction - forecast_uncertainty,
                "upper": candidate["score"] - public_correction + forecast_uncertainty,
                "empirical_correction": public_correction,
                "uncertainty": forecast_uncertainty,
                "official_metric": False,
                "used_for_promotion": False,
            },
            "gates": {**gates, "promote": all(gates.values())},
        },
        prediction,
    )


def evaluate_neural_ensemble(
    *,
    frame: pl.DataFrame,
    step: float = 0.05,
    minimum_uplift: float = 0.002,
    public_correction: float = 0.279,
    forecast_uncertainty: float = 0.02,
    candidate_name: str = "small_v3",
) -> tuple[dict[str, Any], dict[str, np.ndarray]]:
    """Evaluate the two predeclared global blend representations."""

    categories = frame["category"].to_numpy()
    folds = frame["fold"].to_numpy()
    base_score = frame["base_score"].to_numpy()
    small_score = frame["small_score"].to_numpy()
    feature_sets = {
        "raw_logit": np.column_stack([base_score, small_score]),
        "category_rank": np.column_stack(
            [
                category_fold_percentile_ranks(base_score, categories, folds),
                category_fold_percentile_ranks(small_score, categories, folds),
            ]
        ),
    }
    methods: dict[str, Any] = {}
    predictions: dict[str, np.ndarray] = {}
    for name, features in feature_sets.items():
        methods[name], predictions[name] = evaluate_feature_method(
            name=name,
            frame=frame,
            features=features,
            step=step,
            minimum_uplift=minimum_uplift,
            public_correction=public_correction,
            forecast_uncertainty=forecast_uncertainty,
            candidate_name=candidate_name,
        )
    promoted = [method for method in methods.values() if method["gates"]["promote"]]
    winner = (
        max(promoted, key=lambda method: method["local_oof"]["score"])["name"]
        if promoted
        else None
    )
    return (
        {
            "schema_version": 1,
            "protocol": (
                "Limited two-method, two-model cross-fitted OOF comparison. No neural "
                "retraining and no category-specific weights. Public forecast is reported "
                "separately and never replaces the local promotion metric."
            ),
            "rows": frame.height,
            "folds": list(FOLDS),
            "methods": methods,
            "selection": {
                "winner": winner,
                "promoted_methods": [method["name"] for method in promoted],
                "standalone_base_fallback_preserved": True,
            },
        },
        predictions,
    )
