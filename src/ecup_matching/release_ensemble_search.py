"""Leakage-safe no-training search over frozen release-model ensembles."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import asdict
from itertools import combinations
from pathlib import Path
from typing import Any

import numpy as np
import polars as pl

from ecup_matching.blending import (
    category_fold_percentile_ranks,
    cross_fitted_category_blend,
    cross_fitted_macro_blend,
    fit_final_category_weights,
    fit_final_macro_weights,
)
from ecup_matching.metrics import macro_average_precision
from ecup_matching.neural_ensemble import load_ensemble_oof

PAIR_KEYS = ("id1", "id2")
FOLDS = (0, 1, 2, 3, 4)
BASE_MODEL = "rumodernbert_base"


def _score(
    target: np.ndarray,
    prediction: np.ndarray,
    categories: np.ndarray,
) -> dict[str, Any]:
    report = macro_average_precision(target, prediction, categories)
    if len(report.per_category) != 20:
        raise ValueError("release ensemble evaluation requires exactly 20 categories")
    return asdict(report)


def load_release_ensemble_oof(
    *,
    base_cv_root: Path,
    model_oof_paths: Mapping[str, Path],
) -> pl.DataFrame:
    """Load immutable aligned OOF predictions and base-derived stress columns."""

    if BASE_MODEL in model_oof_paths:
        raise ValueError(f"{BASE_MODEL!r} is loaded from base_cv_root, not model_oof_paths")
    if len(model_oof_paths) < 2:
        raise ValueError("at least two non-base release models are required")

    first_name, first_path = next(iter(model_oof_paths.items()))
    frame = load_ensemble_oof(base_cv_root, first_path).rename(
        {"base_score": BASE_MODEL, "small_score": first_name}
    )
    for model_name, path in list(model_oof_paths.items())[1:]:
        required = {*PAIR_KEYS, "target", "category", "fold", "predict"}
        schema = pl.read_parquet_schema(path)
        missing = sorted(required - set(schema))
        if missing:
            raise ValueError(f"{model_name} OOF is missing columns: {missing}")
        candidate = pl.read_parquet(path, columns=sorted(required)).rename(
            {
                "target": f"{model_name}_target",
                "category": f"{model_name}_category",
                "fold": f"{model_name}_fold",
                "predict": model_name,
            }
        )
        if candidate.height != frame.height:
            raise ValueError(
                f"{model_name} OOF row count differs: {candidate.height} != {frame.height}"
            )
        if candidate.select(pl.struct(*PAIR_KEYS).n_unique()).item() != candidate.height:
            raise ValueError(f"{model_name} OOF contains duplicate pair keys")
        frame = frame.join(candidate, on=list(PAIR_KEYS), how="left", validate="1:1")
        if frame[model_name].null_count():
            raise ValueError(f"{model_name} OOF does not cover every base OOF pair")
        mismatches = frame.filter(
            (pl.col("target") != pl.col(f"{model_name}_target"))
            | (pl.col("category") != pl.col(f"{model_name}_category"))
            | (pl.col("fold") != pl.col(f"{model_name}_fold"))
        ).height
        if mismatches:
            raise ValueError(f"{model_name} OOF contract mismatches: {mismatches}")
        frame = frame.drop(
            f"{model_name}_target",
            f"{model_name}_category",
            f"{model_name}_fold",
        )

    model_names = [BASE_MODEL, *model_oof_paths]
    if not np.isfinite(frame.select(model_names).to_numpy()).all():
        raise ValueError("release ensemble OOF contains non-finite scores")
    return frame


def _stress_scores(
    frame: pl.DataFrame,
    target: np.ndarray,
    prediction: np.ndarray,
    categories: np.ndarray,
) -> dict[str, float]:
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
        name: _score(target[mask], prediction[mask], categories[mask])["score"]
        for name, mask in masks.items()
    }


def _weighted_correction(
    final_weights: tuple[float, ...] | dict[str, tuple[float, ...]],
    feature_order: tuple[str, ...],
    public_corrections: Mapping[str, float],
) -> float:
    corrections = np.asarray(
        [public_corrections[model_name] for model_name in feature_order], dtype=np.float64
    )
    if isinstance(final_weights, dict):
        values = [
            float(np.dot(np.asarray(weights), corrections))
            for weights in final_weights.values()
        ]
        return float(np.mean(values))
    return float(np.dot(np.asarray(final_weights), corrections))


def _evaluate_method(
    *,
    frame: pl.DataFrame,
    feature_order: tuple[str, ...],
    features: np.ndarray,
    representation: str,
    scope: str,
    step: float,
    minimum_uplift: float,
    public_corrections: Mapping[str, float],
    forecast_uncertainty: float,
) -> tuple[dict[str, Any], np.ndarray]:
    target = frame["target"].to_numpy().astype(np.int8, copy=False)
    categories = frame["category"].to_numpy()
    folds = frame["fold"].to_numpy()
    base_prediction = frame[BASE_MODEL].to_numpy()

    if scope == "global":
        prediction, records = cross_fitted_macro_blend(
            target,
            features,
            categories,
            folds,
            evaluation_folds=FOLDS,
            step=step,
        )
        final_weights: tuple[float, ...] | dict[str, tuple[float, ...]] = (
            fit_final_macro_weights(
                target,
                features,
                categories,
                folds,
                evaluation_folds=FOLDS,
                step=step,
            )
        )
    elif scope == "category":
        prediction, records = cross_fitted_category_blend(
            target,
            features,
            categories,
            folds,
            evaluation_folds=FOLDS,
            step=step,
        )
        final_weights = fit_final_category_weights(
            target,
            features,
            categories,
            folds,
            evaluation_folds=FOLDS,
            step=step,
        )
    else:
        raise ValueError(f"unsupported blend scope: {scope}")

    candidate = _score(target, prediction, categories)
    reference = _score(target, base_prediction, categories)
    per_fold: dict[str, dict[str, float]] = {}
    for fold in FOLDS:
        mask = folds == fold
        candidate_score = _score(target[mask], prediction[mask], categories[mask])["score"]
        reference_score = _score(
            target[mask], base_prediction[mask], categories[mask]
        )["score"]
        per_fold[str(fold)] = {
            "candidate": candidate_score,
            "reference": reference_score,
            "delta": candidate_score - reference_score,
        }
    category_delta = {
        category: candidate["per_category"][category] - reference["per_category"][category]
        for category in sorted(candidate["per_category"])
    }
    candidate_stress = _stress_scores(frame, target, prediction, categories)
    reference_stress = _stress_scores(frame, target, base_prediction, categories)
    stress_delta = {
        name: candidate_stress[name] - reference_stress[name] for name in candidate_stress
    }
    uplift = candidate["score"] - reference["score"]
    gates = {
        "macro_pr_auc_uplift_at_least_minimum": uplift >= minimum_uplift,
        "positive_at_least_4_of_5_folds": sum(
            item["delta"] > 0.0 for item in per_fold.values()
        )
        >= 4,
        "nonnegative_at_least_16_of_20_categories": sum(
            value >= 0.0 for value in category_delta.values()
        )
        >= 16,
        "worst_category_delta_at_least_minus_0_005": min(category_delta.values()) >= -0.005,
        "all_registered_stress_deltas_nonnegative": all(
            value >= 0.0 for value in stress_delta.values()
        ),
    }
    correction = _weighted_correction(final_weights, feature_order, public_corrections)
    method_name = f"{'+'.join(feature_order)}__{representation}__{scope}"
    return (
        {
            "name": method_name,
            "feature_order": list(feature_order),
            "representation": representation,
            "scope": scope,
            "weight_step": step,
            "cross_fitted_weights": [asdict(record) for record in records],
            "final_deployment_weights": final_weights,
            "local_oof": candidate,
            "candidate_minus_base": {
                "macro_pr_auc": uplift,
                "per_fold": per_fold,
                "positive_folds": sum(item["delta"] > 0.0 for item in per_fold.values()),
                "per_category": category_delta,
                "nonnegative_categories": sum(
                    value >= 0.0 for value in category_delta.values()
                ),
                "worst_category_delta": min(category_delta.values()),
                "stress": stress_delta,
            },
            "public_forecast": {
                "point": candidate["score"] - correction,
                "lower": candidate["score"] - correction - forecast_uncertainty,
                "upper": candidate["score"] - correction + forecast_uncertainty,
                "weighted_standalone_oof_to_public_correction": correction,
                "uncertainty": forecast_uncertainty,
                "official_metric": False,
                "used_for_selection": False,
            },
            "quality_gates": {**gates, "pass": all(gates.values())},
        },
        prediction,
    )


def evaluate_release_ensemble_search(
    *,
    frame: pl.DataFrame,
    model_names: tuple[str, ...],
    official_public_scores: Mapping[str, float],
    step: float = 0.05,
    minimum_uplift: float = 0.002,
    forecast_uncertainty: float = 0.015,
    workers: int = 1,
) -> tuple[dict[str, Any], dict[str, np.ndarray]]:
    """Evaluate all bounded two/three-model global and category-aware blends."""

    if model_names[0] != BASE_MODEL or len(model_names) != 3:
        raise ValueError("model_names must contain base first and exactly three release models")
    if workers < 1:
        raise ValueError("workers must be positive")
    missing_public = sorted(set(model_names) - set(official_public_scores))
    if missing_public:
        raise ValueError(f"official Public scores missing for: {missing_public}")

    target = frame["target"].to_numpy().astype(np.int8, copy=False)
    categories = frame["category"].to_numpy()
    folds = frame["fold"].to_numpy()
    standalone: dict[str, Any] = {}
    public_corrections: dict[str, float] = {}
    for model_name in model_names:
        metric = _score(target, frame[model_name].to_numpy(), categories)
        correction = metric["score"] - official_public_scores[model_name]
        public_corrections[model_name] = correction
        standalone[model_name] = {
            "local_oof": metric,
            "official_public": official_public_scores[model_name],
            "oof_to_public_correction": correction,
        }

    raw = {model_name: frame[model_name].to_numpy() for model_name in model_names}
    ranks = {
        model_name: category_fold_percentile_ranks(
            raw[model_name], categories, folds
        )
        for model_name in model_names
    }
    tasks: list[dict[str, Any]] = []
    for subset_size in (2, 3):
        for feature_order in combinations(model_names, subset_size):
            for representation, values in (("raw_logit", raw), ("category_rank", ranks)):
                features = np.column_stack([values[name] for name in feature_order])
                for scope in ("global", "category"):
                    tasks.append(
                        {
                            "frame": frame,
                            "feature_order": feature_order,
                            "features": features,
                            "representation": representation,
                            "scope": scope,
                            "step": step,
                            "minimum_uplift": minimum_uplift,
                            "public_corrections": public_corrections,
                            "forecast_uncertainty": forecast_uncertainty,
                        }
                    )

    completed: dict[str, tuple[dict[str, Any], np.ndarray]] = {}
    with ThreadPoolExecutor(max_workers=workers, thread_name_prefix="ensemble-search") as pool:
        futures = {pool.submit(_evaluate_method, **task): task for task in tasks}
        for future in as_completed(futures):
            method, prediction = future.result()
            completed[method["name"]] = (method, prediction)
            print(f"ensemble_search_completed={method['name']}", flush=True)

    methods: dict[str, Any] = {}
    predictions: dict[str, np.ndarray] = {}
    for task in tasks:
        method_name = (
            f"{'+'.join(task['feature_order'])}__"
            f"{task['representation']}__{task['scope']}"
        )
        method, prediction = completed[method_name]
        methods[method_name] = method
        predictions[method_name] = prediction

    passed = [method for method in methods.values() if method["quality_gates"]["pass"]]
    ranked = sorted(
        methods.values(), key=lambda method: method["local_oof"]["score"], reverse=True
    )
    return (
        {
            "schema_version": 1,
            "protocol": (
                "Bounded exhaustive no-training search over three immutable release models. "
                "Every weight is selected cross-fitted on other frozen item-disjoint folds."
            ),
            "rows": frame.height,
            "folds": list(FOLDS),
            "model_order": list(model_names),
            "standalone": standalone,
            "methods": methods,
            "selection": {
                "quality_passed": [method["name"] for method in passed],
                "best_quality_passed": (
                    max(passed, key=lambda method: method["local_oof"]["score"])["name"]
                    if passed
                    else None
                ),
                "ranking": [method["name"] for method in ranked],
                "runtime_and_packaging_pending": True,
            },
        },
        predictions,
    )


def _select_router_categories(
    *,
    target: np.ndarray,
    base_prediction: np.ndarray,
    blend_prediction: np.ndarray,
    categories: np.ndarray,
    mask: np.ndarray,
    count: int,
) -> tuple[str, ...]:
    deltas: list[tuple[float, str]] = []
    for category in sorted(np.unique(categories)):
        category_mask = mask & (categories == category)
        candidate = macro_average_precision(
            target[category_mask],
            blend_prediction[category_mask],
            categories[category_mask],
        ).score
        reference = macro_average_precision(
            target[category_mask],
            base_prediction[category_mask],
            categories[category_mask],
        ).score
        deltas.append((candidate - reference, str(category)))
    ranked = sorted(deltas, key=lambda item: (-item[0], item[1]))
    return tuple(category for _, category in ranked[:count])


def evaluate_sparse_category_routers(
    *,
    frame: pl.DataFrame,
    official_public_scores: Mapping[str, float],
    base_weight: float = 0.55,
    lamar_weight: float = 0.45,
    minimum_uplift: float = 0.002,
    forecast_uncertainty: float = 0.015,
    selected_counts: Sequence[int] | None = None,
) -> tuple[dict[str, Any], dict[str, np.ndarray]]:
    """Cross-fit every top-K category router over frozen base and LAMAR logits."""

    if set(official_public_scores) != {BASE_MODEL, "lamar_600m"}:
        raise ValueError("sparse router requires exact base and LAMAR Public scores")
    if base_weight < 0.0 or lamar_weight < 0.0 or not np.isclose(
        base_weight + lamar_weight, 1.0
    ):
        raise ValueError("router blend weights must be nonnegative and sum to one")

    target = frame["target"].to_numpy().astype(np.int8, copy=False)
    categories = frame["category"].to_numpy()
    folds = frame["fold"].to_numpy()
    base_prediction = frame[BASE_MODEL].to_numpy()
    lamar_prediction = frame["lamar_600m"].to_numpy()
    blended_prediction = base_weight * base_prediction + lamar_weight * lamar_prediction
    reference = _score(target, base_prediction, categories)
    reference_stress = _stress_scores(frame, target, base_prediction, categories)
    base_correction = reference["score"] - official_public_scores[BASE_MODEL]
    lamar_metric = _score(target, lamar_prediction, categories)
    lamar_correction = lamar_metric["score"] - official_public_scores["lamar_600m"]
    unique_categories = tuple(sorted(str(value) for value in np.unique(categories)))
    if len(unique_categories) != 20:
        raise ValueError("sparse router evaluation requires exactly 20 categories")
    if selected_counts is None:
        routed_category_counts = tuple(range(1, len(unique_categories) + 1))
    else:
        routed_category_counts = tuple(dict.fromkeys(int(value) for value in selected_counts))
        if not routed_category_counts:
            raise ValueError("selected_counts must contain at least one category count")
        if any(value < 1 or value > len(unique_categories) for value in routed_category_counts):
            raise ValueError("selected_counts values must be between 1 and 20")

    methods: dict[str, Any] = {}
    predictions: dict[str, np.ndarray] = {}
    for selected_count in routed_category_counts:
        prediction = base_prediction.copy()
        fold_selections: list[dict[str, Any]] = []
        for fold in FOLDS:
            train_mask = folds != fold
            validation_mask = folds == fold
            selected = _select_router_categories(
                target=target,
                base_prediction=base_prediction,
                blend_prediction=blended_prediction,
                categories=categories,
                mask=train_mask,
                count=selected_count,
            )
            routed = validation_mask & np.isin(categories, selected)
            prediction[routed] = blended_prediction[routed]
            fold_selections.append(
                {
                    "validation_fold": fold,
                    "selected_categories": list(selected),
                    "training_rows": int(np.count_nonzero(train_mask)),
                    "validation_rows": int(np.count_nonzero(validation_mask)),
                    "routed_validation_rows": int(np.count_nonzero(routed)),
                }
            )

        final_selected = _select_router_categories(
            target=target,
            base_prediction=base_prediction,
            blend_prediction=blended_prediction,
            categories=categories,
            mask=np.ones(len(target), dtype=bool),
            count=selected_count,
        )
        deployment_weights = {
            category: (
                [base_weight, lamar_weight]
                if category in final_selected
                else [1.0, 0.0]
            )
            for category in unique_categories
        }
        candidate = _score(target, prediction, categories)
        per_fold: dict[str, dict[str, float]] = {}
        for fold in FOLDS:
            mask = folds == fold
            candidate_score = _score(target[mask], prediction[mask], categories[mask])["score"]
            reference_score = _score(
                target[mask], base_prediction[mask], categories[mask]
            )["score"]
            per_fold[str(fold)] = {
                "candidate": candidate_score,
                "reference": reference_score,
                "delta": candidate_score - reference_score,
            }
        category_delta = {
            category: candidate["per_category"][category]
            - reference["per_category"][category]
            for category in unique_categories
        }
        candidate_stress = _stress_scores(frame, target, prediction, categories)
        stress_delta = {
            name: candidate_stress[name] - reference_stress[name]
            for name in candidate_stress
        }
        uplift = candidate["score"] - reference["score"]
        gates = {
            "macro_pr_auc_uplift_at_least_minimum": uplift >= minimum_uplift,
            "positive_at_least_4_of_5_folds": sum(
                value["delta"] > 0.0 for value in per_fold.values()
            )
            >= 4,
            "nonnegative_at_least_16_of_20_categories": sum(
                value >= 0.0 for value in category_delta.values()
            )
            >= 16,
            "worst_category_delta_at_least_minus_0_005": min(category_delta.values())
            >= -0.005,
            "all_registered_stress_deltas_nonnegative": all(
                value >= 0.0 for value in stress_delta.values()
            ),
        }
        routed_row_fraction = float(np.mean(np.isin(categories, final_selected)))
        correction = (
            (1.0 - selected_count / 20.0) * base_correction
            + (selected_count / 20.0)
            * (base_weight * base_correction + lamar_weight * lamar_correction)
        )
        method_name = (
            f"{BASE_MODEL}+lamar_600m__raw_logit__top_{selected_count:02d}_category_router"
        )
        methods[method_name] = {
            "name": method_name,
            "feature_order": [BASE_MODEL, "lamar_600m"],
            "representation": "raw_logit",
            "scope": "category",
            "router": "cross_fitted_top_k_category_uplift",
            "selected_category_count": selected_count,
            "cross_fitted_selections": fold_selections,
            "final_selected_categories": list(final_selected),
            "final_deployment_weights": deployment_weights,
            "routed_oof_row_fraction": routed_row_fraction,
            "local_oof": candidate,
            "candidate_minus_base": {
                "macro_pr_auc": uplift,
                "per_fold": per_fold,
                "positive_folds": sum(
                    value["delta"] > 0.0 for value in per_fold.values()
                ),
                "per_category": category_delta,
                "nonnegative_categories": sum(
                    value >= 0.0 for value in category_delta.values()
                ),
                "worst_category_delta": min(category_delta.values()),
                "stress": stress_delta,
            },
            "public_forecast": {
                "point": candidate["score"] - correction,
                "lower": candidate["score"] - correction - forecast_uncertainty,
                "upper": candidate["score"] - correction + forecast_uncertainty,
                "weighted_standalone_oof_to_public_correction": correction,
                "uncertainty": forecast_uncertainty,
                "official_metric": False,
                "used_for_selection": False,
            },
            "quality_gates": {**gates, "pass": all(gates.values())},
        }
        predictions[method_name] = prediction

    passed = [method for method in methods.values() if method["quality_gates"]["pass"]]
    ranking = sorted(
        methods.values(), key=lambda method: method["local_oof"]["score"], reverse=True
    )
    return (
        {
            "schema_version": 1,
            "protocol": (
                "No-training top-K category router. Categories are selected independently "
                "on the other frozen item-disjoint folds; unselected categories preserve "
                "exact base predictions."
            ),
            "rows": frame.height,
            "folds": list(FOLDS),
            "blend_weights": [base_weight, lamar_weight],
            "evaluated_category_counts": list(routed_category_counts),
            "standalone_oof_to_public_corrections": {
                BASE_MODEL: base_correction,
                "lamar_600m": lamar_correction,
            },
            "methods": methods,
            "selection": {
                "quality_passed": [method["name"] for method in passed],
                "ranking": [method["name"] for method in ranking],
                "runtime_and_packaging_pending": True,
            },
        },
        predictions,
    )
