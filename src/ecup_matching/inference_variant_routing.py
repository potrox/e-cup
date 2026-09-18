"""Leakage-safe category routing between two frozen inference variants."""

from __future__ import annotations

from dataclasses import asdict
from typing import Any

import numpy as np
import polars as pl

from ecup_matching.metrics import macro_average_precision


def _score(
    target: np.ndarray,
    prediction: np.ndarray,
    categories: np.ndarray,
    *,
    expected_categories: int,
) -> dict[str, Any]:
    report = macro_average_precision(target, prediction, categories)
    if len(report.per_category) != expected_categories:
        raise ValueError(
            f"expected {expected_categories} categories, got {len(report.per_category)}"
        )
    return asdict(report)


def _category_evidence(
    *,
    target: np.ndarray,
    base: np.ndarray,
    candidate: np.ndarray,
    categories: np.ndarray,
    folds: np.ndarray,
    training_folds: tuple[int, ...],
) -> dict[str, dict[str, Any]]:
    evidence: dict[str, dict[str, Any]] = {}
    training_mask = np.isin(folds, training_folds)
    for category in sorted(str(value) for value in np.unique(categories)):
        category_mask = training_mask & (categories == category)
        aggregate_delta = float(
            macro_average_precision(
                target[category_mask], candidate[category_mask], categories[category_mask]
            ).score
            - macro_average_precision(
                target[category_mask], base[category_mask], categories[category_mask]
            ).score
        )
        per_fold: dict[str, float] = {}
        for fold in training_folds:
            fold_mask = category_mask & (folds == fold)
            per_fold[str(fold)] = float(
                macro_average_precision(
                    target[fold_mask], candidate[fold_mask], categories[fold_mask]
                ).score
                - macro_average_precision(
                    target[fold_mask], base[fold_mask], categories[fold_mask]
                ).score
            )
        evidence[category] = {
            "aggregate_delta": aggregate_delta,
            "per_fold": per_fold,
            "positive_folds": sum(delta > 0.0 for delta in per_fold.values()),
        }
    return evidence


def _select_categories(
    evidence: dict[str, dict[str, Any]],
    *,
    minimum_category_uplift: float,
    minimum_positive_folds: int,
) -> tuple[str, ...]:
    return tuple(
        category
        for category, values in evidence.items()
        if values["aggregate_delta"] >= minimum_category_uplift
        and values["positive_folds"] >= minimum_positive_folds
    )


def _stress_scores(
    frame: pl.DataFrame,
    *,
    target: np.ndarray,
    prediction: np.ndarray,
    categories: np.ndarray,
    expected_categories: int,
) -> dict[str, float] | None:
    required = {
        "surface_similarity",
        "identity_conflicts",
        "hard_negative_min_similarity",
        "hard_positive_max_similarity",
    }
    if not required.issubset(frame.columns):
        return None
    surface = frame["surface_similarity"].to_numpy()
    conflicts = frame["identity_conflicts"].to_numpy()
    hard_negative_min = frame["hard_negative_min_similarity"].to_numpy()
    hard_positive_max = frame["hard_positive_max_similarity"].to_numpy()
    positives = target == 1
    negatives = ~positives
    masks = {
        "hard_negative_challenge": positives
        | (negatives & (surface >= hard_negative_min)),
        "hard_positive_challenge": negatives
        | (positives & (surface <= hard_positive_max)),
        "identity_conflict_challenge": positives | (negatives & (conflicts > 0)),
    }
    return {
        name: _score(
            target[mask],
            prediction[mask],
            categories[mask],
            expected_categories=expected_categories,
        )["score"]
        for name, mask in masks.items()
    }


def evaluate_cross_fitted_category_route(
    frame: pl.DataFrame,
    *,
    base_column: str,
    candidate_column: str,
    minimum_category_uplift: float = 0.003,
    minimum_training_positive_folds: int = 3,
    minimum_deployment_positive_folds: int = 4,
    expected_categories: int = 20,
) -> tuple[dict[str, Any], np.ndarray, np.ndarray]:
    """Route a frozen candidate only where other folds prove category uplift."""

    required = {"target", "category", "fold", base_column, candidate_column}
    missing = sorted(required - set(frame.columns))
    if missing:
        raise ValueError(f"routing frame is missing columns: {missing}")
    if minimum_category_uplift <= 0.0:
        raise ValueError("minimum_category_uplift must be positive")

    target = frame["target"].to_numpy().astype(np.int8, copy=False)
    categories = frame["category"].to_numpy()
    folds = frame["fold"].to_numpy()
    base = frame[base_column].to_numpy()
    candidate = frame[candidate_column].to_numpy()
    if not np.isfinite(np.column_stack((base, candidate))).all():
        raise ValueError("routing scores must be finite")
    unique_folds = tuple(sorted(int(value) for value in np.unique(folds)))
    if len(unique_folds) != 5:
        raise ValueError(f"expected five folds, got {unique_folds}")
    if len(np.unique(categories)) != expected_categories:
        raise ValueError(f"expected {expected_categories} categories")

    routed_prediction = base.copy()
    routed_mask = np.zeros(len(base), dtype=bool)
    fold_records: list[dict[str, Any]] = []
    for validation_fold in unique_folds:
        training_folds = tuple(fold for fold in unique_folds if fold != validation_fold)
        evidence = _category_evidence(
            target=target,
            base=base,
            candidate=candidate,
            categories=categories,
            folds=folds,
            training_folds=training_folds,
        )
        selected = _select_categories(
            evidence,
            minimum_category_uplift=minimum_category_uplift,
            minimum_positive_folds=minimum_training_positive_folds,
        )
        validation_mask = folds == validation_fold
        selected_mask = validation_mask & np.isin(categories, selected)
        routed_prediction[selected_mask] = candidate[selected_mask]
        routed_mask[selected_mask] = True
        fold_records.append(
            {
                "validation_fold": validation_fold,
                "training_folds": list(training_folds),
                "selected_categories": list(selected),
                "routed_validation_rows": int(selected_mask.sum()),
                "category_evidence": evidence,
            }
        )
    if not np.array_equal(routed_prediction[~routed_mask], base[~routed_mask]):
        raise RuntimeError("non-routed predictions differ from the immutable base")

    final_evidence = _category_evidence(
        target=target,
        base=base,
        candidate=candidate,
        categories=categories,
        folds=folds,
        training_folds=unique_folds,
    )
    final_selected = _select_categories(
        final_evidence,
        minimum_category_uplift=minimum_category_uplift,
        minimum_positive_folds=minimum_deployment_positive_folds,
    )
    reference = _score(target, base, categories, expected_categories=expected_categories)
    routed = _score(
        target,
        routed_prediction,
        categories,
        expected_categories=expected_categories,
    )
    per_fold: dict[str, dict[str, float]] = {}
    for fold in unique_folds:
        mask = folds == fold
        reference_score = _score(
            target[mask], base[mask], categories[mask], expected_categories=expected_categories
        )["score"]
        routed_score = _score(
            target[mask],
            routed_prediction[mask],
            categories[mask],
            expected_categories=expected_categories,
        )["score"]
        per_fold[str(fold)] = {
            "reference": reference_score,
            "candidate": routed_score,
            "delta": routed_score - reference_score,
        }
    category_delta = {
        category: routed["per_category"][category] - reference["per_category"][category]
        for category in sorted(reference["per_category"])
    }
    reference_stress = _stress_scores(
        frame,
        target=target,
        prediction=base,
        categories=categories,
        expected_categories=expected_categories,
    )
    candidate_stress = _stress_scores(
        frame,
        target=target,
        prediction=routed_prediction,
        categories=categories,
        expected_categories=expected_categories,
    )
    stress_delta = (
        None
        if reference_stress is None or candidate_stress is None
        else {
            name: candidate_stress[name] - reference_stress[name]
            for name in reference_stress
        }
    )
    report = {
        "schema_version": 1,
        "protocol": (
            "Each held-out fold is routed using category evidence from the other four folds; "
            "non-routed rows preserve the immutable base score bitwise."
        ),
        "config": {
            "base_column": base_column,
            "candidate_column": candidate_column,
            "minimum_category_uplift": minimum_category_uplift,
            "minimum_training_positive_folds": minimum_training_positive_folds,
            "minimum_deployment_positive_folds": minimum_deployment_positive_folds,
        },
        "rows": len(base),
        "routed_oof_rows": int(routed_mask.sum()),
        "routed_oof_fraction": float(routed_mask.mean()),
        "cross_fitted_selections": fold_records,
        "final_deployment_categories": list(final_selected),
        "final_category_evidence": final_evidence,
        "reference": reference,
        "candidate": routed,
        "candidate_minus_reference": {
            "macro_pr_auc": routed["score"] - reference["score"],
            "per_fold": per_fold,
            "positive_folds": sum(value["delta"] > 0.0 for value in per_fold.values()),
            "per_category": category_delta,
            "nonnegative_categories": sum(value >= 0.0 for value in category_delta.values()),
            "worst_category_delta": min(category_delta.values()),
            "stress": stress_delta,
        },
        "invariants": {
            "non_routed_exact_base_parity": True,
            "public_used_for_selection": False,
        },
    }
    return report, routed_prediction, routed_mask
