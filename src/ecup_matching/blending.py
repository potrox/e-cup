"""Leakage-safe category rank normalization and constrained score blending."""

from __future__ import annotations

from dataclasses import dataclass
from itertools import product

import numpy as np
from scipy.stats import rankdata
from sklearn.metrics import average_precision_score


@dataclass(frozen=True, slots=True)
class BlendWeight:
    validation_fold: int
    category: str
    weights: tuple[float, ...]
    training_average_precision: float
    training_rows: int
    validation_rows: int


@dataclass(frozen=True, slots=True)
class MacroBlendWeight:
    validation_fold: int
    weights: tuple[float, ...]
    training_macro_average_precision: float
    training_rows: int
    validation_rows: int


def category_fold_percentile_ranks(
    scores: np.ndarray,
    categories: np.ndarray,
    folds: np.ndarray,
) -> np.ndarray:
    """Map scores to label-free percentile ranks independently by category and fold."""

    scores = np.asarray(scores, dtype=np.float64)
    categories = np.asarray(categories)
    folds = np.asarray(folds)
    if scores.ndim != 1 or categories.ndim != 1 or folds.ndim != 1:
        raise ValueError("scores, categories, and folds must be one-dimensional")
    if not (len(scores) == len(categories) == len(folds)) or not len(scores):
        raise ValueError("scores, categories, and folds must be non-empty and aligned")
    if not np.isfinite(scores).all():
        raise ValueError("scores contain non-finite values")

    result = np.empty(len(scores), dtype=np.float64)
    for category in np.unique(categories):
        for fold in np.unique(folds):
            mask = (categories == category) & (folds == fold)
            count = int(mask.sum())
            if count:
                result[mask] = rankdata(scores[mask], method="average") / (count + 1.0)
    return result


def simplex_candidates(feature_count: int, step: float) -> np.ndarray:
    """Generate deterministic non-negative weights that sum to one."""

    if feature_count < 2:
        raise ValueError("feature_count must be at least two")
    inverse = round(1.0 / step)
    if not 0.0 < step <= 1.0 or not np.isclose(inverse * step, 1.0):
        raise ValueError("step must be an exact divisor of one")

    candidates = [
        values
        for values in product(range(inverse + 1), repeat=feature_count)
        if sum(values) == inverse
    ]
    return np.asarray(candidates, dtype=np.float64) / inverse


def _best_weights(
    target: np.ndarray,
    rank_features: np.ndarray,
    candidates: np.ndarray,
) -> tuple[np.ndarray, float]:
    best_weights: np.ndarray | None = None
    best_score = float("-inf")
    for weights in candidates:
        score = float(average_precision_score(target, rank_features @ weights))
        if score > best_score + 1e-12:
            best_score = score
            best_weights = weights
        elif np.isclose(score, best_score, atol=1e-12) and best_weights is not None:
            if tuple(weights.tolist()) > tuple(best_weights.tolist()):
                best_weights = weights
    assert best_weights is not None
    return best_weights, best_score


def _macro_average_precision(
    target: np.ndarray,
    prediction: np.ndarray,
    categories: np.ndarray,
) -> float:
    scores = [
        average_precision_score(target[categories == category], prediction[categories == category])
        for category in np.unique(categories)
    ]
    return float(np.mean(scores))


def _best_macro_weights(
    target: np.ndarray,
    rank_features: np.ndarray,
    categories: np.ndarray,
    candidates: np.ndarray,
) -> tuple[np.ndarray, float]:
    best_weights: np.ndarray | None = None
    best_score = float("-inf")
    for weights in candidates:
        score = _macro_average_precision(target, rank_features @ weights, categories)
        if score > best_score + 1e-12:
            best_score = score
            best_weights = weights
        elif np.isclose(score, best_score, atol=1e-12) and best_weights is not None:
            if tuple(weights.tolist()) > tuple(best_weights.tolist()):
                best_weights = weights
    assert best_weights is not None
    return best_weights, best_score


def cross_fitted_macro_blend(
    target: np.ndarray,
    rank_features: np.ndarray,
    categories: np.ndarray,
    folds: np.ndarray,
    *,
    evaluation_folds: tuple[int, ...],
    step: float = 0.05,
) -> tuple[np.ndarray, list[MacroBlendWeight]]:
    """Fit one robust macro-AP weight vector on other folds for each held-out fold."""

    target = np.asarray(target, dtype=np.int8)
    rank_features = np.asarray(rank_features, dtype=np.float64)
    categories = np.asarray(categories)
    folds = np.asarray(folds)
    if rank_features.ndim != 2 or rank_features.shape[1] < 2:
        raise ValueError("rank_features must contain at least two score columns")
    if not (len(target) == len(rank_features) == len(categories) == len(folds)):
        raise ValueError("blend inputs must have equal row counts")
    if len(set(evaluation_folds)) < 2:
        raise ValueError("cross-fitted blend requires at least two evaluation folds")
    if not np.isin(target, (0, 1)).all() or not np.isfinite(rank_features).all():
        raise ValueError("invalid blend target or rank feature values")

    selected = np.isin(folds, evaluation_folds)
    predictions = np.full(len(target), np.nan, dtype=np.float64)
    candidates = simplex_candidates(rank_features.shape[1], step)
    records: list[MacroBlendWeight] = []
    for validation_fold in evaluation_folds:
        training_mask = selected & (folds != validation_fold)
        validation_mask = selected & (folds == validation_fold)
        weights, training_score = _best_macro_weights(
            target[training_mask],
            rank_features[training_mask],
            categories[training_mask],
            candidates,
        )
        predictions[validation_mask] = rank_features[validation_mask] @ weights
        records.append(
            MacroBlendWeight(
                validation_fold=int(validation_fold),
                weights=tuple(float(value) for value in weights),
                training_macro_average_precision=training_score,
                training_rows=int(training_mask.sum()),
                validation_rows=int(validation_mask.sum()),
            )
        )
    if not np.isfinite(predictions[selected]).all():
        raise RuntimeError("cross-fitted macro blend predictions are incomplete")
    return predictions, records


def fit_final_macro_weights(
    target: np.ndarray,
    rank_features: np.ndarray,
    categories: np.ndarray,
    folds: np.ndarray,
    *,
    evaluation_folds: tuple[int, ...],
    step: float = 0.05,
) -> tuple[float, ...]:
    """Fit one deployment weight vector after completing held-out evaluation."""

    target = np.asarray(target, dtype=np.int8)
    rank_features = np.asarray(rank_features, dtype=np.float64)
    categories = np.asarray(categories)
    folds = np.asarray(folds)
    selected = np.isin(folds, evaluation_folds)
    candidates = simplex_candidates(rank_features.shape[1], step)
    weights, _ = _best_macro_weights(
        target[selected], rank_features[selected], categories[selected], candidates
    )
    return tuple(float(value) for value in weights)


def cross_fitted_category_blend(
    target: np.ndarray,
    rank_features: np.ndarray,
    categories: np.ndarray,
    folds: np.ndarray,
    *,
    evaluation_folds: tuple[int, ...],
    step: float = 0.05,
) -> tuple[np.ndarray, list[BlendWeight]]:
    """Fit category weights on other evaluation folds and score each held-out fold."""

    target = np.asarray(target, dtype=np.int8)
    rank_features = np.asarray(rank_features, dtype=np.float64)
    categories = np.asarray(categories)
    folds = np.asarray(folds)
    if rank_features.ndim != 2 or rank_features.shape[1] < 2:
        raise ValueError("rank_features must contain at least two score columns")
    if not (len(target) == len(rank_features) == len(categories) == len(folds)):
        raise ValueError("blend inputs must have equal row counts")
    if len(set(evaluation_folds)) < 2:
        raise ValueError("cross-fitted blend requires at least two evaluation folds")
    if not np.isin(target, (0, 1)).all() or not np.isfinite(rank_features).all():
        raise ValueError("invalid blend target or rank feature values")

    selected = np.isin(folds, evaluation_folds)
    predictions = np.full(len(target), np.nan, dtype=np.float64)
    candidates = simplex_candidates(rank_features.shape[1], step)
    records: list[BlendWeight] = []
    for validation_fold in evaluation_folds:
        for category_value in sorted(np.unique(categories[selected]).tolist()):
            category = str(category_value)
            training_mask = selected & (folds != validation_fold) & (categories == category_value)
            validation_mask = selected & (folds == validation_fold) & (categories == category_value)
            if not training_mask.any() or not validation_mask.any():
                raise ValueError(
                    f"missing training or validation rows for {category!r}, fold={validation_fold}"
                )
            weights, training_score = _best_weights(
                target[training_mask], rank_features[training_mask], candidates
            )
            predictions[validation_mask] = rank_features[validation_mask] @ weights
            records.append(
                BlendWeight(
                    validation_fold=int(validation_fold),
                    category=category,
                    weights=tuple(float(value) for value in weights),
                    training_average_precision=training_score,
                    training_rows=int(training_mask.sum()),
                    validation_rows=int(validation_mask.sum()),
                )
            )
    if not np.isfinite(predictions[selected]).all():
        raise RuntimeError("cross-fitted blend predictions are incomplete")
    return predictions, records


def fit_final_category_weights(
    target: np.ndarray,
    rank_features: np.ndarray,
    categories: np.ndarray,
    folds: np.ndarray,
    *,
    evaluation_folds: tuple[int, ...],
    step: float = 0.05,
) -> dict[str, tuple[float, ...]]:
    """Fit deployment weights on all selected folds after honest evaluation."""

    target = np.asarray(target, dtype=np.int8)
    rank_features = np.asarray(rank_features, dtype=np.float64)
    categories = np.asarray(categories)
    folds = np.asarray(folds)
    selected = np.isin(folds, evaluation_folds)
    candidates = simplex_candidates(rank_features.shape[1], step)
    result: dict[str, tuple[float, ...]] = {}
    for category_value in sorted(np.unique(categories[selected]).tolist()):
        mask = selected & (categories == category_value)
        weights, _ = _best_weights(target[mask], rank_features[mask], candidates)
        result[str(category_value)] = tuple(float(value) for value in weights)
    return result
