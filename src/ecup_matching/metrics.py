"""Competition metric implementation and validation."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import ArrayLike
from sklearn.metrics import average_precision_score


@dataclass(frozen=True)
class MacroAveragePrecisionReport:
    """Macro PR-AUC score with category-level evidence."""

    score: float
    per_category: dict[str, float]
    rows_per_category: dict[str, int]
    positives_per_category: dict[str, int]


def macro_average_precision(
    y_true: ArrayLike,
    y_score: ArrayLike,
    categories: ArrayLike,
) -> MacroAveragePrecisionReport:
    """Compute the official unweighted mean of per-category average precision.

    The competition uses ``sklearn.metrics.average_precision_score`` independently
    for each category. This function validates inputs aggressively so a malformed
    local experiment cannot silently produce a misleading score.
    """

    true = np.asarray(y_true)
    score = np.asarray(y_score, dtype=np.float64)
    category = np.asarray(categories)

    if true.ndim != 1 or score.ndim != 1 or category.ndim != 1:
        raise ValueError("y_true, y_score, and categories must be one-dimensional")
    if not (len(true) == len(score) == len(category)):
        raise ValueError("y_true, y_score, and categories must have equal lengths")
    if len(true) == 0:
        raise ValueError("metric input must not be empty")
    if not np.isfinite(score).all():
        raise ValueError("y_score contains non-finite values")

    try:
        true_float = true.astype(np.float64)
    except (TypeError, ValueError) as exc:
        raise ValueError("y_true must contain numeric binary labels") from exc
    if not np.isfinite(true_float).all() or not np.isin(true_float, (0.0, 1.0)).all():
        raise ValueError("y_true must contain only finite binary labels 0 and 1")

    category_values = category.tolist()
    if any(value is None for value in category_values):
        raise ValueError("categories must not contain null values")

    category_strings = np.asarray([str(value) for value in category_values], dtype=object)
    unique_categories = sorted(set(category_strings.tolist()))
    per_category: dict[str, float] = {}
    rows_per_category: dict[str, int] = {}
    positives_per_category: dict[str, int] = {}

    for category_name in unique_categories:
        mask = category_strings == category_name
        category_true = true_float[mask]
        category_score = score[mask]
        per_category[category_name] = float(average_precision_score(category_true, category_score))
        rows_per_category[category_name] = int(mask.sum())
        positives_per_category[category_name] = int(category_true.sum())

    return MacroAveragePrecisionReport(
        score=float(np.mean(list(per_category.values()))),
        per_category=per_category,
        rows_per_category=rows_per_category,
        positives_per_category=positives_per_category,
    )
