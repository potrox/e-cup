"""Diagnostics for organizer-provided probabilistic LLM labels."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import ArrayLike
from scipy.stats import spearmanr


@dataclass(frozen=True)
class MacroSpearmanReport:
    """Unweighted mean of category-level Spearman correlations."""

    score: float
    per_category: dict[str, float]
    rows_per_category: dict[str, int]


def macro_spearman(
    y_soft: ArrayLike,
    y_score: ArrayLike,
    categories: ArrayLike,
) -> MacroSpearmanReport:
    """Measure rank agreement with soft labels independently per category."""

    soft = np.asarray(y_soft, dtype=np.float64)
    score = np.asarray(y_score, dtype=np.float64)
    category = np.asarray(categories)

    if soft.ndim != 1 or score.ndim != 1 or category.ndim != 1:
        raise ValueError("y_soft, y_score, and categories must be one-dimensional")
    if not (len(soft) == len(score) == len(category)):
        raise ValueError("y_soft, y_score, and categories must have equal lengths")
    if len(soft) == 0:
        raise ValueError("metric input must not be empty")
    if not np.isfinite(soft).all() or not np.isfinite(score).all():
        raise ValueError("metric inputs must contain only finite values")
    if ((soft < 0.0) | (soft > 1.0)).any():
        raise ValueError("y_soft must be in [0, 1]")

    category_values = category.tolist()
    if any(value is None for value in category_values):
        raise ValueError("categories must not contain null values")
    category_strings = np.asarray([str(value) for value in category_values], dtype=object)

    per_category: dict[str, float] = {}
    rows_per_category: dict[str, int] = {}
    for category_name in sorted(set(category_strings.tolist())):
        mask = category_strings == category_name
        if np.unique(soft[mask]).size < 2 or np.unique(score[mask]).size < 2:
            raise ValueError(f"Spearman correlation is undefined for category: {category_name}")
        correlation = float(spearmanr(soft[mask], score[mask]).statistic)
        if not np.isfinite(correlation):
            raise ValueError(f"Spearman correlation is non-finite for category: {category_name}")
        per_category[category_name] = correlation
        rows_per_category[category_name] = int(mask.sum())

    return MacroSpearmanReport(
        score=float(np.mean(list(per_category.values()))),
        per_category=per_category,
        rows_per_category=rows_per_category,
    )
