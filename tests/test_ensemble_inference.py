from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from ecup_matching.ensemble_inference import (
    EnsembleModel,
    EnsembleResidual,
    active_model_length_masks,
    active_model_mask,
    apply_category_rank_residual,
    blend_scores,
)
from ecup_matching.neural_inference import NeuralBundle


def test_blend_scores_applies_global_raw_logits() -> None:
    result = blend_scores(
        score_columns=[np.array([0.0, 2.0]), np.array([2.0, 0.0])],
        categories=np.array(["A", "A"]),
        representation="raw_logit",
        scope="global",
        weights=(0.75, 0.25),
    )

    np.testing.assert_allclose(result, [0.5, 1.5])


def test_blend_scores_applies_category_weights() -> None:
    result = blend_scores(
        score_columns=[np.array([0.0, 2.0]), np.array([2.0, 0.0])],
        categories=np.array(["A", "B"]),
        representation="raw_logit",
        scope="category",
        weights={"A": (1.0, 0.0), "B": (0.0, 1.0)},
    )

    np.testing.assert_allclose(result, [0.0, 0.0])


def test_blend_scores_fails_closed_on_unknown_category() -> None:
    with pytest.raises(ValueError, match="missing ensemble weights"):
        blend_scores(
            score_columns=[np.array([0.0]), np.array([1.0])],
            categories=np.array(["unknown"]),
            representation="raw_logit",
            scope="category",
            weights={"known": (0.5, 0.5)},
        )


def test_active_model_mask_skips_zero_weight_routed_rows() -> None:
    categories = np.array(["A", "A", "B", "B"])

    result = active_model_mask(
        categories=categories,
        model_index=1,
        scope="category",
        weights={"A": (0.55, 0.45), "B": (1.0, 0.0)},
    )

    np.testing.assert_array_equal(result, [True, True, False, False])


def test_active_model_length_masks_partition_only_active_rows() -> None:
    categories = np.array(["A", "A", "B", "C"])
    model = EnsembleModel(
        name="base",
        bundle=NeuralBundle(
            model_path=Path("model"),
            max_length=256,
            max_attribute_characters=2048,
            batch_size=32,
            serialization_mode="item_v1",
            serialization_version="item_v1",
            serialization_chunk_size=128,
            bidirectional=False,
        ),
        max_length_by_category={"A": 384},
    )

    result = active_model_length_masks(
        categories=categories,
        model_index=0,
        model=model,
        scope="category",
        weights={
            "A": (0.55, 0.45),
            "B": (1.0, 0.0),
            "C": (0.0, 1.0),
        },
    )

    assert sorted(result) == [256, 384]
    np.testing.assert_array_equal(result[256], [False, False, True, False])
    np.testing.assert_array_equal(result[384], [True, True, False, False])


def test_category_rank_residual_changes_only_explicit_categories() -> None:
    reference = np.array([0.1, 0.9, 10.0, 20.0])
    residual_score = np.array([0.8, 0.2, -3.0, 4.0])
    categories = np.array(["weak", "weak", "strong", "strong"])

    result = apply_category_rank_residual(
        reference=reference,
        residual_score=residual_score,
        categories=categories,
        residual=EnsembleResidual(
            model_name="teacher",
            categories=("weak",),
            reference_weight=0.75,
            model_weight=0.25,
        ),
    )

    np.testing.assert_allclose(result[:2], [5.0 / 12.0, 7.0 / 12.0])
    np.testing.assert_array_equal(result[2:], reference[2:])


def test_category_rank_residual_preserves_reference_when_route_is_absent() -> None:
    reference = np.array([1.0, 2.0])
    result = apply_category_rank_residual(
        reference=reference,
        residual_score=np.zeros(2),
        categories=np.array(["strong", "strong"]),
        residual=EnsembleResidual(
            model_name="teacher",
            categories=("weak",),
            reference_weight=0.75,
            model_weight=0.25,
        ),
    )

    np.testing.assert_array_equal(result, reference)
