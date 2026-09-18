from __future__ import annotations

from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import pytest
import torch

from ecup_matching.neural_training import (
    HardPairThreshold,
    bce_loss,
    category_row_counts,
    hard_pair_thresholds,
    human_stress_suite,
    inverse_category_weights,
    iter_mixed_pair_batches,
    iter_pair_batches,
    parquet_rows,
    replay_rows_for_primary,
    training_steps,
    weighted_bce_loss,
)


def test_training_steps_include_partial_batches_and_accumulation() -> None:
    assert training_steps(rows=101, batch_size=32, gradient_accumulation=2, epochs=3) == 6


def test_soft_bce_accepts_probabilistic_targets() -> None:
    logits = torch.tensor([-1.0, 0.0, 1.0])
    labels = torch.tensor([0.0, 0.5, 1.0])

    loss = bce_loss(logits, labels)

    assert np.isfinite(float(loss))
    assert float(loss) > 0.0


def test_soft_bce_rejects_invalid_targets() -> None:
    with pytest.raises(ValueError, match=r"\[0, 1\]"):
        bce_loss(torch.tensor([0.0]), torch.tensor([1.1]))


def test_category_mean_bce_gives_categories_equal_aggregate_weight() -> None:
    logits = torch.tensor([-3.0, -2.0, -1.0, -4.0])
    labels = torch.ones(4)
    categories = ["a", "a", "a", "b"]
    weights = inverse_category_weights({"a": 3, "b": 1})

    loss = weighted_bce_loss(
        logits,
        labels,
        categories=categories,
        category_weights=weights,
    )
    row_losses = torch.nn.functional.binary_cross_entropy_with_logits(
        logits, labels, reduction="none"
    )
    expected = (row_losses[:3].mean() + row_losses[3]) * 0.5

    assert torch.allclose(loss, expected)


def test_confidence_weighting_drops_only_ambiguous_soft_label() -> None:
    logits = torch.tensor([-1.0, 4.0, 1.0])
    labels = torch.tensor([0.0, 0.5, 1.0])

    loss = weighted_bce_loss(logits, labels, confidence_gamma=1.0)
    expected = bce_loss(logits[[0, 2]], labels[[0, 2]])

    assert torch.allclose(loss, expected)


def test_pair_v2_stream_uses_single_canonical_text_and_preserves_diagnostics(tmp_path) -> None:
    path = tmp_path / "pairs.parquet"
    pq.write_table(
        pa.table(
            {
                "id1": [1, 2],
                "id2": [3, 4],
                "target": [1.0, 0.0],
                "category": ["a", "b"],
                "right_category": ["a", "b"],
                "left_name": ["ACME X100", "ACME A100"],
                "left_attributes": ['{"модель":"X100"}', '{"модель":"A100"}'],
                "right_name": ["X100 ACME", "ACME A200"],
                "right_attributes": ['{"модель":"X100"}', '{"модель":"A200"}'],
                "surface_similarity": [0.9, 0.8],
                "fold": [0, 1],
            }
        ),
        path,
    )

    batches = iter_pair_batches(
        path,
        batch_size=2,
        read_batch_size=2,
        max_attribute_characters=512,
        seed=2026,
        epoch=0,
        shuffle=False,
        pair_swap_probability=0.0,
        serialization_mode="pair_v2",
    )
    batch = next(batches)

    assert batch.right_texts == ["", ""]
    assert np.allclose(batch.surface_similarity, [0.9, 0.8])
    assert batch.identity_matches.tolist() == [2, 0]
    assert batch.identity_conflicts.tolist() == [0, 2]


def test_stream_carries_partial_read_chunks_into_exact_training_batches(tmp_path) -> None:
    path = tmp_path / "partial-chunks.parquet"
    rows = 23
    pq.write_table(
        pa.table(
            {
                "id1": list(range(rows)),
                "id2": list(range(100, 100 + rows)),
                "target": [float(index % 2) for index in range(rows)],
                "category": ["a"] * rows,
                "left_name": [f"left {index}" for index in range(rows)],
                "left_attributes": ["{}"] * rows,
                "right_name": [f"right {index}" for index in range(rows)],
                "right_attributes": ["{}"] * rows,
            }
        ),
        path,
        row_group_size=5,
    )

    batches = list(
        iter_pair_batches(
            path,
            batch_size=4,
            read_batch_size=5,
            max_attribute_characters=64,
            seed=2026,
            epoch=0,
            shuffle=False,
            pair_swap_probability=0.0,
            serialization_mode="item_v1",
        )
    )

    assert [len(batch) for batch in batches] == [4, 4, 4, 4, 4, 3]
    assert len(batches) == training_steps(rows, 4, 1, 1)
    assert np.concatenate([batch.id1 for batch in batches]).tolist() == list(range(rows))

    shuffled = list(
        iter_pair_batches(
            path,
            batch_size=4,
            read_batch_size=5,
            max_attribute_characters=64,
            seed=2026,
            epoch=0,
            shuffle=True,
            pair_swap_probability=0.0,
            serialization_mode="item_v1",
        )
    )
    assert [len(batch) for batch in shuffled] == [4, 4, 4, 4, 4, 3]
    assert sorted(np.concatenate([batch.id1 for batch in shuffled]).tolist()) == list(range(rows))


def test_mixed_replay_preserves_all_primary_rows_and_requested_fraction(tmp_path) -> None:
    primary_path = tmp_path / "primary.parquet"
    replay_path = tmp_path / "replay.parquet"

    def write_pairs(path: Path, ids: list[int]) -> None:
        pq.write_table(
            pa.table(
                {
                    "id1": ids,
                    "id2": [value + 1000 for value in ids],
                    "target": [float(value % 2) for value in ids],
                    "category": ["a"] * len(ids),
                    "left_name": [f"left {value}" for value in ids],
                    "left_attributes": ["{}"] * len(ids),
                    "right_name": [f"right {value}" for value in ids],
                    "right_attributes": ["{}"] * len(ids),
                }
            ),
            path,
            row_group_size=4,
        )

    write_pairs(primary_path, list(range(8)))
    write_pairs(replay_path, list(range(100, 108)))
    primary = iter_pair_batches(
        primary_path,
        batch_size=4,
        read_batch_size=4,
        max_attribute_characters=64,
        seed=2026,
        epoch=0,
        shuffle=False,
        pair_swap_probability=0.0,
    )
    replay = iter_pair_batches(
        replay_path,
        batch_size=3,
        read_batch_size=4,
        max_attribute_characters=64,
        seed=2027,
        epoch=0,
        shuffle=False,
        pair_swap_probability=0.0,
    )

    mixed = list(iter_mixed_pair_batches(primary, replay, replay_fraction=0.2))

    assert replay_rows_for_primary(4, 0.2) == 1
    assert [(primary_rows, replay_rows) for _, primary_rows, replay_rows in mixed] == [
        (4, 1),
        (4, 1),
    ]
    assert [len(batch) for batch, _, _ in mixed] == [5, 5]
    primary_ids = np.concatenate([batch.id1[:primary_rows] for batch, primary_rows, _ in mixed])
    assert primary_ids.tolist() == list(range(8))


def test_mixed_replay_fails_closed_when_replay_data_is_exhausted(tmp_path) -> None:
    path = tmp_path / "pairs.parquet"
    pq.write_table(
        pa.table(
            {
                "id1": [1, 2],
                "id2": [3, 4],
                "target": [0.0, 1.0],
                "category": ["a", "a"],
                "left_name": ["left 1", "left 2"],
                "left_attributes": ["{}", "{}"],
                "right_name": ["right 1", "right 2"],
                "right_attributes": ["{}", "{}"],
            }
        ),
        path,
    )
    primary = iter_pair_batches(
        path,
        batch_size=1,
        read_batch_size=2,
        max_attribute_characters=64,
        seed=1,
        epoch=0,
        shuffle=False,
        pair_swap_probability=0.0,
    )

    with pytest.raises(ValueError, match="replay dataset exhausted"):
        list(iter_mixed_pair_batches(primary, iter(()), replay_fraction=0.2))


def test_item_v1_hard_replay_computes_identity_conflicts(tmp_path) -> None:
    path = tmp_path / "pairs.parquet"
    pq.write_table(
        pa.table(
            {
                "id1": [1, 3],
                "id2": [2, 4],
                "target": [0.0, 1.0],
                "category": ["a", "a"],
                "left_name": ["phone x100", "same"],
                "left_attributes": ['{"model":"x100"}', "{}"],
                "right_name": ["phone x200", "same"],
                "right_attributes": ['{"model":"x200"}', "{}"],
                "surface_similarity": [0.9, 0.1],
            }
        ),
        path,
    )
    thresholds = {
        "a": HardPairThreshold(
            hard_negative_min_similarity=0.8,
            hard_positive_max_similarity=0.2,
            negative_candidates=1,
            positive_candidates=1,
        )
    }

    batch = next(
        iter_pair_batches(
            path,
            batch_size=2,
            read_batch_size=2,
            max_attribute_characters=256,
            seed=2026,
            epoch=0,
            shuffle=True,
            pair_swap_probability=0.5,
            serialization_mode="item_v1",
            hard_replay_fraction=0.5,
            hard_pair_thresholds=thresholds,
        )
    )

    assert batch.identity_conflicts.max() > 0


def test_pair_v2_rejects_redundant_swap_augmentation(tmp_path) -> None:
    path = tmp_path / "empty.parquet"
    pq.write_table(
        pa.table(
            {
                "id1": pa.array([], type=pa.int64()),
                "id2": pa.array([], type=pa.int64()),
                "target": pa.array([], type=pa.float32()),
                "category": pa.array([], type=pa.string()),
                "left_name": pa.array([], type=pa.string()),
                "left_attributes": pa.array([], type=pa.string()),
                "right_name": pa.array([], type=pa.string()),
                "right_attributes": pa.array([], type=pa.string()),
            }
        ),
        path,
    )
    batches = iter_pair_batches(
        path,
        batch_size=1,
        read_batch_size=1,
        max_attribute_characters=32,
        seed=1,
        epoch=0,
        shuffle=False,
        pair_swap_probability=0.5,
        serialization_mode="pair_v2",
    )

    with pytest.raises(ValueError, match="canonical"):
        next(batches)


def test_category_counts_respect_outer_fold_filter(tmp_path) -> None:
    path = tmp_path / "counts.parquet"
    pq.write_table(
        pa.table(
            {
                "category": ["a", "a", "b"],
                "fold": [0, 1, 1],
                "inner_fold": [0, 0, 1],
            }
        ),
        path,
    )

    assert category_row_counts(path, exclude_fold=0) == {"a": 1, "b": 1}
    assert category_row_counts(path, exclude_fold=0, include_inner_fold=1) == {"b": 1}
    assert parquet_rows(path, exclude_fold=0, exclude_inner_fold=0) == 1


def test_hard_pair_thresholds_are_fitted_only_on_selected_training_folds(tmp_path) -> None:
    path = tmp_path / "hard-pairs.parquet"
    pq.write_table(
        pa.table(
            {
                "category": ["a"] * 6,
                "target": [0.0, 0.0, 1.0, 1.0, 0.0, 1.0],
                "surface_similarity": [0.2, 0.8, 0.1, 0.7, 1.0, 0.0],
                "fold": [0, 0, 0, 0, 1, 1],
            }
        ),
        path,
    )

    thresholds = hard_pair_thresholds(
        path,
        exclude_fold=1,
        negative_target_max=0.2,
        positive_target_min=0.8,
        negative_similarity_quantile=0.75,
        positive_similarity_quantile=0.25,
    )

    assert thresholds["a"].negative_candidates == 2
    assert thresholds["a"].positive_candidates == 2
    assert thresholds["a"].hard_negative_min_similarity == pytest.approx(0.65)
    assert thresholds["a"].hard_positive_max_similarity == pytest.approx(0.25)


def test_hard_replay_keeps_training_budget_fixed(tmp_path) -> None:
    path = tmp_path / "replay.parquet"
    rows = 16
    pq.write_table(
        pa.table(
            {
                "id1": list(range(rows)),
                "id2": list(range(100, 100 + rows)),
                "target": [0.0, 1.0] * (rows // 2),
                "category": ["a"] * rows,
                "left_name": [f"left {index}" for index in range(rows)],
                "left_attributes": ["{}"] * rows,
                "right_name": [f"right {index}" for index in range(rows)],
                "right_attributes": ["{}"] * rows,
                "surface_similarity": [0.9, 0.1] * (rows // 2),
            }
        ),
        path,
    )
    threshold = HardPairThreshold(
        hard_negative_min_similarity=0.8,
        hard_positive_max_similarity=0.2,
        negative_candidates=8,
        positive_candidates=8,
    )
    batches = iter_pair_batches(
        path,
        batch_size=4,
        read_batch_size=16,
        max_attribute_characters=64,
        seed=2026,
        epoch=0,
        shuffle=True,
        pair_swap_probability=0.0,
        serialization_mode="pair_v2",
        hard_replay_fraction=0.25,
        hard_pair_thresholds={"a": threshold},
    )

    emitted = sum(len(batch) for batch in batches)

    assert emitted == rows


def test_human_stress_suite_builds_binary_challenges_from_train_thresholds() -> None:
    threshold = HardPairThreshold(
        hard_negative_min_similarity=0.8,
        hard_positive_max_similarity=0.2,
        negative_candidates=2,
        positive_candidates=2,
    )

    report = human_stress_suite(
        targets=np.asarray([1.0, 1.0, 0.0, 0.0]),
        scores=np.asarray([0.9, 0.2, 0.8, 0.1]),
        categories=np.asarray(["a", "a", "a", "a"], dtype=object),
        surface_similarity=np.asarray([0.1, 0.9, 0.9, 0.1]),
        identity_conflicts=np.asarray([0, 0, 1, 0]),
        thresholds={"a": threshold},
    )

    assert report["hard_negative_rows"] == 1
    assert report["hard_positive_rows"] == 1
    assert report["conflict_negative_rows"] == 1
    assert report["hard_negative_challenge"]["eligible_categories"] == 1
    assert report["hard_positive_challenge"]["eligible_categories"] == 1
    assert report["identity_conflict_challenge"]["eligible_categories"] == 1
