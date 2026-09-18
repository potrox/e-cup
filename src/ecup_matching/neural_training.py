"""Streaming training and evaluation primitives for neural product matching."""

from __future__ import annotations

import math
import random
import time
from collections.abc import Iterator, Mapping
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Literal

import duckdb
import numpy as np
import pyarrow as pa
import pyarrow.dataset as arrow_dataset
import pyarrow.parquet as pq
import torch
import torch.nn.functional as functional
from sklearn.metrics import average_precision_score
from transformers import PreTrainedModel, PreTrainedTokenizerBase

from ecup_matching.llm_diagnostics import macro_spearman
from ecup_matching.metrics import macro_average_precision
from ecup_matching.serialization import serialize_items, serialize_pairs

SerializationMode = Literal["item_v1", "pair_v2"]
LossMode = Literal["row_mean", "category_mean"]

PAIR_COLUMNS = (
    "id1",
    "id2",
    "target",
    "category",
    "left_name",
    "left_attributes",
    "right_name",
    "right_attributes",
)


@dataclass(slots=True)
class PairTextBatch:
    id1: np.ndarray
    id2: np.ndarray
    target: np.ndarray
    categories: list[str]
    left_texts: list[str]
    right_texts: list[str]
    surface_similarity: np.ndarray
    identity_matches: np.ndarray
    identity_conflicts: np.ndarray
    missing_identity_fields: np.ndarray

    def __len__(self) -> int:
        return len(self.target)


@dataclass(frozen=True, slots=True)
class EvaluationResult:
    selection_metric: float
    metrics: dict[str, Any]
    prediction_path: Path | None
    rows: int
    seconds: float
    rows_per_second: float
    peak_gpu_memory_bytes: int


@dataclass(frozen=True, slots=True)
class HardPairThreshold:
    """Training-only surface thresholds for deterministic hard-pair curriculum."""

    hard_negative_min_similarity: float
    hard_positive_max_similarity: float
    negative_candidates: int
    positive_candidates: int


def _validate_split_filters(
    *,
    include_fold: int | None,
    exclude_fold: int | None,
    include_inner_fold: int | None,
    exclude_inner_fold: int | None,
) -> None:
    if include_fold is not None and exclude_fold is not None:
        raise ValueError("include_fold and exclude_fold are mutually exclusive")
    if include_inner_fold is not None and exclude_inner_fold is not None:
        raise ValueError("include_inner_fold and exclude_inner_fold are mutually exclusive")


def parquet_rows(
    path: Path,
    *,
    include_fold: int | None = None,
    exclude_fold: int | None = None,
    include_inner_fold: int | None = None,
    exclude_inner_fold: int | None = None,
) -> int:
    if not path.is_file():
        raise FileNotFoundError(path)
    _validate_split_filters(
        include_fold=include_fold,
        exclude_fold=exclude_fold,
        include_inner_fold=include_inner_fold,
        exclude_inner_fold=exclude_inner_fold,
    )
    if all(
        value is None
        for value in (include_fold, exclude_fold, include_inner_fold, exclude_inner_fold)
    ):
        return pq.ParquetFile(path).metadata.num_rows
    dataset = arrow_dataset.dataset(path, format="parquet")
    if (
        include_fold is not None or exclude_fold is not None
    ) and "fold" not in dataset.schema.names:
        raise ValueError(f"fold filtering requested for dataset without fold column: {path}")
    if (
        include_inner_fold is not None or exclude_inner_fold is not None
    ) and "inner_fold" not in dataset.schema.names:
        raise ValueError(f"inner fold filtering requested for dataset without inner_fold: {path}")
    expression = None
    if include_fold is not None:
        expression = arrow_dataset.field("fold") == include_fold
    elif exclude_fold is not None:
        expression = arrow_dataset.field("fold") != exclude_fold
    if include_inner_fold is not None:
        inner_expression = arrow_dataset.field("inner_fold") == include_inner_fold
        expression = inner_expression if expression is None else expression & inner_expression
    elif exclude_inner_fold is not None:
        inner_expression = arrow_dataset.field("inner_fold") != exclude_inner_fold
        expression = inner_expression if expression is None else expression & inner_expression
    if expression is None:
        raise RuntimeError("split filter expression was not constructed")
    return dataset.count_rows(filter=expression)


def _python_strings(batch: pa.RecordBatch, name: str) -> list[str]:
    return ["" if value is None else str(value) for value in batch.column(name).to_pylist()]


def _serialize_record_batch(
    batch: pa.RecordBatch,
    *,
    max_attribute_characters: int,
    serialization_mode: SerializationMode,
    compute_pair_diagnostics: bool,
) -> PairTextBatch:
    categories = _python_strings(batch, "category")
    left_names = _python_strings(batch, "left_name")
    left_attributes = _python_strings(batch, "left_attributes")
    right_names = _python_strings(batch, "right_name")
    right_attributes = _python_strings(batch, "right_attributes")
    if serialization_mode == "item_v1":
        left_texts = serialize_items(
            categories,
            left_names,
            left_attributes,
            max_attribute_characters=max_attribute_characters,
        )
        right_texts = serialize_items(
            categories,
            right_names,
            right_attributes,
            max_attribute_characters=max_attribute_characters,
        )
        if compute_pair_diagnostics:
            diagnostics = serialize_pairs(
                categories,
                left_names,
                left_attributes,
                right_names,
                right_attributes,
                max_attribute_characters=max_attribute_characters,
            )
            identity_matches = np.asarray(
                [pair.identity_matches for pair in diagnostics], dtype=np.int16
            )
            identity_conflicts = np.asarray(
                [pair.identity_conflicts for pair in diagnostics], dtype=np.int16
            )
            missing_identity_fields = np.asarray(
                [pair.missing_identity_fields for pair in diagnostics], dtype=np.int16
            )
        else:
            identity_matches = np.zeros(len(categories), dtype=np.int16)
            identity_conflicts = np.zeros(len(categories), dtype=np.int16)
            missing_identity_fields = np.zeros(len(categories), dtype=np.int16)
    elif serialization_mode == "pair_v2":
        serialized = serialize_pairs(
            categories,
            left_names,
            left_attributes,
            right_names,
            right_attributes,
            max_attribute_characters=max_attribute_characters,
        )
        left_texts = [pair.text for pair in serialized]
        right_texts = [""] * len(serialized)
        identity_matches = np.asarray(
            [pair.identity_matches for pair in serialized], dtype=np.int16
        )
        identity_conflicts = np.asarray(
            [pair.identity_conflicts for pair in serialized], dtype=np.int16
        )
        missing_identity_fields = np.asarray(
            [pair.missing_identity_fields for pair in serialized], dtype=np.int16
        )
    else:
        raise ValueError(f"unknown serialization mode: {serialization_mode}")

    if "surface_similarity" in batch.schema.names:
        surface_similarity = (
            batch.column("surface_similarity")
            .to_numpy(zero_copy_only=False)
            .astype(np.float32, copy=True)
        )
    else:
        surface_similarity = np.full(len(categories), np.nan, dtype=np.float32)
    return PairTextBatch(
        id1=batch.column("id1").to_numpy(zero_copy_only=False).astype(np.int64, copy=True),
        id2=batch.column("id2").to_numpy(zero_copy_only=False).astype(np.int64, copy=True),
        target=batch.column("target").to_numpy(zero_copy_only=False).astype(np.float32, copy=False),
        categories=categories,
        left_texts=left_texts,
        right_texts=right_texts,
        surface_similarity=surface_similarity,
        identity_matches=identity_matches,
        identity_conflicts=identity_conflicts,
        missing_identity_fields=missing_identity_fields,
    )


def _take_batch(batch: PairTextBatch, indices: np.ndarray) -> PairTextBatch:
    positions = indices.tolist()
    return PairTextBatch(
        id1=batch.id1[indices],
        id2=batch.id2[indices],
        target=batch.target[indices],
        categories=[batch.categories[index] for index in positions],
        left_texts=[batch.left_texts[index] for index in positions],
        right_texts=[batch.right_texts[index] for index in positions],
        surface_similarity=batch.surface_similarity[indices],
        identity_matches=batch.identity_matches[indices],
        identity_conflicts=batch.identity_conflicts[indices],
        missing_identity_fields=batch.missing_identity_fields[indices],
    )


def _concat_pair_batches(left: PairTextBatch, right: PairTextBatch) -> PairTextBatch:
    return PairTextBatch(
        id1=np.concatenate((left.id1, right.id1)),
        id2=np.concatenate((left.id2, right.id2)),
        target=np.concatenate((left.target, right.target)),
        categories=[*left.categories, *right.categories],
        left_texts=[*left.left_texts, *right.left_texts],
        right_texts=[*left.right_texts, *right.right_texts],
        surface_similarity=np.concatenate((left.surface_similarity, right.surface_similarity)),
        identity_matches=np.concatenate((left.identity_matches, right.identity_matches)),
        identity_conflicts=np.concatenate((left.identity_conflicts, right.identity_conflicts)),
        missing_identity_fields=np.concatenate(
            (left.missing_identity_fields, right.missing_identity_fields)
        ),
    )


def replay_rows_for_primary(primary_rows: int, replay_fraction: float) -> int:
    """Return replay rows needed to obtain the requested fraction in a mixed batch."""

    if primary_rows < 1:
        raise ValueError("primary_rows must be positive")
    if not 0.0 < replay_fraction < 0.5:
        raise ValueError("replay_fraction must be in (0, 0.5)")
    return max(1, round(primary_rows * replay_fraction / (1.0 - replay_fraction)))


def iter_mixed_pair_batches(
    primary_batches: Iterator[PairTextBatch],
    replay_batches: Iterator[PairTextBatch],
    *,
    replay_fraction: float,
) -> Iterator[tuple[PairTextBatch, int, int]]:
    """Append deterministic replay rows while preserving every primary training row."""

    if not 0.0 < replay_fraction < 0.5:
        raise ValueError("replay_fraction must be in (0, 0.5)")
    pending_replay: PairTextBatch | None = None
    for primary in primary_batches:
        required_replay_rows = replay_rows_for_primary(len(primary), replay_fraction)
        replay_parts: list[PairTextBatch] = []
        collected = 0
        while collected < required_replay_rows:
            if pending_replay is None:
                try:
                    pending_replay = next(replay_batches)
                except StopIteration as error:
                    raise ValueError(
                        "replay dataset exhausted before the primary epoch completed"
                    ) from error
            needed = required_replay_rows - collected
            take = min(needed, len(pending_replay))
            replay_parts.append(_take_batch(pending_replay, np.arange(take)))
            collected += take
            if take == len(pending_replay):
                pending_replay = None
            else:
                pending_replay = _take_batch(
                    pending_replay,
                    np.arange(take, len(pending_replay)),
                )

        replay = replay_parts[0]
        for part in replay_parts[1:]:
            replay = _concat_pair_batches(replay, part)
        yield _concat_pair_batches(primary, replay), len(primary), len(replay)


def _swap_pair_order(batch: PairTextBatch, rng: np.random.Generator, probability: float) -> None:
    if not 0.0 <= probability <= 1.0:
        raise ValueError("pair swap probability must be in [0, 1]")
    if probability == 0.0:
        return
    for index in np.flatnonzero(rng.random(len(batch)) < probability):
        batch.left_texts[index], batch.right_texts[index] = (
            batch.right_texts[index],
            batch.left_texts[index],
        )
        batch.id1[index], batch.id2[index] = batch.id2[index], batch.id1[index]


def _sample_indices(
    rng: np.random.Generator,
    candidates: np.ndarray,
    count: int,
) -> np.ndarray:
    if count <= 0:
        return np.empty(0, dtype=np.int64)
    if len(candidates) == 0:
        return np.empty(0, dtype=np.int64)
    return rng.choice(candidates, size=count, replace=len(candidates) < count).astype(
        np.int64, copy=False
    )


def _hard_replay_order(
    batch: PairTextBatch,
    rng: np.random.Generator,
    *,
    replay_fraction: float,
    thresholds: Mapping[str, HardPairThreshold],
    negative_target_max: float,
    positive_target_min: float,
) -> np.ndarray:
    """Allocate a fixed-size stream between base rows and train-only replay rows."""

    if replay_fraction == 0.0:
        return rng.permutation(len(batch))
    if not np.isfinite(batch.surface_similarity).all():
        raise ValueError("hard replay requires finite surface_similarity for every training row")

    negative_limits = np.empty(len(batch), dtype=np.float32)
    positive_limits = np.empty(len(batch), dtype=np.float32)
    for index, category in enumerate(batch.categories):
        try:
            threshold = thresholds[category]
        except KeyError as error:
            raise ValueError(f"missing hard-pair threshold for category: {category}") from error
        negative_limits[index] = threshold.hard_negative_min_similarity
        positive_limits[index] = threshold.hard_positive_max_similarity

    hard_negative = np.flatnonzero(
        (batch.target <= negative_target_max)
        & ((batch.surface_similarity >= negative_limits) | (batch.identity_conflicts > 0))
    )
    hard_positive = np.flatnonzero(
        (batch.target >= positive_target_min) & (batch.surface_similarity <= positive_limits)
    )
    easy_negative = np.flatnonzero(
        (batch.target <= negative_target_max)
        & (batch.surface_similarity < negative_limits)
        & (batch.identity_conflicts == 0)
    )
    if len(hard_negative) + len(hard_positive) == 0:
        return rng.permutation(len(batch))

    replay_rows = min(len(batch) - 1, max(1, round(len(batch) * replay_fraction)))
    hard_rows = (replay_rows + 1) // 2
    easy_rows = replay_rows - hard_rows
    hard_negative_rows = hard_rows // 2
    hard_positive_rows = hard_rows - hard_negative_rows
    if len(hard_negative) == 0:
        hard_positive_rows = hard_rows
        hard_negative_rows = 0
    elif len(hard_positive) == 0:
        hard_negative_rows = hard_rows
        hard_positive_rows = 0

    replay = np.concatenate(
        (
            _sample_indices(rng, hard_negative, hard_negative_rows),
            _sample_indices(rng, hard_positive, hard_positive_rows),
            _sample_indices(rng, easy_negative, easy_rows),
        )
    )
    if len(replay) < replay_rows:
        fallback = np.setdiff1d(np.arange(len(batch)), replay, assume_unique=False)
        replay = np.concatenate((replay, _sample_indices(rng, fallback, replay_rows - len(replay))))

    base_rows = len(batch) - len(replay)
    base = rng.choice(len(batch), size=base_rows, replace=False).astype(np.int64, copy=False)
    order = np.concatenate((base, replay))
    rng.shuffle(order)
    return order


def iter_pair_batches(
    path: Path,
    *,
    batch_size: int,
    read_batch_size: int,
    max_attribute_characters: int,
    seed: int,
    epoch: int,
    shuffle: bool,
    pair_swap_probability: float,
    serialization_mode: SerializationMode = "item_v1",
    hard_replay_fraction: float = 0.0,
    hard_pair_thresholds: Mapping[str, HardPairThreshold] | None = None,
    hard_negative_target_max: float = 0.2,
    hard_positive_target_min: float = 0.8,
    compute_pair_diagnostics: bool = False,
    max_rows: int | None = None,
    include_fold: int | None = None,
    exclude_fold: int | None = None,
    include_inner_fold: int | None = None,
    exclude_inner_fold: int | None = None,
) -> Iterator[PairTextBatch]:
    """Stream serialized pairs with deterministic row-group shuffling and bucketing."""

    if batch_size < 1 or read_batch_size < batch_size:
        raise ValueError("read_batch_size must be at least batch_size")
    if max_rows is not None and max_rows < 1:
        raise ValueError("max_rows must be positive")
    _validate_split_filters(
        include_fold=include_fold,
        exclude_fold=exclude_fold,
        include_inner_fold=include_inner_fold,
        exclude_inner_fold=exclude_inner_fold,
    )
    if serialization_mode == "pair_v2" and pair_swap_probability != 0.0:
        raise ValueError("pair_v2 is canonical; pair swap probability must be zero")
    if serialization_mode not in ("item_v1", "pair_v2"):
        raise ValueError(f"unknown serialization mode: {serialization_mode}")
    if not 0.0 <= hard_replay_fraction < 1.0:
        raise ValueError("hard_replay_fraction must be in [0, 1)")
    if hard_replay_fraction and not shuffle:
        raise ValueError("hard replay is only valid for shuffled training batches")
    if hard_replay_fraction and hard_pair_thresholds is None:
        raise ValueError("hard replay requires train-only category thresholds")
    if not 0.0 <= hard_negative_target_max < hard_positive_target_min <= 1.0:
        raise ValueError("hard target thresholds must satisfy 0 <= negative < positive <= 1")

    parquet = pq.ParquetFile(path)
    filtered_by_fold = include_fold is not None or exclude_fold is not None
    filtered_by_inner_fold = include_inner_fold is not None or exclude_inner_fold is not None
    if filtered_by_fold and "fold" not in parquet.schema_arrow.names:
        raise ValueError(f"fold filtering requested for dataset without fold column: {path}")
    if filtered_by_inner_fold and "inner_fold" not in parquet.schema_arrow.names:
        raise ValueError(f"inner fold filtering requested for dataset without inner_fold: {path}")
    columns = list(PAIR_COLUMNS)
    if filtered_by_fold:
        columns.append("fold")
    if filtered_by_inner_fold:
        columns.append("inner_fold")
    if "surface_similarity" in parquet.schema_arrow.names:
        columns.append("surface_similarity")
    rng = np.random.default_rng(seed + epoch * 1_000_003)
    row_groups = np.arange(parquet.num_row_groups)
    if shuffle:
        rng.shuffle(row_groups)

    emitted = 0
    pending: PairTextBatch | None = None
    for row_group in row_groups.tolist():
        batches = parquet.iter_batches(
            batch_size=read_batch_size,
            row_groups=[row_group],
            columns=columns,
            use_threads=True,
        )
        for arrow_batch in batches:
            if filtered_by_fold or filtered_by_inner_fold:
                mask = np.ones(len(arrow_batch), dtype=bool)
                if filtered_by_fold:
                    folds = arrow_batch.column("fold").to_numpy(zero_copy_only=False)
                    if include_fold is not None:
                        mask &= folds == include_fold
                    else:
                        mask &= folds != exclude_fold
                if filtered_by_inner_fold:
                    inner_folds = arrow_batch.column("inner_fold").to_numpy(zero_copy_only=False)
                    if include_inner_fold is not None:
                        mask &= inner_folds == include_inner_fold
                    else:
                        mask &= inner_folds != exclude_inner_fold
                positions = np.flatnonzero(mask)
                if len(positions) == 0:
                    continue
                arrow_batch = arrow_batch.take(pa.array(positions))
            batch = _serialize_record_batch(
                arrow_batch,
                max_attribute_characters=max_attribute_characters,
                serialization_mode=serialization_mode,
                compute_pair_diagnostics=compute_pair_diagnostics or hard_replay_fraction > 0.0,
            )
            _swap_pair_order(batch, rng, pair_swap_probability)

            if shuffle:
                order = _hard_replay_order(
                    batch,
                    rng,
                    replay_fraction=hard_replay_fraction,
                    thresholds=hard_pair_thresholds or {},
                    negative_target_max=hard_negative_target_max,
                    positive_target_min=hard_positive_target_min,
                )
                # Sorting by approximate token length reduces padding. Batch order remains random.
                order = order[
                    np.argsort(
                        np.fromiter(
                            (
                                len(batch.left_texts[index]) + len(batch.right_texts[index])
                                for index in order
                            ),
                            dtype=np.int64,
                        ),
                        kind="stable",
                    )
                ]
            else:
                order = np.arange(len(batch))

            ready: list[PairTextBatch] = []
            if pending is not None:
                needed = batch_size - len(pending)
                if len(order) < needed:
                    pending = _concat_pair_batches(pending, _take_batch(batch, order))
                    continue
                if shuffle:
                    merge_indices = order[-needed:]
                    order = order[:-needed]
                else:
                    merge_indices = order[:needed]
                    order = order[needed:]
                ready.append(_concat_pair_batches(pending, _take_batch(batch, merge_indices)))
                pending = None

            full_rows = len(order) - len(order) % batch_size
            ready.extend(
                _take_batch(batch, order[start : start + batch_size])
                for start in range(0, full_rows, batch_size)
            )
            if full_rows < len(order):
                pending = _take_batch(batch, order[full_rows:])
            if shuffle:
                rng.shuffle(ready)

            for output in ready:
                if max_rows is not None:
                    remaining = max_rows - emitted
                    if remaining <= 0:
                        return
                    if len(output) > remaining:
                        output = _take_batch(output, np.arange(remaining))
                emitted += len(output)
                yield output

    if pending is not None:
        if max_rows is not None:
            remaining = max_rows - emitted
            if remaining <= 0:
                return
            if len(pending) > remaining:
                pending = _take_batch(pending, np.arange(remaining))
        if len(pending):
            yield pending


def prefetch_batches[BatchValue](
    batches: Iterator[BatchValue],
) -> Iterator[BatchValue]:
    """Overlap bounded CPU/Arrow preparation with the current CUDA step."""

    sentinel = object()

    def next_or_sentinel() -> BatchValue | object:
        return next(batches, sentinel)

    with ThreadPoolExecutor(max_workers=1, thread_name_prefix="pair-prefetch") as executor:
        future = executor.submit(next_or_sentinel)
        while True:
            batch = future.result()
            if batch is sentinel:
                return
            future = executor.submit(next_or_sentinel)
            yield batch  # type: ignore[misc]


def tokenize_pair_batch(
    tokenizer: PreTrainedTokenizerBase,
    batch: PairTextBatch,
    *,
    max_length: int,
    device: torch.device,
) -> dict[str, torch.Tensor]:
    empty_right = [not value for value in batch.right_texts]
    if all(empty_right):
        encoded = tokenizer(
            batch.left_texts,
            padding=True,
            truncation=True,
            max_length=max_length,
            pad_to_multiple_of=8,
            return_tensors="pt",
        )
    elif any(empty_right):
        raise ValueError("a batch cannot mix single-text and paired-text serialization")
    else:
        encoded = tokenizer(
            batch.left_texts,
            batch.right_texts,
            padding=True,
            truncation=True,
            max_length=max_length,
            pad_to_multiple_of=8,
            return_tensors="pt",
        )
    return {name: tensor.to(device, non_blocking=True) for name, tensor in encoded.items()}


def _inference_logits(
    model: PreTrainedModel,
    tokenizer: PreTrainedTokenizerBase,
    batch: PairTextBatch,
    *,
    max_length: int,
    device: torch.device,
    bidirectional: bool,
) -> torch.Tensor:
    inputs = tokenize_pair_batch(tokenizer, batch, max_length=max_length, device=device)
    with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
        logits = model(**inputs).logits.reshape(-1).float()
    if not bidirectional:
        return logits
    if all(not text for text in batch.right_texts):
        raise ValueError("bidirectional scoring is invalid for canonical pair_v2 serialization")

    reverse = PairTextBatch(
        id1=batch.id2,
        id2=batch.id1,
        target=batch.target,
        categories=batch.categories,
        left_texts=batch.right_texts,
        right_texts=batch.left_texts,
        surface_similarity=batch.surface_similarity,
        identity_matches=batch.identity_matches,
        identity_conflicts=batch.identity_conflicts,
        missing_identity_fields=batch.missing_identity_fields,
    )
    reverse_inputs = tokenize_pair_batch(
        tokenizer,
        reverse,
        max_length=max_length,
        device=device,
    )
    with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
        reverse_logits = model(**reverse_inputs).logits.reshape(-1).float()
    return (logits + reverse_logits) * 0.5


def _metric_payload(
    targets: np.ndarray,
    scores: np.ndarray,
    categories: np.ndarray,
    validation_kind: str,
) -> tuple[float, dict[str, Any]]:
    if validation_kind == "human":
        report = macro_average_precision(targets, scores, categories)
        return report.score, {"human_macro_average_precision": asdict(report)}
    if validation_kind != "llm":
        raise ValueError(f"unknown validation kind: {validation_kind}")

    majority = macro_average_precision(targets >= (5.0 / 9.0), scores, categories)
    unanimous_mask = np.isin(targets, (0.0, 1.0))
    unanimous = macro_average_precision(
        targets[unanimous_mask],
        scores[unanimous_mask],
        categories[unanimous_mask],
    )
    rank = macro_spearman(targets, scores, categories)
    return majority.score, {
        "llm_majority_macro_average_precision": asdict(majority),
        "llm_unanimous_macro_average_precision": asdict(unanimous),
        "llm_soft_macro_spearman": asdict(rank),
        "unanimous_rows": int(unanimous_mask.sum()),
    }


def _challenge_average_precision(
    targets: np.ndarray,
    scores: np.ndarray,
    categories: np.ndarray,
    mask: np.ndarray,
) -> dict[str, Any]:
    per_category: dict[str, float] = {}
    rows_per_category: dict[str, int] = {}
    positives_per_category: dict[str, int] = {}
    excluded_categories: list[str] = []
    for category in sorted(set(categories.tolist())):
        category_mask = mask & (categories == category)
        category_targets = targets[category_mask]
        if len(category_targets) == 0 or len(np.unique(category_targets)) != 2:
            excluded_categories.append(str(category))
            continue
        per_category[str(category)] = float(
            average_precision_score(category_targets, scores[category_mask])
        )
        rows_per_category[str(category)] = int(category_mask.sum())
        positives_per_category[str(category)] = int(category_targets.sum())
    return {
        "score": float(np.mean(list(per_category.values()))) if per_category else None,
        "eligible_categories": len(per_category),
        "excluded_categories": excluded_categories,
        "per_category": per_category,
        "rows_per_category": rows_per_category,
        "positives_per_category": positives_per_category,
    }


def human_stress_suite(
    targets: np.ndarray,
    scores: np.ndarray,
    categories: np.ndarray,
    surface_similarity: np.ndarray,
    identity_conflicts: np.ndarray,
    thresholds: Mapping[str, HardPairThreshold],
) -> dict[str, Any]:
    """Evaluate fixed train-derived challenge slices without changing organizer metric."""

    if not (
        len(targets)
        == len(scores)
        == len(categories)
        == len(surface_similarity)
        == len(identity_conflicts)
    ):
        raise ValueError("stress-suite arrays must have equal lengths")
    if not np.isin(targets, (0.0, 1.0)).all():
        raise ValueError("human stress suite requires binary targets")
    if not np.isfinite(surface_similarity).all():
        raise ValueError("human stress suite requires finite surface similarity")

    negative_limits = np.empty(len(targets), dtype=np.float32)
    positive_limits = np.empty(len(targets), dtype=np.float32)
    for index, category in enumerate(categories.tolist()):
        try:
            threshold = thresholds[str(category)]
        except KeyError as error:
            raise ValueError(f"missing stress threshold for category: {category}") from error
        negative_limits[index] = threshold.hard_negative_min_similarity
        positive_limits[index] = threshold.hard_positive_max_similarity

    positives = targets == 1.0
    negatives = ~positives
    hard_negative_rows = negatives & (surface_similarity >= negative_limits)
    hard_positive_rows = positives & (surface_similarity <= positive_limits)
    conflict_negative_rows = negatives & (identity_conflicts > 0)
    return {
        "threshold_source": "training rows only",
        "hard_negative_rows": int(hard_negative_rows.sum()),
        "hard_positive_rows": int(hard_positive_rows.sum()),
        "conflict_negative_rows": int(conflict_negative_rows.sum()),
        "hard_negative_challenge": _challenge_average_precision(
            targets,
            scores,
            categories,
            positives | hard_negative_rows,
        ),
        "hard_positive_challenge": _challenge_average_precision(
            targets,
            scores,
            categories,
            negatives | hard_positive_rows,
        ),
        "identity_conflict_challenge": _challenge_average_precision(
            targets,
            scores,
            categories,
            positives | conflict_negative_rows,
        ),
    }


def evaluate_model(
    model: PreTrainedModel,
    tokenizer: PreTrainedTokenizerBase,
    validation_path: Path,
    *,
    validation_kind: str,
    device: torch.device,
    batch_size: int,
    read_batch_size: int,
    max_length: int,
    max_attribute_characters: int,
    seed: int,
    bidirectional: bool,
    serialization_mode: SerializationMode = "item_v1",
    stress_thresholds: Mapping[str, HardPairThreshold] | None = None,
    prediction_path: Path | None,
    max_rows: int | None = None,
    include_fold: int | None = None,
    exclude_fold: int | None = None,
    include_inner_fold: int | None = None,
    exclude_inner_fold: int | None = None,
) -> EvaluationResult:
    model.eval()
    torch.cuda.reset_peak_memory_stats(device)
    started = time.perf_counter()
    ids1: list[np.ndarray] = []
    ids2: list[np.ndarray] = []
    targets: list[np.ndarray] = []
    scores: list[np.ndarray] = []
    categories: list[str] = []
    surface_similarities: list[np.ndarray] = []
    identity_matches: list[np.ndarray] = []
    identity_conflicts: list[np.ndarray] = []
    missing_identity_fields: list[np.ndarray] = []

    batches = iter_pair_batches(
        validation_path,
        batch_size=batch_size,
        read_batch_size=read_batch_size,
        max_attribute_characters=max_attribute_characters,
        seed=seed,
        epoch=0,
        shuffle=False,
        pair_swap_probability=0.0,
        serialization_mode=serialization_mode,
        compute_pair_diagnostics=stress_thresholds is not None,
        max_rows=max_rows,
        include_fold=include_fold,
        exclude_fold=exclude_fold,
        include_inner_fold=include_inner_fold,
        exclude_inner_fold=exclude_inner_fold,
    )
    with torch.inference_mode():
        for batch in prefetch_batches(batches):
            logits = _inference_logits(
                model,
                tokenizer,
                batch,
                max_length=max_length,
                device=device,
                bidirectional=bidirectional,
            )
            ids1.append(batch.id1.copy())
            ids2.append(batch.id2.copy())
            targets.append(batch.target.copy())
            scores.append(logits.cpu().numpy())
            categories.extend(batch.categories)
            surface_similarities.append(batch.surface_similarity.copy())
            identity_matches.append(batch.identity_matches.copy())
            identity_conflicts.append(batch.identity_conflicts.copy())
            missing_identity_fields.append(batch.missing_identity_fields.copy())

    elapsed = time.perf_counter() - started
    all_id1 = np.concatenate(ids1)
    all_id2 = np.concatenate(ids2)
    all_targets = np.concatenate(targets)
    all_scores = np.concatenate(scores)
    all_categories = np.asarray(categories, dtype=object)
    all_surface_similarities = np.concatenate(surface_similarities)
    all_identity_matches = np.concatenate(identity_matches)
    all_identity_conflicts = np.concatenate(identity_conflicts)
    all_missing_identity_fields = np.concatenate(missing_identity_fields)
    if max_rows is None and len(set(all_categories.tolist())) != 20:
        raise ValueError("full validation must contain exactly 20 categories")
    selection_metric, metrics = _metric_payload(
        all_targets,
        all_scores,
        all_categories,
        validation_kind,
    )
    if stress_thresholds is not None:
        if validation_kind != "human":
            raise ValueError("stress suite is defined only for human gold validation")
        metrics["human_stress_suite"] = human_stress_suite(
            all_targets,
            all_scores,
            all_categories,
            all_surface_similarities,
            all_identity_conflicts,
            stress_thresholds,
        )

    if prediction_path is not None:
        prediction_path.parent.mkdir(parents=True, exist_ok=True)
        pq.write_table(
            pa.table(
                {
                    "id1": all_id1,
                    "id2": all_id2,
                    "target": all_targets,
                    "category": all_categories,
                    "score": all_scores,
                    "surface_similarity": all_surface_similarities,
                    "identity_matches": all_identity_matches,
                    "identity_conflicts": all_identity_conflicts,
                    "missing_identity_fields": all_missing_identity_fields,
                }
            ),
            prediction_path,
            compression="zstd",
        )

    return EvaluationResult(
        selection_metric=selection_metric,
        metrics=metrics,
        prediction_path=prediction_path,
        rows=len(all_targets),
        seconds=elapsed,
        rows_per_second=len(all_targets) / elapsed,
        peak_gpu_memory_bytes=torch.cuda.max_memory_allocated(device),
    )


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def training_steps(rows: int, batch_size: int, gradient_accumulation: int, epochs: int) -> int:
    batches = math.ceil(rows / batch_size)
    return math.ceil(batches / gradient_accumulation) * epochs


def bce_loss(logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
    """Reference row-mean BCE retained for exact v1 reproduction."""

    return weighted_bce_loss(logits, labels)


def category_row_counts(
    path: Path,
    *,
    include_fold: int | None = None,
    exclude_fold: int | None = None,
    include_inner_fold: int | None = None,
    exclude_inner_fold: int | None = None,
) -> dict[str, int]:
    """Count training categories under the same outer-fold filter as training."""

    _validate_split_filters(
        include_fold=include_fold,
        exclude_fold=exclude_fold,
        include_inner_fold=include_inner_fold,
        exclude_inner_fold=exclude_inner_fold,
    )
    parquet = pq.ParquetFile(path)
    filtered_by_fold = include_fold is not None or exclude_fold is not None
    filtered_by_inner_fold = include_inner_fold is not None or exclude_inner_fold is not None
    if filtered_by_fold and "fold" not in parquet.schema_arrow.names:
        raise ValueError(f"fold filtering requested for dataset without fold column: {path}")
    if filtered_by_inner_fold and "inner_fold" not in parquet.schema_arrow.names:
        raise ValueError(f"inner fold filtering requested for dataset without inner_fold: {path}")
    columns = ["category"]
    if filtered_by_fold:
        columns.append("fold")
    if filtered_by_inner_fold:
        columns.append("inner_fold")
    counts: dict[str, int] = {}
    for batch in parquet.iter_batches(batch_size=65_536, columns=columns, use_threads=True):
        categories = _python_strings(batch, "category")
        if filtered_by_fold or filtered_by_inner_fold:
            mask = np.ones(len(batch), dtype=bool)
            if filtered_by_fold:
                folds = batch.column("fold").to_numpy(zero_copy_only=False)
                if include_fold is not None:
                    mask &= folds == include_fold
                else:
                    mask &= folds != exclude_fold
            if filtered_by_inner_fold:
                inner_folds = batch.column("inner_fold").to_numpy(zero_copy_only=False)
                if include_inner_fold is not None:
                    mask &= inner_folds == include_inner_fold
                else:
                    mask &= inner_folds != exclude_inner_fold
            categories = [category for category, keep in zip(categories, mask, strict=True) if keep]
        for category in categories:
            counts[category] = counts.get(category, 0) + 1
    if not counts or any(not category or count < 1 for category, count in counts.items()):
        raise ValueError("training category counts must be non-empty and positive")
    return dict(sorted(counts.items()))


def inverse_category_weights(counts: dict[str, int]) -> dict[str, float]:
    """Give every observed category equal aggregate weight in the training objective."""

    if not counts or any(count < 1 for count in counts.values()):
        raise ValueError("category counts must be positive")
    total = sum(counts.values())
    categories = len(counts)
    return {category: total / (categories * count) for category, count in counts.items()}


def hard_pair_thresholds(
    path: Path,
    *,
    include_fold: int | None = None,
    exclude_fold: int | None = None,
    include_inner_fold: int | None = None,
    exclude_inner_fold: int | None = None,
    negative_target_max: float = 0.2,
    positive_target_min: float = 0.8,
    negative_similarity_quantile: float = 0.75,
    positive_similarity_quantile: float = 0.25,
) -> dict[str, HardPairThreshold]:
    """Fit category thresholds on training rows only; validation is never scanned."""

    _validate_split_filters(
        include_fold=include_fold,
        exclude_fold=exclude_fold,
        include_inner_fold=include_inner_fold,
        exclude_inner_fold=exclude_inner_fold,
    )
    if not 0.0 <= negative_target_max < positive_target_min <= 1.0:
        raise ValueError("hard target thresholds must satisfy 0 <= negative < positive <= 1")
    if not 0.0 < negative_similarity_quantile < 1.0:
        raise ValueError("negative_similarity_quantile must be in (0, 1)")
    if not 0.0 < positive_similarity_quantile < 1.0:
        raise ValueError("positive_similarity_quantile must be in (0, 1)")
    parquet = pq.ParquetFile(path)
    required = {"target", "category", "surface_similarity"}
    missing = required - set(parquet.schema_arrow.names)
    if missing:
        raise ValueError(f"hard-pair source is missing columns: {sorted(missing)}")
    if (
        include_fold is not None or exclude_fold is not None
    ) and "fold" not in parquet.schema_arrow.names:
        raise ValueError(f"fold filtering requested for dataset without fold column: {path}")
    if (
        include_inner_fold is not None or exclude_inner_fold is not None
    ) and "inner_fold" not in parquet.schema_arrow.names:
        raise ValueError(f"inner fold filtering requested for dataset without inner_fold: {path}")

    escaped = str(path.resolve()).replace("'", "''")
    filters: list[str] = []
    if include_fold is not None:
        filters.append(f"fold = {int(include_fold)}")
    elif exclude_fold is not None:
        filters.append(f"fold != {int(exclude_fold)}")
    if include_inner_fold is not None:
        filters.append(f"inner_fold = {int(include_inner_fold)}")
    elif exclude_inner_fold is not None:
        filters.append(f"inner_fold != {int(exclude_inner_fold)}")
    fold_filter = f"WHERE {' AND '.join(filters)}" if filters else ""
    connection = duckdb.connect()
    try:
        rows = connection.execute(
            f"""
            SELECT
                CAST(category AS VARCHAR) AS category,
                quantile_cont(surface_similarity, {negative_similarity_quantile})
                    FILTER (WHERE target <= {negative_target_max}) AS hard_negative_min,
                quantile_cont(surface_similarity, {positive_similarity_quantile})
                    FILTER (WHERE target >= {positive_target_min}) AS hard_positive_max,
                count(*) FILTER (WHERE target <= {negative_target_max}) AS negative_candidates,
                count(*) FILTER (WHERE target >= {positive_target_min}) AS positive_candidates
            FROM read_parquet('{escaped}')
            {fold_filter}
            GROUP BY category
            ORDER BY category
            """
        ).fetchall()
    finally:
        connection.close()

    result: dict[str, HardPairThreshold] = {}
    for category, negative_min, positive_max, negative_rows, positive_rows in rows:
        if negative_min is None or positive_max is None:
            raise ValueError(f"category lacks hard-pair candidates: {category}")
        result[str(category)] = HardPairThreshold(
            hard_negative_min_similarity=float(negative_min),
            hard_positive_max_similarity=float(positive_max),
            negative_candidates=int(negative_rows),
            positive_candidates=int(positive_rows),
        )
    if not result:
        raise ValueError("hard-pair source contains no categories")
    return result


def weighted_bce_loss(
    logits: torch.Tensor,
    labels: torch.Tensor,
    *,
    categories: list[str] | None = None,
    category_weights: dict[str, float] | None = None,
    confidence_gamma: float = 0.0,
) -> torch.Tensor:
    """Compute soft BCE with explicit category and soft-label confidence weights."""

    if logits.shape != labels.shape:
        raise ValueError("logits and labels must have equal shapes")
    if not torch.isfinite(labels).all() or torch.any((labels < 0.0) | (labels > 1.0)):
        raise ValueError("labels must be finite probabilities in [0, 1]")
    if confidence_gamma < 0.0 or not math.isfinite(confidence_gamma):
        raise ValueError("confidence_gamma must be finite and non-negative")
    if (categories is None) != (category_weights is None):
        raise ValueError("categories and category_weights must be provided together")
    if categories is not None and len(categories) != len(labels):
        raise ValueError("categories and labels must have equal lengths")

    losses = functional.binary_cross_entropy_with_logits(logits, labels, reduction="none")
    weights = torch.ones_like(losses)
    if category_weights is not None and categories is not None:
        try:
            resolved = [category_weights[category] for category in categories]
        except KeyError as error:
            raise ValueError(f"missing category weight: {error.args[0]}") from error
        weights = weights * torch.as_tensor(resolved, dtype=losses.dtype, device=losses.device)
    if confidence_gamma:
        confidence = torch.abs(labels * 2.0 - 1.0).pow(confidence_gamma)
        weights = weights * confidence
    denominator = weights.sum()
    if not torch.isfinite(denominator) or denominator <= 0.0:
        raise ValueError("effective loss weights must have a positive finite sum")
    return torch.sum(losses * weights) / denominator
