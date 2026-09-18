"""Offline multi-model inference for a frozen E-CUP Task 1 ensemble."""

from __future__ import annotations

import argparse
import gc
import json
import os
import time
from dataclasses import dataclass, replace
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from ecup_matching.neural_inference import (
    NeuralBundle,
    _join_pair_text,
    _load_inputs,
    _peak_process_rss_bytes,
    _safe_bundle_path,
    _score_neural_pairs,
)
from ecup_matching.serialization import ITEM_SERIALIZER_VERSION, serialize_items

_RUNTIME_RESOURCES_PREFIX = "runtime_resources="


@dataclass(frozen=True, slots=True)
class EnsembleModel:
    name: str
    bundle: NeuralBundle
    max_length_by_category: dict[str, int]


@dataclass(frozen=True, slots=True)
class EnsembleResidual:
    model_name: str
    categories: tuple[str, ...]
    reference_weight: float
    model_weight: float


@dataclass(frozen=True, slots=True)
class EnsemblePostBlend:
    model_name: str
    weights: dict[str, tuple[float, ...]]


@dataclass(frozen=True, slots=True)
class EnsembleBundle:
    models: tuple[EnsembleModel, ...]
    feature_order: tuple[str, ...]
    representation: str
    scope: str
    weights: tuple[float, ...] | dict[str, tuple[float, ...]]
    residual: EnsembleResidual | None
    post_blend: EnsemblePostBlend | None

    @classmethod
    def load(cls, bundle_root: Path) -> EnsembleBundle:
        manifest_path = bundle_root / "ensemble_manifest.json"
        if not manifest_path.is_file():
            raise FileNotFoundError(manifest_path)
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        if manifest.get("schema_version") != 1:
            raise ValueError("unsupported ensemble manifest schema_version")
        integrity = manifest.get("runtime_integrity")
        if integrity != {
            "artifact_size": True,
            "artifact_sha256": False,
            "archive_crc_and_sha256_verified_offline": True,
        }:
            raise ValueError("unsupported ensemble runtime integrity contract")

        models: list[EnsembleModel] = []
        for model in manifest.get("models", []):
            if not isinstance(model, dict):
                raise ValueError("invalid ensemble model entry")
            name = str(model["name"])
            model_path = (bundle_root / str(model["path"])).resolve()
            try:
                model_path.relative_to(bundle_root.resolve())
            except ValueError as error:
                raise ValueError("ensemble model path escapes bundle root") from error
            if not model_path.is_dir():
                raise FileNotFoundError(model_path)
            artifacts = model.get("artifacts")
            if not isinstance(artifacts, list) or not artifacts:
                raise ValueError(f"model artifacts are missing: {name}")
            for artifact in artifacts:
                path = _safe_bundle_path(bundle_root, str(artifact["path"]))
                if path.stat().st_size != int(artifact["bytes"]):
                    raise ValueError(f"model artifact size mismatch: {path}")
            inference = model.get("inference")
            if not isinstance(inference, dict):
                raise ValueError(f"model inference config is missing: {name}")
            if inference.get("serialization_mode") != "item_v1":
                raise ValueError("ensemble supports only the frozen item_v1 serializer")
            if inference.get("serialization_version") != ITEM_SERIALIZER_VERSION:
                raise ValueError("ensemble serializer version mismatch")
            bidirectional = inference.get("bidirectional")
            if not isinstance(bidirectional, bool):
                raise ValueError("ensemble models must declare boolean bidirectional inference")
            length_bucketing = inference.get("length_bucketing", False)
            if not isinstance(length_bucketing, bool):
                raise ValueError("ensemble model length_bucketing must be boolean")
            default_max_length = int(inference["max_length"])
            raw_length_router = inference.get("max_length_by_category", {})
            if not isinstance(raw_length_router, dict):
                raise ValueError("max_length_by_category must be an object")
            max_length_by_category = {
                str(category): int(max_length)
                for category, max_length in raw_length_router.items()
            }
            if any(max_length < 8 for max_length in max_length_by_category.values()):
                raise ValueError("routed max lengths must be at least eight")
            if any(
                max_length == default_max_length
                for max_length in max_length_by_category.values()
            ):
                raise ValueError("routed max lengths must differ from the model default")
            models.append(
                EnsembleModel(
                    name=name,
                    bundle=NeuralBundle(
                        model_path=model_path,
                        max_length=default_max_length,
                        max_attribute_characters=int(
                            inference["max_attribute_characters"]
                        ),
                        batch_size=int(inference["batch_size"]),
                        serialization_mode="item_v1",
                        serialization_version=ITEM_SERIALIZER_VERSION,
                        serialization_chunk_size=int(
                            inference["serialization_chunk_size"]
                        ),
                        bidirectional=bidirectional,
                        load_in_bf16=bool(inference.get("load_in_bf16", False)),
                        length_bucketing=length_bucketing,
                    ),
                    max_length_by_category=max_length_by_category,
                )
            )
        if len(models) < 2 or len({model.name for model in models}) != len(models):
            raise ValueError("ensemble requires at least two uniquely named models")

        ensemble = manifest.get("ensemble")
        if not isinstance(ensemble, dict):
            raise ValueError("ensemble configuration is missing")
        feature_order = tuple(str(name) for name in ensemble["feature_order"])
        model_names = tuple(model.name for model in models)
        if len(feature_order) < 2 or len(set(feature_order)) != len(feature_order):
            raise ValueError("ensemble feature order must contain unique model names")
        if any(name not in model_names for name in feature_order):
            raise ValueError("ensemble feature order contains an unknown packaged model")
        representation = str(ensemble["representation"])
        scope = str(ensemble["scope"])
        if representation not in ("raw_logit", "category_rank"):
            raise ValueError(f"unsupported ensemble representation: {representation}")
        if scope == "global":
            weights: tuple[float, ...] | dict[str, tuple[float, ...]] = tuple(
                float(value) for value in ensemble["weights"]
            )
            _validate_weights(weights, len(feature_order))
        elif scope == "category":
            weights = {
                str(category): tuple(float(value) for value in values)
                for category, values in ensemble["weights"].items()
            }
            if len(weights) != 20:
                raise ValueError("category ensemble must contain exactly 20 weight vectors")
            for values in weights.values():
                _validate_weights(values, len(feature_order))
        else:
            raise ValueError(f"unsupported ensemble scope: {scope}")
        residual_payload = manifest.get("residual")
        residual: EnsembleResidual | None = None
        if residual_payload is not None:
            if not isinstance(residual_payload, dict):
                raise ValueError("ensemble residual must be an object")
            if residual_payload.get("representation") != "category_rank":
                raise ValueError("ensemble residual supports only category_rank")
            if residual_payload.get("scope") != "category_subset":
                raise ValueError("ensemble residual supports only category_subset")
            residual_model_name = str(residual_payload["model_name"])
            if residual_model_name not in model_names:
                raise ValueError("ensemble residual references an unknown model")
            if residual_model_name in feature_order:
                raise ValueError("ensemble residual model must not be in base feature order")
            residual_categories = tuple(
                str(category) for category in residual_payload["categories"]
            )
            if not residual_categories or len(set(residual_categories)) != len(
                residual_categories
            ):
                raise ValueError("ensemble residual categories must be non-empty and unique")
            residual_weights = (
                float(residual_payload["reference_weight"]),
                float(residual_payload["model_weight"]),
            )
            _validate_weights(residual_weights, 2)
            residual = EnsembleResidual(
                model_name=residual_model_name,
                categories=residual_categories,
                reference_weight=residual_weights[0],
                model_weight=residual_weights[1],
            )
        elif feature_order != model_names:
            raise ValueError("unreferenced packaged models require an ensemble residual")

        post_blend_payload = manifest.get("post_blend")
        post_blend: EnsemblePostBlend | None = None
        if post_blend_payload is not None:
            if not isinstance(post_blend_payload, dict):
                raise ValueError("ensemble post_blend must be an object")
            if post_blend_payload.get("representation") != "category_rank":
                raise ValueError("ensemble post_blend supports only category_rank")
            if post_blend_payload.get("scope") != "category":
                raise ValueError("ensemble post_blend supports only category scope")
            post_blend_model_name = str(post_blend_payload["model_name"])
            if post_blend_model_name not in feature_order:
                raise ValueError("ensemble post_blend must reference a base feature model")
            post_blend_weights = {
                str(category): tuple(float(value) for value in values)
                for category, values in post_blend_payload["weights"].items()
            }
            if len(post_blend_weights) != 20:
                raise ValueError("category post_blend must contain exactly 20 weight vectors")
            for values in post_blend_weights.values():
                _validate_weights(values, 2)
            post_blend = EnsemblePostBlend(
                model_name=post_blend_model_name,
                weights=post_blend_weights,
            )

        routed_models = [model for model in models if model.max_length_by_category]
        if routed_models and scope != "category":
            raise ValueError("length routing requires category-scoped ensemble weights")
        if isinstance(weights, dict):
            for model_index, model_name in enumerate(feature_order):
                model = models[model_names.index(model_name)]
                for category in model.max_length_by_category:
                    category_weights = weights.get(category)
                    if category_weights is None:
                        raise ValueError(
                            f"length-routed category has no ensemble weights: {category}"
                        )
                    if category_weights[model_index] <= 0.0:
                        raise ValueError(
                            f"length-routed model is inactive for category: {category}"
                        )
        return cls(
            models=tuple(models),
            feature_order=feature_order,
            representation=representation,
            scope=scope,
            weights=weights,
            residual=residual,
            post_blend=post_blend,
        )


def _validate_weights(weights: tuple[float, ...], feature_count: int) -> None:
    values = np.asarray(weights, dtype=np.float64)
    if (
        len(values) != feature_count
        or not np.isfinite(values).all()
        or (values < 0.0).any()
        or not np.isclose(values.sum(), 1.0)
    ):
        raise ValueError("ensemble weights must be finite, nonnegative, and sum to one")


def _category_percentile_ranks(scores: np.ndarray, categories: np.ndarray) -> np.ndarray:
    frame = pd.DataFrame({"score": scores, "category": categories})
    ranks = frame.groupby("category", sort=False)["score"].rank(method="average")
    counts = frame.groupby("category", sort=False)["score"].transform("size")
    result = (ranks / (counts + 1.0)).to_numpy(dtype=np.float64)
    if not np.isfinite(result).all():
        raise RuntimeError("category rank normalization produced non-finite values")
    return result


def blend_scores(
    *,
    score_columns: list[np.ndarray],
    categories: np.ndarray,
    representation: str,
    scope: str,
    weights: tuple[float, ...] | dict[str, tuple[float, ...]],
) -> np.ndarray:
    if len(score_columns) < 2 or any(len(column) != len(categories) for column in score_columns):
        raise ValueError("ensemble score columns must be aligned and non-empty")
    if representation == "raw_logit":
        features = np.column_stack(score_columns).astype(np.float64, copy=False)
    elif representation == "category_rank":
        features = np.column_stack(
            [_category_percentile_ranks(column, categories) for column in score_columns]
        )
    else:
        raise ValueError(f"unsupported representation: {representation}")

    if scope == "global":
        if not isinstance(weights, tuple):
            raise ValueError("global ensemble requires one weight vector")
        _validate_weights(weights, features.shape[1])
        result = features @ np.asarray(weights, dtype=np.float64)
    elif scope == "category":
        if not isinstance(weights, dict):
            raise ValueError("category ensemble requires category weight vectors")
        result = np.empty(len(categories), dtype=np.float64)
        for category in np.unique(categories):
            category_weights = weights.get(str(category))
            if category_weights is None:
                raise ValueError(f"missing ensemble weights for category: {category}")
            _validate_weights(category_weights, features.shape[1])
            mask = categories == category
            result[mask] = features[mask] @ np.asarray(
                category_weights, dtype=np.float64
            )
    else:
        raise ValueError(f"unsupported scope: {scope}")
    if not np.isfinite(result).all():
        raise RuntimeError("ensemble produced non-finite scores")
    return result


def active_model_mask(
    *,
    categories: np.ndarray,
    model_index: int,
    scope: str,
    weights: tuple[float, ...] | dict[str, tuple[float, ...]],
) -> np.ndarray:
    """Return rows whose routed blend has a non-zero weight for one model."""

    if scope == "global":
        if not isinstance(weights, tuple):
            raise ValueError("global ensemble requires one weight vector")
        return np.full(len(categories), weights[model_index] > 0.0, dtype=bool)
    if scope != "category" or not isinstance(weights, dict):
        raise ValueError("category ensemble requires category weight vectors")
    active = np.empty(len(categories), dtype=bool)
    for category in np.unique(categories):
        category_weights = weights.get(str(category))
        if category_weights is None:
            raise ValueError(f"missing ensemble weights for category: {category}")
        active[categories == category] = category_weights[model_index] > 0.0
    return active


def active_model_length_masks(
    *,
    categories: np.ndarray,
    model_index: int,
    model: EnsembleModel,
    scope: str,
    weights: tuple[float, ...] | dict[str, tuple[float, ...]],
) -> dict[int, np.ndarray]:
    """Partition active model rows by the deterministic category length router."""

    active = active_model_mask(
        categories=categories,
        model_index=model_index,
        scope=scope,
        weights=weights,
    )
    lengths = np.full(len(categories), model.bundle.max_length, dtype=np.int32)
    for category, max_length in model.max_length_by_category.items():
        lengths[categories == category] = max_length
    return {
        int(max_length): active & (lengths == max_length)
        for max_length in sorted(np.unique(lengths[active]).tolist())
    }


def apply_category_rank_residual(
    *,
    reference: np.ndarray,
    residual_score: np.ndarray,
    categories: np.ndarray,
    residual: EnsembleResidual,
) -> np.ndarray:
    """Apply a frozen rank residual only inside its explicit category subset."""

    if not (len(reference) == len(residual_score) == len(categories)) or not len(reference):
        raise ValueError("residual score inputs must be non-empty and aligned")
    if not np.isfinite(np.column_stack((reference, residual_score))).all():
        raise ValueError("residual score inputs must be finite")
    routed = np.isin(categories, residual.categories)
    if not routed.any():
        return np.asarray(reference, dtype=np.float64).copy()
    result = np.asarray(reference, dtype=np.float64).copy()
    reference_rank = _category_percentile_ranks(reference, categories)
    residual_rank = _category_percentile_ranks(residual_score, categories)
    result[routed] = (
        residual.reference_weight * reference_rank[routed]
        + residual.model_weight * residual_rank[routed]
    )
    if not np.array_equal(result[~routed], reference[~routed]):
        raise RuntimeError("residual changed non-routed reference predictions")
    return result


def predict_ensemble(
    *,
    items_path: Path,
    matches_path: Path,
    output_path: Path,
    bundle_root: Path,
) -> pd.DataFrame:
    started = time.perf_counter()
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA GPU is required for ensemble inference")
    torch.cuda.reset_peak_memory_stats()
    bundle = EnsembleBundle.load(bundle_root)
    items, matches = _load_inputs(items_path, matches_path)
    names1, attributes1, names2, attributes2, categories = _join_pair_text(items, matches)

    attribute_limits = {model.bundle.max_attribute_characters for model in bundle.models}
    if len(attribute_limits) != 1:
        raise ValueError("shared item serialization requires one attribute limit")
    item_texts = serialize_items(
        items["category"].fillna("").astype(str).tolist(),
        items["name"].fillna("").astype(str).tolist(),
        items["attributes"].fillna("").astype(str).tolist(),
        max_attribute_characters=attribute_limits.pop(),
    )
    text_by_id = pd.Series(np.asarray(item_texts, dtype=object), index=items["id"])
    serialized1 = text_by_id.reindex(matches["id1"].to_numpy()).to_numpy()
    serialized2 = text_by_id.reindex(matches["id2"].to_numpy()).to_numpy()
    if pd.isna(serialized1).any() or pd.isna(serialized2).any():
        raise RuntimeError("pre-serialized item lookup lost pair coverage")

    score_by_model: dict[str, np.ndarray] = {}
    for model in bundle.models:
        if model.name in bundle.feature_order:
            model_index = bundle.feature_order.index(model.name)
            length_masks = active_model_length_masks(
                categories=categories,
                model_index=model_index,
                model=model,
                scope=bundle.scope,
                weights=bundle.weights,
            )
            if bundle.post_blend is not None and model.name == bundle.post_blend.model_name:
                post_blend_active = active_model_mask(
                    categories=categories,
                    model_index=1,
                    scope="category",
                    weights=bundle.post_blend.weights,
                )
                routed_lengths = np.full(
                    len(categories), model.bundle.max_length, dtype=np.int32
                )
                for category, max_length in model.max_length_by_category.items():
                    routed_lengths[categories == category] = max_length
                for max_length in np.unique(routed_lengths[post_blend_active]).tolist():
                    required = post_blend_active & (routed_lengths == max_length)
                    existing = length_masks.get(int(max_length))
                    length_masks[int(max_length)] = (
                        required if existing is None else existing | required
                    )
        elif bundle.residual is not None and model.name == bundle.residual.model_name:
            active = np.isin(categories, bundle.residual.categories)
            if model.max_length_by_category:
                raise ValueError("residual models do not support max_length_by_category")
            length_masks = {model.bundle.max_length: active} if active.any() else {}
        else:
            raise RuntimeError(f"packaged model has no inference route: {model.name}")
        active_rows = sum(int(np.count_nonzero(mask)) for mask in length_masks.values())
        print(
            f"ensemble_model_start={model.name} active_rows={active_rows}",
            flush=True,
        )
        scores = np.zeros(len(categories), dtype=np.float32)
        for max_length, active in length_masks.items():
            routed_rows = int(np.count_nonzero(active))
            print(
                f"ensemble_model_route={model.name} max_length={max_length} "
                f"active_rows={routed_rows}",
                flush=True,
            )
            scores[active] = _score_neural_pairs(
                bundle=replace(model.bundle, max_length=max_length),
                names1=serialized1[active],
                attributes1=attributes1[active],
                names2=serialized2[active],
                attributes2=attributes2[active],
                categories=categories[active],
                pre_serialized=True,
            )
            gc.collect()
            torch.cuda.empty_cache()
        score_by_model[model.name] = scores
        gc.collect()
        torch.cuda.empty_cache()
        print(f"ensemble_model_done={model.name}", flush=True)

    predictions = blend_scores(
        score_columns=[score_by_model[name] for name in bundle.feature_order],
        categories=categories,
        representation=bundle.representation,
        scope=bundle.scope,
        weights=bundle.weights,
    )
    if bundle.residual is not None:
        predictions = apply_category_rank_residual(
            reference=predictions,
            residual_score=score_by_model[bundle.residual.model_name],
            categories=categories,
            residual=bundle.residual,
        )
    if bundle.post_blend is not None:
        predictions = blend_scores(
            score_columns=[predictions, score_by_model[bundle.post_blend.model_name]],
            categories=categories,
            representation="category_rank",
            scope="category",
            weights=bundle.post_blend.weights,
        )
    result = pd.DataFrame(
        {
            "id1": matches["id1"].to_numpy(copy=True),
            "id2": matches["id2"].to_numpy(copy=True),
            "predict": predictions,
        }
    )
    output_path = output_path.resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = output_path.with_name(f".{output_path.name}.{os.getpid()}.tmp")
    result.to_csv(temporary_path, index=False)
    os.replace(temporary_path, output_path)
    elapsed = time.perf_counter() - started
    print(f"wrote={output_path} rows={len(result)} elapsed_seconds={elapsed:.3f}", flush=True)
    device = torch.cuda.current_device()
    resources = {
        "schema_version": 1,
        "pairs": len(result),
        "elapsed_seconds": elapsed,
        "peak_process_rss_bytes": _peak_process_rss_bytes(),
        "peak_gpu_allocated_bytes": torch.cuda.max_memory_allocated(device),
        "peak_gpu_reserved_bytes": torch.cuda.max_memory_reserved(device),
        "gpu_total_memory_bytes": torch.cuda.get_device_properties(device).total_memory,
        "gpu_name": torch.cuda.get_device_name(device),
    }
    print(_RUNTIME_RESOURCES_PREFIX + json.dumps(resources, sort_keys=True), flush=True)
    return result


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--items_path", "--items-path", "-i", dest="items_path", required=True)
    parser.add_argument("--matches_path", "--matches-path", dest="matches_path", required=True)
    parser.add_argument(
        "--output_path", "--output-path", "-o", dest="output_path", required=True
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    args = _parse_args(argv)
    predict_ensemble(
        items_path=Path(args.items_path),
        matches_path=Path(args.matches_path),
        output_path=Path(args.output_path),
        bundle_root=Path(__file__).resolve().parents[1],
    )


if __name__ == "__main__":
    main()
