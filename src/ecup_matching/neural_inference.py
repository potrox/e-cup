"""Offline neural-only inference for E-CUP Task 1."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer

from ecup_matching.serialization import (
    ITEM_SERIALIZER_VERSION,
    PAIR_SERIALIZER_VERSION,
    serialize_items,
    serialize_pairs,
)

_ITEM_COLUMNS = ("id", "name", "attributes", "category")
_MATCH_COLUMNS = ("id1", "id2")
_RUNTIME_RESOURCES_PREFIX = "runtime_resources="


def _sha256(path: Path, chunk_size: int = 8 * 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


def _peak_process_rss_bytes() -> int | None:
    """Return peak RSS for this process when the host exposes getrusage."""
    try:
        import resource
    except ImportError:
        return None
    peak_rss = int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
    # Linux reports KiB while macOS reports bytes.
    return peak_rss if sys.platform == "darwin" else peak_rss * 1024


def _safe_bundle_path(bundle_root: Path, relative_path: str) -> Path:
    candidate = (bundle_root / relative_path).resolve()
    try:
        candidate.relative_to(bundle_root.resolve())
    except ValueError as error:
        raise ValueError(f"artifact path escapes bundle root: {relative_path}") from error
    if not candidate.is_file():
        raise FileNotFoundError(candidate)
    return candidate


@dataclass(frozen=True, slots=True)
class NeuralBundle:
    model_path: Path
    max_length: int
    max_attribute_characters: int
    batch_size: int
    serialization_mode: str
    serialization_version: str
    serialization_chunk_size: int
    bidirectional: bool
    load_in_bf16: bool = False
    length_bucketing: bool = False

    @classmethod
    def load(cls, bundle_root: Path) -> NeuralBundle:
        manifest_path = bundle_root / "neural_manifest.json"
        if not manifest_path.is_file():
            raise FileNotFoundError(manifest_path)
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        if manifest.get("schema_version") != 1:
            raise ValueError("unsupported neural manifest schema_version")

        model = manifest.get("model")
        if not isinstance(model, dict):
            raise ValueError("neural manifest is missing model metadata")
        model_path = (bundle_root / str(model["path"])).resolve()
        try:
            model_path.relative_to(bundle_root.resolve())
        except ValueError as error:
            raise ValueError("neural model path escapes bundle root") from error
        if not model_path.is_dir():
            raise FileNotFoundError(model_path)
        artifacts = model.get("artifacts")
        if not isinstance(artifacts, list) or not artifacts:
            raise ValueError("neural model artifacts are missing")
        for artifact in artifacts:
            path = _safe_bundle_path(bundle_root, str(artifact["path"]))
            if path.stat().st_size != int(artifact["bytes"]):
                raise ValueError(f"neural artifact size mismatch: {path}")
            if _sha256(path) != str(artifact["sha256"]):
                raise ValueError(f"neural artifact SHA-256 mismatch: {path}")

        config = manifest.get("inference")
        if not isinstance(config, dict):
            raise ValueError("neural manifest is missing inference config")
        serialization_mode = str(config["serialization_mode"])
        if serialization_mode not in ("item_v1", "pair_v2"):
            raise ValueError(f"unsupported neural serialization mode: {serialization_mode}")
        expected_version = (
            PAIR_SERIALIZER_VERSION if serialization_mode == "pair_v2" else ITEM_SERIALIZER_VERSION
        )
        serialization_version = str(config["serialization_version"])
        if serialization_version != expected_version:
            raise ValueError(
                "bundle serializer version does not match packaged runtime: "
                f"{serialization_version} != {expected_version}"
            )
        batch_size = int(config["batch_size"])
        chunk_size = int(config["serialization_chunk_size"])
        max_length = int(config["max_length"])
        max_attribute_characters = int(config["max_attribute_characters"])
        bidirectional = config.get("bidirectional")
        if not isinstance(bidirectional, bool):
            raise ValueError("neural manifest must declare boolean bidirectional inference")
        load_in_bf16 = config.get("load_in_bf16", False)
        if not isinstance(load_in_bf16, bool):
            raise ValueError("neural manifest load_in_bf16 must be boolean")
        length_bucketing = config.get("length_bucketing", False)
        if not isinstance(length_bucketing, bool):
            raise ValueError("neural manifest length_bucketing must be boolean")
        if serialization_mode == "pair_v2" and bidirectional:
            raise ValueError("pair_v2 is canonical and cannot use bidirectional inference")
        if batch_size < 1 or chunk_size < batch_size:
            raise ValueError("invalid neural batch or serialization chunk size")
        if max_length < 8 or max_attribute_characters < 1:
            raise ValueError("invalid neural text limits")
        return cls(
            model_path=model_path,
            max_length=max_length,
            max_attribute_characters=max_attribute_characters,
            batch_size=batch_size,
            serialization_mode=serialization_mode,
            serialization_version=serialization_version,
            serialization_chunk_size=chunk_size,
            bidirectional=bidirectional,
            load_in_bf16=load_in_bf16,
            length_bucketing=length_bucketing,
        )


def _validate_required_columns(
    frame: pd.DataFrame,
    required: tuple[str, ...],
    source_name: str,
) -> None:
    missing = sorted(set(required) - set(frame.columns))
    if missing:
        raise ValueError(f"{source_name} is missing required columns: {missing}")


def _load_inputs(items_path: Path, matches_path: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    if not items_path.is_file():
        raise FileNotFoundError(items_path)
    if not matches_path.is_file():
        raise FileNotFoundError(matches_path)
    items = pd.read_parquet(items_path, columns=list(_ITEM_COLUMNS))
    matches = pd.read_parquet(matches_path, columns=list(_MATCH_COLUMNS))
    _validate_required_columns(items, _ITEM_COLUMNS, "items")
    _validate_required_columns(matches, _MATCH_COLUMNS, "matches")
    if items.empty:
        raise ValueError("items input is empty")
    if matches.empty:
        raise ValueError("matches input is empty")
    if items["id"].isna().any() or items["id"].duplicated().any():
        raise ValueError("item IDs must be non-null and unique")
    if matches[list(_MATCH_COLUMNS)].isna().any().any():
        raise ValueError("match IDs must be non-null")
    return items, matches


def _join_pair_text(
    items: pd.DataFrame,
    matches: pd.DataFrame,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    indexed = items.set_index("id", verify_integrity=True)
    left = indexed.reindex(matches["id1"].to_numpy())
    right = indexed.reindex(matches["id2"].to_numpy())
    missing_left = left["category"].isna().to_numpy()
    missing_right = right["category"].isna().to_numpy()
    if missing_left.any() or missing_right.any():
        missing_count = int(np.count_nonzero(missing_left | missing_right))
        raise ValueError(f"{missing_count} pairs reference missing item IDs")
    categories_left = left["category"].astype(str).to_numpy()
    categories_right = right["category"].astype(str).to_numpy()
    mismatch = categories_left != categories_right
    if mismatch.any():
        raise ValueError(f"{int(np.count_nonzero(mismatch))} pairs cross item categories")
    return (
        left["name"].fillna("").astype(str).to_numpy(),
        left["attributes"].fillna("").astype(str).to_numpy(),
        right["name"].fillna("").astype(str).to_numpy(),
        right["attributes"].fillna("").astype(str).to_numpy(),
        categories_left,
    )


def _score_neural_pairs(
    *,
    bundle: NeuralBundle,
    names1: np.ndarray,
    attributes1: np.ndarray,
    names2: np.ndarray,
    attributes2: np.ndarray,
    categories: np.ndarray,
    pre_serialized: bool = False,
) -> np.ndarray:
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA GPU is required for neural inference")
    device = torch.device("cuda")
    torch.set_float32_matmul_precision("high")
    torch.backends.cuda.matmul.allow_tf32 = True
    tokenizer = AutoTokenizer.from_pretrained(bundle.model_path, local_files_only=True)
    model_kwargs: dict[str, object] = {
        "num_labels": 1,
        "attn_implementation": "sdpa",
        "local_files_only": True,
    }
    if bundle.load_in_bf16:
        model_kwargs["dtype"] = torch.bfloat16
    model = AutoModelForSequenceClassification.from_pretrained(
        bundle.model_path,
        **model_kwargs,
    )
    model.config.compile_model = False
    model.to(device)
    model.eval()

    def stable_length_bucket_order(
        left: list[str],
        right: list[str] | None,
    ) -> np.ndarray:
        if not bundle.length_bucketing:
            return np.arange(len(left))
        approximate_lengths = np.fromiter(
            (
                len(left[index]) + (len(right[index]) if right is not None else 0)
                for index in range(len(left))
            ),
            dtype=np.int64,
            count=len(left),
        )
        return np.argsort(approximate_lengths, kind="stable")

    def serialize_chunk(start: int) -> tuple[np.ndarray, list[str], list[str] | None]:
        stop = min(start + bundle.serialization_chunk_size, len(categories))
        positions = np.arange(start, stop)
        chunk_categories = categories[start:stop].tolist()
        if pre_serialized:
            if bundle.serialization_mode != "item_v1":
                raise ValueError("pre-serialized inference supports only item_v1")
            left = names1[start:stop].tolist()
            right = names2[start:stop].tolist()
        elif bundle.serialization_mode == "pair_v2":
            pairs = serialize_pairs(
                chunk_categories,
                names1[start:stop].tolist(),
                attributes1[start:stop].tolist(),
                names2[start:stop].tolist(),
                attributes2[start:stop].tolist(),
                max_attribute_characters=bundle.max_attribute_characters,
            )
            left = [pair.text for pair in pairs]
            right: list[str] | None = None
        else:
            left = serialize_items(
                chunk_categories,
                names1[start:stop].tolist(),
                attributes1[start:stop].tolist(),
                max_attribute_characters=bundle.max_attribute_characters,
            )
            right = serialize_items(
                chunk_categories,
                names2[start:stop].tolist(),
                attributes2[start:stop].tolist(),
                max_attribute_characters=bundle.max_attribute_characters,
            )
        order = stable_length_bucket_order(left, right)
        if bundle.length_bucketing:
            positions = positions[order]
            left = [left[index] for index in order]
            if right is not None:
                right = [right[index] for index in order]
        return positions, left, right

    scores = np.empty(len(categories), dtype=np.float32)
    started = time.perf_counter()
    completed = 0
    chunk_starts = iter(range(0, len(categories), bundle.serialization_chunk_size))
    with (
        torch.inference_mode(),
        torch.autocast(device_type="cuda", dtype=torch.bfloat16),
        ThreadPoolExecutor(max_workers=1, thread_name_prefix="runtime-serialize") as executor,
    ):
        first_start = next(chunk_starts, None)
        future = executor.submit(serialize_chunk, first_start) if first_start is not None else None
        while future is not None:
            positions, left_texts, right_texts = future.result()
            next_start = next(chunk_starts, None)
            future = (
                executor.submit(serialize_chunk, next_start) if next_start is not None else None
            )
            for batch_start in range(0, len(positions), bundle.batch_size):
                batch_stop = batch_start + bundle.batch_size
                batch_positions = positions[batch_start:batch_stop]
                left = left_texts[batch_start:batch_stop]
                right = right_texts[batch_start:batch_stop] if right_texts is not None else None

                def score_direction(
                    first: list[str],
                    second: list[str] | None,
                ) -> torch.Tensor:
                    inputs = tokenizer(
                        first,
                        second,
                        padding=True,
                        truncation=True,
                        max_length=bundle.max_length,
                        pad_to_multiple_of=8,
                        return_tensors="pt",
                    )
                    inputs = {
                        name: tensor.to(device, non_blocking=True)
                        for name, tensor in inputs.items()
                    }
                    return model(**inputs).logits.reshape(-1).float()

                logits = score_direction(left, right)
                if bundle.bidirectional:
                    if right is None:
                        raise RuntimeError("bidirectional inference requires item-pair inputs")
                    logits = (logits + score_direction(right, left)) * 0.5
                scores[batch_positions] = logits.cpu().numpy()
                completed += len(batch_positions)
                if completed % (bundle.batch_size * 100) == 0 or completed == len(scores):
                    elapsed = max(time.perf_counter() - started, 1e-9)
                    print(
                        f"neural={completed}/{len(scores)} "
                        f"rate={completed / elapsed:.0f}_pairs_per_second",
                        flush=True,
                    )
    if not np.isfinite(scores).all():
        raise RuntimeError("neural model produced non-finite scores")
    return scores


def predict_neural(
    *,
    items_path: Path,
    matches_path: Path,
    output_path: Path,
    bundle_root: Path,
) -> pd.DataFrame:
    started = time.perf_counter()
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()
    bundle = NeuralBundle.load(bundle_root)
    items, matches = _load_inputs(items_path, matches_path)
    names1, attributes1, names2, attributes2, categories = _join_pair_text(items, matches)
    predictions = _score_neural_pairs(
        bundle=bundle,
        names1=names1,
        attributes1=attributes1,
        names2=names2,
        attributes2=attributes2,
        categories=categories,
    )
    result = pd.DataFrame(
        {
            "id1": matches["id1"].to_numpy(copy=True),
            "id2": matches["id2"].to_numpy(copy=True),
            "predict": predictions.astype(np.float64),
        }
    )
    output_path = output_path.resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = output_path.with_name(f".{output_path.name}.{os.getpid()}.tmp")
    result.to_csv(temporary_path, index=False)
    os.replace(temporary_path, output_path)
    elapsed_seconds = time.perf_counter() - started
    print(
        f"wrote={output_path} rows={len(result)} elapsed_seconds={elapsed_seconds:.3f}",
        flush=True,
    )
    device = torch.cuda.current_device()
    resources = {
        "schema_version": 1,
        "pairs": len(result),
        "elapsed_seconds": elapsed_seconds,
        "peak_process_rss_bytes": _peak_process_rss_bytes(),
        "peak_gpu_allocated_bytes": torch.cuda.max_memory_allocated(device),
        "peak_gpu_reserved_bytes": torch.cuda.max_memory_reserved(device),
        "gpu_total_memory_bytes": torch.cuda.get_device_properties(device).total_memory,
        "gpu_name": torch.cuda.get_device_name(device),
    }
    print(
        _RUNTIME_RESOURCES_PREFIX + json.dumps(resources, sort_keys=True),
        flush=True,
    )
    return result


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--items_path", "--items-path", "-i", dest="items_path", required=True)
    parser.add_argument("--matches_path", "--matches-path", dest="matches_path", required=True)
    parser.add_argument(
        "--output_path",
        "--output-path",
        "-o",
        dest="output_path",
        required=True,
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    args = _parse_args(argv)
    predict_neural(
        items_path=Path(args.items_path),
        matches_path=Path(args.matches_path),
        output_path=Path(args.output_path),
        bundle_root=Path(__file__).resolve().parents[1],
    )


if __name__ == "__main__":
    main()
