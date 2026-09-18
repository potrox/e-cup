from __future__ import annotations

import argparse
import hashlib
import json
import os
import time
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np
import polars as pl
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from ecup_matching.metrics import macro_average_precision
from ecup_matching.teacher_distillation import (
    TEACHER_PROMPT_VERSION,
    build_teacher_user_prompt,
    choose_binary_token_ids,
)


def _parse_categories(raw: str) -> tuple[str, ...]:
    values = tuple(value.strip() for value in raw.split(",") if value.strip())
    if not values or len(values) != len(set(values)):
        raise argparse.ArgumentTypeError("categories must be unique and non-empty")
    return values


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Score local product pairs with a causal LLM teacher"
    )
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--categories", type=_parse_categories)
    parser.add_argument("--prompt-mode", choices=("strict", "category_rules"), required=True)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--max-length", type=int, default=768)
    parser.add_argument("--max-attribute-characters", type=int, default=1600)
    parser.add_argument("--include-fold", type=int)
    parser.add_argument("--exclude-fold", type=int)
    parser.add_argument("--include-inner-fold", type=int)
    parser.add_argument("--exclude-inner-fold", type=int)
    parser.add_argument("--max-rows", type=int)
    parser.add_argument("--shard-count", type=int, default=1)
    parser.add_argument("--shard-index", type=int, default=0)
    parser.add_argument("--bidirectional", action="store_true")
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def _sha256(path: Path, chunk_size: int = 8 * 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while block := handle.read(chunk_size):
            digest.update(block)
    return digest.hexdigest()


def _load_rows(args: argparse.Namespace) -> pl.DataFrame:
    frame = pl.read_parquet(args.input)
    if "teacher_source_row" not in frame.columns:
        frame = frame.with_row_index("teacher_source_row")
    elif frame["teacher_source_row"].null_count() or (
        frame["teacher_source_row"].n_unique() != frame.height
    ):
        raise ValueError("teacher_source_row must be non-null and unique")
    required = {
        "id1",
        "id2",
        "target",
        "category",
        "left_name",
        "left_attributes",
        "right_name",
        "right_attributes",
    }
    missing = required - set(frame.columns)
    if missing:
        raise ValueError(f"teacher input is missing columns: {sorted(missing)}")
    for column, include, exclude in (
        ("fold", args.include_fold, args.exclude_fold),
        ("inner_fold", args.include_inner_fold, args.exclude_inner_fold),
    ):
        if include is not None and exclude is not None:
            raise ValueError(f"{column} filters are mutually exclusive")
        if (include is not None or exclude is not None) and column not in frame.columns:
            raise ValueError(f"teacher input has no {column} column")
        if include is not None:
            frame = frame.filter(pl.col(column) == include)
        elif exclude is not None:
            frame = frame.filter(pl.col(column) != exclude)
    if args.categories is not None:
        frame = frame.filter(pl.col("category").is_in(args.categories))
    if args.shard_count < 1 or not 0 <= args.shard_index < args.shard_count:
        raise ValueError("shard-index must be in [0, shard-count)")
    if args.shard_count > 1:
        frame = frame.filter(
            pl.col("teacher_source_row") % args.shard_count == args.shard_index
        )
    if args.max_rows is not None:
        if args.max_rows < 1:
            raise ValueError("max-rows must be positive")
        frame = frame.sort(
            pl.struct("category", "id1", "id2").hash(seed=args.seed)
        ).head(args.max_rows)
    if frame.is_empty():
        raise ValueError("teacher filters selected no rows")
    if frame.select(pl.struct("id1", "id2").n_unique()).item() != frame.height:
        raise ValueError("teacher input contains duplicate pair keys")
    return frame


def _chat_prompt(tokenizer: Any, user_prompt: str) -> str:
    messages = [
        {
            "role": "system",
            "content": (
                "Ты строгий эксперт по дедупликации товарных карточек. "
                "Следуй пользовательскому критерию и возвращай только 0 или 1."
            ),
        },
        {"role": "user", "content": user_prompt},
    ]
    try:
        return tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=False,
        )
    except TypeError:
        return tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )


def _build_prompts(
    frame: pl.DataFrame,
    tokenizer: Any,
    args: argparse.Namespace,
    *,
    reverse: bool,
) -> list[str]:
    prompts: list[str] = []
    for row in frame.iter_rows(named=True):
        user = build_teacher_user_prompt(
            category=row["category"],
            left_name=row["left_name"],
            left_attributes=row["left_attributes"],
            right_name=row["right_name"],
            right_attributes=row["right_attributes"],
            mode=args.prompt_mode,
            reverse=reverse,
            max_attribute_characters=args.max_attribute_characters,
        )
        prompts.append(_chat_prompt(tokenizer, user))
    return prompts


def _score_prompts(
    model: torch.nn.Module,
    tokenizer: Any,
    prompts: list[str],
    *,
    batch_size: int,
    max_length: int,
    label_ids: dict[str, int | str],
) -> tuple[np.ndarray, dict[str, float | int]]:
    lengths = np.fromiter((len(prompt) for prompt in prompts), dtype=np.int64, count=len(prompts))
    order = np.argsort(lengths, kind="stable")
    scores = np.empty(len(prompts), dtype=np.float32)
    device = torch.device("cuda")
    output_embeddings = model.get_output_embeddings()
    if output_embeddings is None or not hasattr(output_embeddings, "weight"):
        raise ValueError("teacher model has no accessible output embedding weights")
    token_ids = torch.tensor(
        [int(label_ids["negative"]), int(label_ids["positive"])],
        device=device,
        dtype=torch.long,
    )
    label_weights = output_embeddings.weight.index_select(0, token_ids).float()
    label_bias = None
    if getattr(output_embeddings, "bias", None) is not None:
        label_bias = output_embeddings.bias.index_select(0, token_ids).float()
    backbone = getattr(model, getattr(model, "base_model_prefix", "model"), None)
    if backbone is None or backbone is model:
        backbone = getattr(model, "model", None)
    if backbone is None:
        raise ValueError("unable to resolve teacher backbone")

    torch.cuda.reset_peak_memory_stats(device)
    started = time.perf_counter()
    with torch.inference_mode():
        for batch_index, offset in enumerate(range(0, len(order), batch_size), start=1):
            indices = order[offset : offset + batch_size]
            batch_prompts = [prompts[index] for index in indices]
            inputs = tokenizer(
                batch_prompts,
                padding=True,
                truncation=True,
                max_length=max_length,
                pad_to_multiple_of=8,
                return_tensors="pt",
            )
            inputs = {name: value.to(device, non_blocking=True) for name, value in inputs.items()}
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                hidden = backbone(**inputs, use_cache=False, return_dict=True).last_hidden_state
            last_hidden = hidden[:, -1, :].float()
            logits = torch.nn.functional.linear(last_hidden, label_weights, label_bias)
            batch_scores = (logits[:, 1] - logits[:, 0]).cpu().numpy()
            scores[indices] = batch_scores
            completed = min(offset + len(indices), len(order))
            if batch_index == 1 or completed == len(order) or batch_index % 25 == 0:
                print(f"teacher scored {completed:,}/{len(order):,}", flush=True)
    torch.cuda.synchronize(device)
    seconds = time.perf_counter() - started
    if not np.isfinite(scores).all():
        raise RuntimeError("teacher returned non-finite scores")
    return scores, {
        "rows": len(scores),
        "seconds": seconds,
        "rows_per_second": len(scores) / seconds,
        "peak_gpu_memory_bytes": int(torch.cuda.max_memory_allocated(device)),
        "peak_gpu_reserved_bytes": int(torch.cuda.max_memory_reserved(device)),
    }


def main() -> None:
    args = _parse_args()
    if not args.model.is_dir() or not args.input.is_file():
        raise FileNotFoundError(args.model if not args.model.is_dir() else args.input)
    if args.batch_size < 1 or args.max_length < 64 or args.max_attribute_characters < 1:
        raise ValueError("invalid teacher sizing arguments")
    manifest_path = args.output.with_suffix(".manifest.json")
    if not args.overwrite and (args.output.exists() or manifest_path.exists()):
        raise FileExistsError(args.output)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    if args.overwrite:
        args.output.unlink(missing_ok=True)
        manifest_path.unlink(missing_ok=True)
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA GPU is required")
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    torch.set_float32_matmul_precision("high")
    torch.backends.cuda.matmul.allow_tf32 = True

    frame = _load_rows(args)
    load_started = time.perf_counter()
    tokenizer = AutoTokenizer.from_pretrained(args.model, local_files_only=True)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"
    tokenizer.truncation_side = "right"
    label_ids = dict(choose_binary_token_ids(tokenizer))
    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        dtype=torch.bfloat16,
        attn_implementation="sdpa",
        local_files_only=True,
    )
    model.requires_grad_(False)
    model.eval()
    model.to("cuda")
    load_seconds = time.perf_counter() - load_started

    forward_prompts = _build_prompts(frame, tokenizer, args, reverse=False)
    forward, forward_stats = _score_prompts(
        model,
        tokenizer,
        forward_prompts,
        batch_size=args.batch_size,
        max_length=args.max_length,
        label_ids=label_ids,
    )
    directions = 1
    backward = None
    if args.bidirectional:
        backward_prompts = _build_prompts(frame, tokenizer, args, reverse=True)
        backward, backward_stats = _score_prompts(
            model,
            tokenizer,
            backward_prompts,
            batch_size=args.batch_size,
            max_length=args.max_length,
            label_ids=label_ids,
        )
        teacher_logit = (forward.astype(np.float64) + backward.astype(np.float64)) * 0.5
        directions = 2
        scoring_seconds = float(forward_stats["seconds"]) + float(backward_stats["seconds"])
        peak_allocated = max(
            int(forward_stats["peak_gpu_memory_bytes"]),
            int(backward_stats["peak_gpu_memory_bytes"]),
        )
        peak_reserved = max(
            int(forward_stats["peak_gpu_reserved_bytes"]),
            int(backward_stats["peak_gpu_reserved_bytes"]),
        )
    else:
        teacher_logit = forward.astype(np.float64)
        scoring_seconds = float(forward_stats["seconds"])
        peak_allocated = int(forward_stats["peak_gpu_memory_bytes"])
        peak_reserved = int(forward_stats["peak_gpu_reserved_bytes"])
    teacher_probability = 1.0 / (1.0 + np.exp(-np.clip(teacher_logit, -40.0, 40.0)))
    columns = [
        pl.Series("teacher_logit", teacher_logit.astype(np.float32)),
        pl.Series("teacher_probability", teacher_probability.astype(np.float32)),
        pl.Series("teacher_logit_forward", forward.astype(np.float32)),
    ]
    if backward is not None:
        columns.append(pl.Series("teacher_logit_backward", backward.astype(np.float32)))
    output = frame.with_columns(*columns).sort("teacher_source_row")
    temporary = args.output.with_suffix(".tmp.parquet")
    output.write_parquet(temporary, compression="zstd", statistics=True)
    os.replace(temporary, args.output)

    target = output["target"].to_numpy()
    categories = output["category"].to_numpy()
    binary = np.isin(target, (0.0, 1.0)).all()
    metric = None
    if binary:
        report = macro_average_precision(target, teacher_logit, categories)
        metric = {
            "scope": "selected categories only",
            "macro_average_precision": report.score,
            "per_category": report.per_category,
            "rows_per_category": report.rows_per_category,
            "positives_per_category": report.positives_per_category,
        }
    payload = {
        "schema_version": 1,
        "created_at_utc": datetime.now(UTC).isoformat(),
        "teacher": {
            "model": str(args.model.resolve()),
            "config_sha256": _sha256(args.model / "config.json"),
            "prompt_version": TEACHER_PROMPT_VERSION,
            "prompt_mode": args.prompt_mode,
            "binary_tokens": label_ids,
        },
        "data": {
            "source": str(args.input.resolve()),
            "source_sha256": _sha256(args.input),
            "rows": output.height,
            "categories": sorted(output["category"].unique().to_list()),
        },
        "config": {
            "batch_size": args.batch_size,
            "max_length": args.max_length,
            "max_attribute_characters": args.max_attribute_characters,
            "include_fold": args.include_fold,
            "exclude_fold": args.exclude_fold,
            "include_inner_fold": args.include_inner_fold,
            "exclude_inner_fold": args.exclude_inner_fold,
            "max_rows": args.max_rows,
            "shard_count": args.shard_count,
            "shard_index": args.shard_index,
            "bidirectional": args.bidirectional,
            "seed": args.seed,
        },
        "metric": metric,
        "runtime": {
            "load_seconds": load_seconds,
            "score_seconds": scoring_seconds,
            "rows_per_second": output.height / scoring_seconds,
            "forward_passes": directions,
            "peak_gpu_memory_bytes": peak_allocated,
            "peak_gpu_reserved_bytes": peak_reserved,
            "gpu": torch.cuda.get_device_name(0),
        },
        "artifact": {
            "path": str(args.output.resolve()),
            "bytes": args.output.stat().st_size,
            "sha256": _sha256(args.output),
        },
    }
    manifest_path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(manifest_path)


if __name__ == "__main__":
    main()
