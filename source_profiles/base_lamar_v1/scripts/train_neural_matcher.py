from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import math
import random
import shutil
import time
from collections.abc import Mapping
from dataclasses import asdict
from datetime import UTC, datetime
from itertools import islice
from pathlib import Path
from typing import Any

import numpy as np
import torch
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    get_linear_schedule_with_warmup,
)

from ecup_matching.neural_training import (
    HardPairThreshold,
    category_row_counts,
    evaluate_model,
    inverse_category_weights,
    iter_pair_batches,
    parquet_rows,
    prefetch_batches,
    seed_everything,
    tokenize_pair_batch,
    training_steps,
    weighted_bce_loss,
)
from ecup_matching.neural_training import (
    hard_pair_thresholds as fit_hard_pair_thresholds,
)
from ecup_matching.serialization import ITEM_SERIALIZER_VERSION, PAIR_SERIALIZER_VERSION


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train a single-logit product matcher")
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--train", type=Path, required=True)
    parser.add_argument("--validation", type=Path, required=True)
    parser.add_argument("--validation-kind", choices=("human", "llm"), required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--eval-batch-size", type=int, default=256)
    parser.add_argument("--gradient-accumulation", type=int, default=1)
    parser.add_argument("--learning-rate", type=float, default=2e-5)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--warmup-ratio", type=float, default=0.05)
    parser.add_argument("--max-length", type=int, default=256)
    parser.add_argument("--max-attribute-characters", type=int, default=2048)
    parser.add_argument(
        "--serialization-mode",
        choices=("item_v1", "pair_v2"),
        default="item_v1",
    )
    parser.add_argument("--read-batch-size", type=int, default=4096)
    parser.add_argument("--pair-swap-probability", type=float)
    parser.add_argument(
        "--loss-mode",
        choices=("row_mean", "category_mean"),
        default="row_mean",
    )
    parser.add_argument("--confidence-gamma", type=float, default=0.0)
    parser.add_argument(
        "--hard-replay-fraction",
        type=float,
        choices=(0.0, 0.125, 0.25),
        default=0.0,
    )
    parser.add_argument("--hard-negative-target-max", type=float, default=0.2)
    parser.add_argument("--hard-positive-target-min", type=float, default=0.8)
    parser.add_argument("--hard-negative-quantile", type=float, default=0.75)
    parser.add_argument("--hard-positive-quantile", type=float, default=0.25)
    parser.add_argument("--stress-suite", action="store_true")
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--log-every", type=int, default=100)
    parser.add_argument("--max-train-rows", type=int)
    parser.add_argument("--max-validation-rows", type=int)
    parser.add_argument("--train-include-fold", type=int)
    parser.add_argument("--train-exclude-fold", type=int)
    parser.add_argument("--train-include-inner-fold", type=int)
    parser.add_argument("--train-exclude-inner-fold", type=int)
    parser.add_argument("--validation-include-fold", type=int)
    parser.add_argument("--validation-exclude-fold", type=int)
    parser.add_argument("--validation-include-inner-fold", type=int)
    parser.add_argument("--validation-exclude-inner-fold", type=int)
    parser.add_argument("--eval-bidirectional", action="store_true")
    parser.add_argument("--evaluate-before-training", action="store_true")
    parser.add_argument(
        "--checkpoint-selection",
        choices=("best_validation", "last_epoch"),
        default="best_validation",
    )
    parser.add_argument("--gradient-checkpointing", action="store_true")
    parser.add_argument("--no-prefetch", action="store_true")
    parser.add_argument("--checkpoint-every-optimizer-steps", type=int, default=0)
    parser.add_argument("--keep-resume-checkpoints", action="store_true")
    parser.add_argument("--resume-from-checkpoint", type=Path)
    parser.add_argument("--stop-after-optimizer-steps", type=int)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def _sha256(path: Path, chunk_size: int = 8 * 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


def _package_versions() -> dict[str, str]:
    packages = ("torch", "transformers", "tokenizers", "pyarrow", "numpy", "scikit-learn")
    return {name: importlib.metadata.version(name) for name in packages}


def _checkpoint_artifacts(path: Path) -> list[dict[str, Any]]:
    return [
        {
            "path": str(file.resolve()),
            "bytes": file.stat().st_size,
            "sha256": _sha256(file),
        }
        for file in sorted(path.iterdir())
        if file.is_file()
    ]


def _training_code_fingerprints() -> list[dict[str, Any]]:
    project_root = Path(__file__).resolve().parents[1]
    paths = (
        Path(__file__).resolve(),
        project_root / "src/ecup_matching/neural_training.py",
        project_root / "src/ecup_matching/serialization.py",
        project_root / "src/ecup_matching/metrics.py",
        project_root / "src/ecup_matching/llm_diagnostics.py",
    )
    return [
        {
            "path": path.relative_to(project_root).as_posix(),
            "bytes": path.stat().st_size,
            "sha256": _sha256(path),
        }
        for path in paths
    ]


def _validate_args(args: argparse.Namespace) -> None:
    if args.epochs < 1:
        raise ValueError("epochs must be positive")
    if args.batch_size < 1 or args.eval_batch_size < 1:
        raise ValueError("batch sizes must be positive")
    if args.gradient_accumulation < 1:
        raise ValueError("gradient_accumulation must be positive")
    if not 0.0 <= args.warmup_ratio < 1.0:
        raise ValueError("warmup_ratio must be in [0, 1)")
    if not 0.0 <= args.pair_swap_probability <= 1.0:
        raise ValueError("pair_swap_probability must be in [0, 1]")
    if args.serialization_mode == "pair_v2" and args.pair_swap_probability != 0.0:
        raise ValueError("pair_v2 is canonical; pair swap probability must be zero")
    if args.serialization_mode == "pair_v2" and args.eval_bidirectional:
        raise ValueError("pair_v2 is canonical; bidirectional evaluation is invalid")
    if args.confidence_gamma < 0.0 or not math.isfinite(args.confidence_gamma):
        raise ValueError("confidence_gamma must be finite and non-negative")
    if not 0.0 <= args.hard_negative_target_max < args.hard_positive_target_min <= 1.0:
        raise ValueError("hard target thresholds must satisfy 0 <= negative < positive <= 1")
    if not 0.0 < args.hard_negative_quantile < 1.0:
        raise ValueError("hard_negative_quantile must be in (0, 1)")
    if not 0.0 < args.hard_positive_quantile < 1.0:
        raise ValueError("hard_positive_quantile must be in (0, 1)")
    if args.stress_suite and args.validation_kind != "human":
        raise ValueError("stress suite is defined only for human gold validation")
    if args.max_length < 8 or args.max_attribute_characters < 1:
        raise ValueError("text limits are invalid")
    if args.checkpoint_every_optimizer_steps < 0:
        raise ValueError("checkpoint_every_optimizer_steps must be non-negative")
    if args.stop_after_optimizer_steps is not None and args.stop_after_optimizer_steps < 1:
        raise ValueError("stop_after_optimizer_steps must be positive")
    if args.resume_from_checkpoint is not None and args.overwrite:
        raise ValueError("resume_from_checkpoint and overwrite are mutually exclusive")
    if args.train_include_fold is not None and args.train_exclude_fold is not None:
        raise ValueError("train fold filters are mutually exclusive")
    if args.train_include_inner_fold is not None and args.train_exclude_inner_fold is not None:
        raise ValueError("train inner fold filters are mutually exclusive")
    if args.validation_include_fold is not None and args.validation_exclude_fold is not None:
        raise ValueError("validation fold filters are mutually exclusive")
    if (
        args.validation_include_inner_fold is not None
        and args.validation_exclude_inner_fold is not None
    ):
        raise ValueError("validation inner fold filters are mutually exclusive")
    for path in (args.model, args.train, args.validation):
        if not path.exists():
            raise FileNotFoundError(path)
    if args.resume_from_checkpoint is not None:
        if not args.resume_from_checkpoint.is_dir():
            raise FileNotFoundError(args.resume_from_checkpoint)
        if not (args.resume_from_checkpoint / "training_state.pt").is_file():
            raise FileNotFoundError(args.resume_from_checkpoint / "training_state.pt")


def _prepare_output(
    path: Path,
    overwrite: bool,
    resume_from_checkpoint: Path | None,
) -> None:
    if resume_from_checkpoint is not None:
        if not path.is_dir():
            raise FileNotFoundError(path)
        try:
            resume_from_checkpoint.resolve().relative_to(path.resolve())
        except ValueError as error:
            raise ValueError("resume checkpoint must be inside output_dir") from error
        return
    if path.exists():
        if not overwrite:
            raise FileExistsError(path)
        resolved = path.resolve()
        if resolved.parent == resolved or len(resolved.parts) < 3:
            raise ValueError(f"refusing to remove broad path: {resolved}")
        shutil.rmtree(resolved)
    path.mkdir(parents=True)


def _canonical_resume_config(args: argparse.Namespace) -> dict[str, Any]:
    operational_fields = {
        "checkpoint_every_optimizer_steps",
        "keep_resume_checkpoints",
        "log_every",
        "no_prefetch",
        "output_dir",
        "overwrite",
        "resume_from_checkpoint",
        "stop_after_optimizer_steps",
    }
    config: dict[str, Any] = {}
    for key, value in vars(args).items():
        if key in operational_fields:
            continue
        if isinstance(value, Path):
            config[key] = str(value.resolve())
        else:
            config[key] = value
    return config


def _capture_rng_state() -> dict[str, Any]:
    return {
        "python": random.getstate(),
        "numpy": np.random.get_state(),
        "torch_cpu": torch.get_rng_state(),
        "torch_cuda": torch.cuda.get_rng_state_all(),
    }


def _restore_rng_state(state: Mapping[str, Any]) -> None:
    random.setstate(state["python"])
    np.random.set_state(state["numpy"])
    torch.set_rng_state(state["torch_cpu"])
    torch.cuda.set_rng_state_all(state["torch_cuda"])


def _load_resume_state(
    args: argparse.Namespace,
    code_fingerprints: list[dict[str, Any]],
) -> dict[str, Any] | None:
    if args.resume_from_checkpoint is None:
        return None
    state_path = args.resume_from_checkpoint / "training_state.pt"
    state = torch.load(state_path, map_location="cpu", weights_only=False)
    if not isinstance(state, dict) or state.get("schema_version") != 1:
        raise ValueError("unsupported or corrupt training resume state")
    if state.get("run_config") != _canonical_resume_config(args):
        raise ValueError("resume run configuration does not match the checkpoint")
    if state.get("training_code") != code_fingerprints:
        raise ValueError("training code changed since the resume checkpoint was written")
    return state


def _save_resume_checkpoint(
    *,
    model: torch.nn.Module,
    tokenizer: Any,
    optimizer: torch.optim.Optimizer,
    scheduler: Any,
    args: argparse.Namespace,
    code_fingerprints: list[dict[str, Any]],
    original_model_source: Path,
    epoch: int,
    processed_batches: int,
    global_step: int,
    optimizer_step: int,
    running_loss: float,
    rows_seen: int,
    history: list[dict[str, Any]],
    best_metric: float,
    best_epoch: int,
    total_steps: int,
    warmup_steps: int,
    epoch_elapsed_seconds: float,
    run_elapsed_seconds: float,
    peak_gpu_memory_bytes: int,
    peak_gpu_reserved_bytes: int,
) -> Path:
    checkpoint = args.output_dir / f"resume_step_{optimizer_step:08d}"
    if checkpoint.exists():
        shutil.rmtree(checkpoint)
    checkpoint.mkdir(parents=True)
    model.save_pretrained(checkpoint, safe_serialization=True)
    tokenizer.save_pretrained(checkpoint)
    payload = {
        "schema_version": 1,
        "saved_at_utc": datetime.now(UTC).isoformat(),
        "run_config": _canonical_resume_config(args),
        "training_code": code_fingerprints,
        "original_model_source": str(original_model_source.resolve()),
        "epoch": epoch,
        "processed_batches": processed_batches,
        "global_step": global_step,
        "optimizer_step": optimizer_step,
        "running_loss": running_loss,
        "rows_seen": rows_seen,
        "history": history,
        "best_metric": best_metric,
        "best_epoch": best_epoch,
        "total_steps": total_steps,
        "warmup_steps": warmup_steps,
        "epoch_elapsed_seconds": epoch_elapsed_seconds,
        "run_elapsed_seconds": run_elapsed_seconds,
        "peak_gpu_memory_bytes": peak_gpu_memory_bytes,
        "peak_gpu_reserved_bytes": peak_gpu_reserved_bytes,
        "optimizer": optimizer.state_dict(),
        "scheduler": scheduler.state_dict(),
        "rng": _capture_rng_state(),
    }
    torch.save(payload, checkpoint / "training_state.pt")

    pointer_path = args.output_dir / "latest_resume_checkpoint.json"
    pointer_tmp = pointer_path.with_suffix(".tmp")
    pointer_tmp.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "checkpoint": str(checkpoint.resolve()),
                "optimizer_step": optimizer_step,
                "saved_at_utc": payload["saved_at_utc"],
            },
            ensure_ascii=False,
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    pointer_tmp.replace(pointer_path)

    if not args.keep_resume_checkpoints:
        for older in args.output_dir.glob("resume_step_*"):
            if older != checkpoint and older.is_dir():
                shutil.rmtree(older)
    return checkpoint


def _write_interrupted_manifest(
    args: argparse.Namespace,
    checkpoint: Path,
    optimizer_step: int,
) -> None:
    path = args.output_dir / "training_incomplete.json"
    path.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "status": "stopped_at_registered_optimizer_step",
                "optimizer_step": optimizer_step,
                "resume_checkpoint": str(checkpoint.resolve()),
            },
            ensure_ascii=False,
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )


def _evaluate(
    *,
    model: torch.nn.Module,
    tokenizer: Any,
    args: argparse.Namespace,
    device: torch.device,
    name: str,
    stress_thresholds: Mapping[str, HardPairThreshold] | None,
) -> dict[str, Any]:
    prediction_path = args.output_dir / f"predictions_{name}.parquet"
    result = evaluate_model(
        model,
        tokenizer,
        args.validation,
        validation_kind=args.validation_kind,
        device=device,
        batch_size=args.eval_batch_size,
        read_batch_size=args.read_batch_size,
        max_length=args.max_length,
        max_attribute_characters=args.max_attribute_characters,
        seed=args.seed,
        bidirectional=args.eval_bidirectional,
        serialization_mode=args.serialization_mode,
        stress_thresholds=stress_thresholds,
        prediction_path=prediction_path,
        max_rows=args.max_validation_rows,
        include_fold=args.validation_include_fold,
        exclude_fold=args.validation_exclude_fold,
        include_inner_fold=args.validation_include_inner_fold,
        exclude_inner_fold=args.validation_exclude_inner_fold,
    )
    payload = {
        "selection_metric": result.selection_metric,
        "metrics": result.metrics,
        "prediction_path": str(prediction_path.resolve()),
        "rows": result.rows,
        "seconds": result.seconds,
        "rows_per_second": result.rows_per_second,
        "peak_gpu_memory_bytes": result.peak_gpu_memory_bytes,
    }
    print(
        f"evaluation={name} metric={result.selection_metric:.9f} rows={result.rows} "
        f"seconds={result.seconds:.2f} rows_per_second={result.rows_per_second:.1f}",
        flush=True,
    )
    return payload


def main() -> None:
    args = _parse_args()
    if args.pair_swap_probability is None:
        args.pair_swap_probability = 0.0 if args.serialization_mode == "pair_v2" else 0.5
    _validate_args(args)
    _prepare_output(args.output_dir, args.overwrite, args.resume_from_checkpoint)
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA GPU is required for this training pipeline")

    seed_everything(args.seed)
    code_fingerprints = _training_code_fingerprints()
    resume_state = _load_resume_state(args, code_fingerprints)
    serialization_version = (
        PAIR_SERIALIZER_VERSION if args.serialization_mode == "pair_v2" else ITEM_SERIALIZER_VERSION
    )
    torch.set_float32_matmul_precision("high")
    torch.backends.cuda.matmul.allow_tf32 = True
    device = torch.device("cuda")
    gpu_name = torch.cuda.get_device_name(device)

    model_load_path = args.resume_from_checkpoint or args.model
    tokenizer = AutoTokenizer.from_pretrained(model_load_path, local_files_only=True)
    model = AutoModelForSequenceClassification.from_pretrained(
        model_load_path,
        num_labels=1,
        ignore_mismatched_sizes=True,
        attn_implementation="sdpa",
        local_files_only=True,
    )
    model.config.compile_model = False
    model.config.problem_type = "regression"
    if args.gradient_checkpointing:
        model.gradient_checkpointing_enable()
    model.to(device)
    original_model_source = (
        Path(resume_state["original_model_source"]) if resume_state is not None else args.model
    )
    if resume_state is None:
        # Loading a base checkpoint initializes a new classifier. Reset afterward so paired
        # fresh runs see identical augmentation and dropout RNG.
        seed_everything(args.seed)

    available_rows = parquet_rows(
        args.train,
        include_fold=args.train_include_fold,
        exclude_fold=args.train_exclude_fold,
        include_inner_fold=args.train_include_inner_fold,
        exclude_inner_fold=args.train_exclude_inner_fold,
    )
    train_rows = min(available_rows, args.max_train_rows or available_rows)
    total_steps = training_steps(
        train_rows,
        args.batch_size,
        args.gradient_accumulation,
        args.epochs,
    )
    warmup_steps = int(total_steps * args.warmup_ratio)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.learning_rate,
        weight_decay=args.weight_decay,
        fused=True,
    )
    scheduler = get_linear_schedule_with_warmup(optimizer, warmup_steps, total_steps)
    if resume_state is not None:
        if (
            resume_state["total_steps"] != total_steps
            or resume_state["warmup_steps"] != warmup_steps
        ):
            raise ValueError("resume scheduler contract does not match the current run")
        optimizer.load_state_dict(resume_state["optimizer"])
        scheduler.load_state_dict(resume_state["scheduler"])
        _restore_rng_state(resume_state["rng"])
    training_category_counts = category_row_counts(
        args.train,
        include_fold=args.train_include_fold,
        exclude_fold=args.train_exclude_fold,
        include_inner_fold=args.train_include_inner_fold,
        exclude_inner_fold=args.train_exclude_inner_fold,
    )
    category_weights = None
    if args.loss_mode == "category_mean":
        category_weights = inverse_category_weights(training_category_counts)
    training_hard_pair_thresholds = None
    if args.hard_replay_fraction or args.stress_suite:
        training_hard_pair_thresholds = fit_hard_pair_thresholds(
            args.train,
            include_fold=args.train_include_fold,
            exclude_fold=args.train_exclude_fold,
            include_inner_fold=args.train_include_inner_fold,
            exclude_inner_fold=args.train_exclude_inner_fold,
            negative_target_max=args.hard_negative_target_max,
            positive_target_min=args.hard_positive_target_min,
            negative_similarity_quantile=args.hard_negative_quantile,
            positive_similarity_quantile=args.hard_positive_quantile,
        )

    resumed_run_elapsed_seconds = (
        float(resume_state["run_elapsed_seconds"]) if resume_state is not None else 0.0
    )
    run_started = time.perf_counter()
    history: list[dict[str, Any]] = (
        list(resume_state["history"]) if resume_state is not None else []
    )
    if args.evaluate_before_training and resume_state is None:
        history.append(
            {
                "name": "before_training",
                **_evaluate(
                    model=model,
                    tokenizer=tokenizer,
                    args=args,
                    device=device,
                    name="before_training",
                    stress_thresholds=training_hard_pair_thresholds,
                ),
            }
        )

    best_metric = float(resume_state["best_metric"]) if resume_state is not None else -math.inf
    best_epoch = int(resume_state["best_epoch"]) if resume_state is not None else 0
    global_step = int(resume_state["global_step"]) if resume_state is not None else 0
    optimizer_step = int(resume_state["optimizer_step"]) if resume_state is not None else 0
    start_epoch = int(resume_state["epoch"]) if resume_state is not None else 0
    resume_processed_batches = (
        int(resume_state["processed_batches"]) if resume_state is not None else 0
    )
    resume_running_loss = float(resume_state["running_loss"]) if resume_state is not None else 0.0
    resume_rows_seen = int(resume_state["rows_seen"]) if resume_state is not None else 0
    resume_epoch_elapsed_seconds = (
        float(resume_state["epoch_elapsed_seconds"]) if resume_state is not None else 0.0
    )
    if not 0 <= start_epoch < args.epochs:
        raise ValueError("resume epoch is outside the configured training range")
    run_peak_gpu_memory_bytes = (
        int(resume_state["peak_gpu_memory_bytes"]) if resume_state is not None else 0
    )
    run_peak_gpu_reserved_bytes = (
        int(resume_state["peak_gpu_reserved_bytes"]) if resume_state is not None else 0
    )
    torch.cuda.reset_peak_memory_stats(device)

    latest_resume_checkpoint: Path | None = args.resume_from_checkpoint
    for epoch in range(start_epoch, args.epochs):
        model.train()
        epoch_started = time.perf_counter()
        resuming_epoch = resume_state is not None and epoch == start_epoch
        epoch_elapsed_offset = resume_epoch_elapsed_seconds if resuming_epoch else 0.0
        processed_batches = resume_processed_batches if resuming_epoch else 0
        running_loss = resume_running_loss if resuming_epoch else 0.0
        rows_seen = resume_rows_seen if resuming_epoch else 0
        optimizer.zero_grad(set_to_none=True)
        batches = iter_pair_batches(
            args.train,
            batch_size=args.batch_size,
            read_batch_size=args.read_batch_size,
            max_attribute_characters=args.max_attribute_characters,
            seed=args.seed,
            epoch=epoch,
            shuffle=True,
            pair_swap_probability=args.pair_swap_probability,
            serialization_mode=args.serialization_mode,
            hard_replay_fraction=args.hard_replay_fraction,
            hard_pair_thresholds=training_hard_pair_thresholds,
            hard_negative_target_max=args.hard_negative_target_max,
            hard_positive_target_min=args.hard_positive_target_min,
            max_rows=args.max_train_rows,
            include_fold=args.train_include_fold,
            exclude_fold=args.train_exclude_fold,
            include_inner_fold=args.train_include_inner_fold,
            exclude_inner_fold=args.train_exclude_inner_fold,
        )
        if processed_batches:
            batches = islice(batches, processed_batches, None)
        if not args.no_prefetch:
            batches = prefetch_batches(batches)
        for batch_index, batch in enumerate(
            batches,
            start=processed_batches + 1,
        ):
            inputs = tokenize_pair_batch(
                tokenizer,
                batch,
                max_length=args.max_length,
                device=device,
            )
            labels = torch.from_numpy(batch.target).to(device, non_blocking=True)
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                logits = model(**inputs).logits.reshape(-1).float()
                loss = (
                    weighted_bce_loss(
                        logits,
                        labels,
                        categories=batch.categories if category_weights is not None else None,
                        category_weights=category_weights,
                        confidence_gamma=args.confidence_gamma,
                    )
                    / args.gradient_accumulation
                )
            loss.backward()

            global_step += 1
            rows_seen += len(batch)
            running_loss += float(loss.detach()) * args.gradient_accumulation
            did_optimizer_step = batch_index % args.gradient_accumulation == 0
            if did_optimizer_step:
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad(set_to_none=True)
                optimizer_step += 1

            should_checkpoint = (
                did_optimizer_step
                and args.checkpoint_every_optimizer_steps > 0
                and optimizer_step % args.checkpoint_every_optimizer_steps == 0
            )
            should_stop = (
                did_optimizer_step
                and args.stop_after_optimizer_steps is not None
                and optimizer_step >= args.stop_after_optimizer_steps
            )
            if should_checkpoint or should_stop:
                latest_resume_checkpoint = _save_resume_checkpoint(
                    model=model,
                    tokenizer=tokenizer,
                    optimizer=optimizer,
                    scheduler=scheduler,
                    args=args,
                    code_fingerprints=code_fingerprints,
                    original_model_source=original_model_source,
                    epoch=epoch,
                    processed_batches=batch_index,
                    global_step=global_step,
                    optimizer_step=optimizer_step,
                    running_loss=running_loss,
                    rows_seen=rows_seen,
                    history=history,
                    best_metric=best_metric,
                    best_epoch=best_epoch,
                    total_steps=total_steps,
                    warmup_steps=warmup_steps,
                    epoch_elapsed_seconds=(
                        epoch_elapsed_offset + time.perf_counter() - epoch_started
                    ),
                    run_elapsed_seconds=(
                        resumed_run_elapsed_seconds + time.perf_counter() - run_started
                    ),
                    peak_gpu_memory_bytes=max(
                        run_peak_gpu_memory_bytes,
                        torch.cuda.max_memory_allocated(device),
                    ),
                    peak_gpu_reserved_bytes=max(
                        run_peak_gpu_reserved_bytes,
                        torch.cuda.max_memory_reserved(device),
                    ),
                )
            if should_stop:
                if latest_resume_checkpoint is None:
                    raise RuntimeError("registered stop did not create a resume checkpoint")
                _write_interrupted_manifest(args, latest_resume_checkpoint, optimizer_step)
                print(args.output_dir / "training_incomplete.json", flush=True)
                return

            if batch_index % args.log_every == 0:
                elapsed = epoch_elapsed_offset + time.perf_counter() - epoch_started
                print(
                    f"epoch={epoch + 1} batch={batch_index} rows={rows_seen}/{train_rows} "
                    f"loss={running_loss / batch_index:.6f} "
                    f"rows_per_second={rows_seen / elapsed:.1f} "
                    f"lr={scheduler.get_last_lr()[0]:.3e}",
                    flush=True,
                )

        if batch_index % args.gradient_accumulation != 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad(set_to_none=True)
            optimizer_step += 1

        training_seconds = epoch_elapsed_offset + time.perf_counter() - epoch_started
        training_peak_gpu_memory_bytes = torch.cuda.max_memory_allocated(device)
        training_peak_gpu_reserved_bytes = torch.cuda.max_memory_reserved(device)
        run_peak_gpu_memory_bytes = max(
            run_peak_gpu_memory_bytes,
            training_peak_gpu_memory_bytes,
        )
        run_peak_gpu_reserved_bytes = max(
            run_peak_gpu_reserved_bytes,
            training_peak_gpu_reserved_bytes,
        )
        evaluation = _evaluate(
            model=model,
            tokenizer=tokenizer,
            args=args,
            device=device,
            name=f"epoch_{epoch + 1}",
            stress_thresholds=training_hard_pair_thresholds,
        )
        epoch_dir = args.output_dir / f"epoch_{epoch + 1}"
        model.save_pretrained(epoch_dir, safe_serialization=True)
        tokenizer.save_pretrained(epoch_dir)
        epoch_record = {
            "epoch": epoch + 1,
            "training_loss": running_loss / batch_index,
            "training_rows": rows_seen,
            "training_seconds": training_seconds,
            "training_rows_per_second": rows_seen / training_seconds,
            "training_peak_gpu_memory_bytes": training_peak_gpu_memory_bytes,
            "training_peak_gpu_reserved_bytes": training_peak_gpu_reserved_bytes,
            "optimizer_steps": optimizer_step,
            "evaluation": evaluation,
            "checkpoint": str(epoch_dir.resolve()),
        }
        history.append(epoch_record)
        should_select = args.checkpoint_selection == "last_epoch" or (
            evaluation["selection_metric"] > best_metric
        )
        if should_select:
            best_metric = float(evaluation["selection_metric"])
            best_epoch = epoch + 1
            best_dir = args.output_dir / "best"
            if best_dir.exists():
                shutil.rmtree(best_dir)
            shutil.copytree(epoch_dir, best_dir)

        if args.checkpoint_every_optimizer_steps > 0 and epoch + 1 < args.epochs:
            latest_resume_checkpoint = _save_resume_checkpoint(
                model=model,
                tokenizer=tokenizer,
                optimizer=optimizer,
                scheduler=scheduler,
                args=args,
                code_fingerprints=code_fingerprints,
                original_model_source=original_model_source,
                epoch=epoch + 1,
                processed_batches=0,
                global_step=global_step,
                optimizer_step=optimizer_step,
                running_loss=0.0,
                rows_seen=0,
                history=history,
                best_metric=best_metric,
                best_epoch=best_epoch,
                total_steps=total_steps,
                warmup_steps=warmup_steps,
                epoch_elapsed_seconds=0.0,
                run_elapsed_seconds=(
                    resumed_run_elapsed_seconds + time.perf_counter() - run_started
                ),
                peak_gpu_memory_bytes=max(
                    run_peak_gpu_memory_bytes,
                    torch.cuda.max_memory_allocated(device),
                ),
                peak_gpu_reserved_bytes=max(
                    run_peak_gpu_reserved_bytes,
                    torch.cuda.max_memory_reserved(device),
                ),
            )

    incomplete_manifest = args.output_dir / "training_incomplete.json"
    if incomplete_manifest.exists():
        incomplete_manifest.unlink()

    manifest = {
        "schema_version": 1,
        "config": {
            key: str(value) if isinstance(value, Path) else value
            for key, value in vars(args).items()
        },
        "model_source": str(original_model_source.resolve()),
        "serialization_version": serialization_version,
        "training_code": code_fingerprints,
        "train": {
            "path": str(args.train.resolve()),
            "sha256": _sha256(args.train),
            "available_rows": available_rows,
            "used_rows": train_rows,
            "category_counts": training_category_counts,
            "category_weights": category_weights,
            "hard_pair_thresholds": (
                {
                    category: asdict(threshold)
                    for category, threshold in training_hard_pair_thresholds.items()
                }
                if training_hard_pair_thresholds is not None
                else None
            ),
        },
        "validation": {
            "path": str(args.validation.resolve()),
            "sha256": _sha256(args.validation),
        },
        "best_epoch": best_epoch,
        "best_selection_metric": best_metric,
        "best_checkpoint": str((args.output_dir / "best").resolve()),
        "best_checkpoint_artifacts": _checkpoint_artifacts(args.output_dir / "best"),
        "history": history,
        "runtime": {
            "total_seconds": (resumed_run_elapsed_seconds + time.perf_counter() - run_started),
            "gpu": gpu_name,
            "peak_gpu_memory_bytes": max(
                run_peak_gpu_memory_bytes,
                torch.cuda.max_memory_allocated(device),
            ),
            "peak_gpu_reserved_bytes": max(
                run_peak_gpu_reserved_bytes,
                torch.cuda.max_memory_reserved(device),
            ),
            "packages": _package_versions(),
        },
        "resume": {
            "resumed_from": (
                str(args.resume_from_checkpoint.resolve())
                if args.resume_from_checkpoint is not None
                else None
            ),
            "latest_resumable_checkpoint": (
                str(latest_resume_checkpoint.resolve())
                if latest_resume_checkpoint is not None
                else None
            ),
            "checkpoint_every_optimizer_steps": args.checkpoint_every_optimizer_steps,
        },
    }
    manifest_path = args.output_dir / "training_manifest.json"
    manifest_path.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(manifest_path, flush=True)


if __name__ == "__main__":
    main()
