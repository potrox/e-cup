from __future__ import annotations

import argparse
import json
from dataclasses import asdict
from pathlib import Path

import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer

from ecup_matching.neural_training import (
    evaluate_model,
    hard_pair_thresholds,
    seed_everything,
)
from ecup_matching.serialization import ITEM_SERIALIZER_VERSION, PAIR_SERIALIZER_VERSION


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate a trained neural matcher")
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--validation", type=Path, required=True)
    parser.add_argument("--validation-kind", choices=("human", "llm"), required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--read-batch-size", type=int, default=4096)
    parser.add_argument("--max-length", type=int, default=256)
    parser.add_argument("--max-attribute-characters", type=int, default=2048)
    parser.add_argument(
        "--serialization-mode",
        choices=("item_v1", "pair_v2"),
        default="item_v1",
    )
    parser.add_argument("--include-fold", type=int)
    parser.add_argument("--exclude-fold", type=int)
    parser.add_argument("--include-inner-fold", type=int)
    parser.add_argument("--exclude-inner-fold", type=int)
    parser.add_argument("--max-rows", type=int)
    parser.add_argument("--bidirectional", action="store_true")
    parser.add_argument("--stress-reference", type=Path)
    parser.add_argument("--stress-reference-include-fold", type=int)
    parser.add_argument("--stress-reference-exclude-fold", type=int)
    parser.add_argument("--stress-reference-include-inner-fold", type=int)
    parser.add_argument("--stress-reference-exclude-inner-fold", type=int)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    if args.include_fold is not None and args.exclude_fold is not None:
        raise ValueError("fold filters are mutually exclusive")
    if args.include_inner_fold is not None and args.exclude_inner_fold is not None:
        raise ValueError("inner fold filters are mutually exclusive")
    if args.serialization_mode == "pair_v2" and args.bidirectional:
        raise ValueError("pair_v2 is canonical; bidirectional scoring is invalid")
    if (
        args.stress_reference_include_fold is not None
        and args.stress_reference_exclude_fold is not None
    ):
        raise ValueError("stress reference fold filters are mutually exclusive")
    if (
        args.stress_reference_include_inner_fold is not None
        and args.stress_reference_exclude_inner_fold is not None
    ):
        raise ValueError("stress reference inner fold filters are mutually exclusive")
    if args.stress_reference is not None and args.validation_kind != "human":
        raise ValueError("stress suite is defined only for human gold validation")
    for path in (args.model, args.validation):
        if not path.exists():
            raise FileNotFoundError(path)
    report_path = args.output_dir / "evaluation.json"
    prediction_path = args.output_dir / "predictions.parquet"
    if not args.overwrite and (report_path.exists() or prediction_path.exists()):
        raise FileExistsError(args.output_dir)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA GPU is required")

    seed_everything(args.seed)
    torch.set_float32_matmul_precision("high")
    torch.backends.cuda.matmul.allow_tf32 = True
    device = torch.device("cuda")
    tokenizer = AutoTokenizer.from_pretrained(args.model, local_files_only=True)
    model = AutoModelForSequenceClassification.from_pretrained(
        args.model,
        num_labels=1,
        attn_implementation="sdpa",
        local_files_only=True,
    )
    model.config.compile_model = False
    model.to(device)
    stress_thresholds = None
    if args.stress_reference is not None:
        stress_thresholds = hard_pair_thresholds(
            args.stress_reference,
            include_fold=args.stress_reference_include_fold,
            exclude_fold=args.stress_reference_exclude_fold,
            include_inner_fold=args.stress_reference_include_inner_fold,
            exclude_inner_fold=args.stress_reference_exclude_inner_fold,
        )

    result = evaluate_model(
        model,
        tokenizer,
        args.validation,
        validation_kind=args.validation_kind,
        device=device,
        batch_size=args.batch_size,
        read_batch_size=args.read_batch_size,
        max_length=args.max_length,
        max_attribute_characters=args.max_attribute_characters,
        seed=args.seed,
        bidirectional=args.bidirectional,
        serialization_mode=args.serialization_mode,
        stress_thresholds=stress_thresholds,
        prediction_path=prediction_path,
        max_rows=args.max_rows,
        include_fold=args.include_fold,
        exclude_fold=args.exclude_fold,
        include_inner_fold=args.include_inner_fold,
        exclude_inner_fold=args.exclude_inner_fold,
    )
    payload = {
        "schema_version": 1,
        "model": str(args.model.resolve()),
        "validation": str(args.validation.resolve()),
        "validation_kind": args.validation_kind,
        "serialization_version": (
            PAIR_SERIALIZER_VERSION
            if args.serialization_mode == "pair_v2"
            else ITEM_SERIALIZER_VERSION
        ),
        "config": {
            "batch_size": args.batch_size,
            "read_batch_size": args.read_batch_size,
            "max_length": args.max_length,
            "max_attribute_characters": args.max_attribute_characters,
            "serialization_mode": args.serialization_mode,
            "include_fold": args.include_fold,
            "exclude_fold": args.exclude_fold,
            "include_inner_fold": args.include_inner_fold,
            "exclude_inner_fold": args.exclude_inner_fold,
            "max_rows": args.max_rows,
            "bidirectional": args.bidirectional,
            "seed": args.seed,
        },
        "stress_thresholds": (
            {category: asdict(threshold) for category, threshold in stress_thresholds.items()}
            if stress_thresholds is not None
            else None
        ),
        "selection_metric": result.selection_metric,
        "metrics": result.metrics,
        "rows": result.rows,
        "seconds": result.seconds,
        "rows_per_second": result.rows_per_second,
        "peak_gpu_memory_bytes": result.peak_gpu_memory_bytes,
        "predictions": str(prediction_path.resolve()),
    }
    report_path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(report_path)


if __name__ == "__main__":
    main()
