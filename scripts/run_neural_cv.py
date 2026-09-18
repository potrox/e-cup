from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path


def _parse_folds(raw: str) -> tuple[int, ...]:
    folds = tuple(dict.fromkeys(int(value) for value in raw.split(",")))
    if not folds or any(fold not in range(5) for fold in folds):
        raise argparse.ArgumentTypeError("folds must be a non-empty subset of 0,1,2,3,4")
    return folds


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run one frozen neural recipe across outer folds")
    model_group = parser.add_mutually_exclusive_group(required=True)
    model_group.add_argument("--model", type=Path)
    model_group.add_argument(
        "--model-root",
        type=Path,
        help="Fold checkpoint root containing fold_N/best for fold-matched continuation",
    )
    parser.add_argument("--human-data", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--folds", type=_parse_folds, default=(0, 1, 2, 3, 4))
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--eval-batch-size", type=int, default=256)
    parser.add_argument("--learning-rate", type=float, default=1e-5)
    parser.add_argument("--max-length", type=int, choices=(256, 384), default=256)
    parser.add_argument("--max-attribute-characters", type=int, default=2048)
    parser.add_argument(
        "--serialization-mode",
        choices=("item_v1", "pair_v2"),
        default="pair_v2",
    )
    parser.add_argument(
        "--loss-mode",
        choices=("row_mean", "category_mean"),
        default="row_mean",
    )
    parser.add_argument("--confidence-gamma", type=float, choices=(0.0, 1.0, 2.0), default=0.0)
    parser.add_argument(
        "--hard-replay-fraction",
        type=float,
        choices=(0.0, 0.125, 0.25),
        default=0.0,
    )
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument(
        "--checkpoint-selection",
        choices=("best_validation", "last_epoch"),
        default="last_epoch",
    )
    parser.add_argument(
        "--eval-bidirectional",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument("--stress-suite", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--gradient-checkpointing", action="store_true")
    parser.add_argument("--no-prefetch", action="store_true")
    parser.add_argument("--resume", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    for path in (args.model or args.model_root, args.human_data):
        if not path.exists():
            raise FileNotFoundError(path)
    pair_swap_probability = "0" if args.serialization_mode == "pair_v2" else "0.5"

    train_script = Path(__file__).with_name("train_neural_matcher.py")
    for fold in args.folds:
        model = args.model or (args.model_root / f"fold_{fold}" / "best")
        if not model.is_dir():
            raise FileNotFoundError(model)
        output_dir = args.output_root / f"fold_{fold}"
        manifest = output_dir / "training_manifest.json"
        if manifest.is_file() and args.resume:
            print(f"skip completed fold={fold}", flush=True)
            continue
        command = [
            sys.executable,
            str(train_script),
            "--model",
            str(model),
            "--train",
            str(args.human_data),
            "--validation",
            str(args.human_data),
            "--validation-kind",
            "human",
            "--output-dir",
            str(output_dir),
            "--epochs",
            str(args.epochs),
            "--batch-size",
            str(args.batch_size),
            "--eval-batch-size",
            str(args.eval_batch_size),
            "--learning-rate",
            str(args.learning_rate),
            "--max-length",
            str(args.max_length),
            "--max-attribute-characters",
            str(args.max_attribute_characters),
            "--serialization-mode",
            args.serialization_mode,
            "--pair-swap-probability",
            pair_swap_probability,
            "--loss-mode",
            args.loss_mode,
            "--confidence-gamma",
            str(args.confidence_gamma),
            "--hard-replay-fraction",
            str(args.hard_replay_fraction),
            "--seed",
            str(args.seed),
            "--checkpoint-selection",
            args.checkpoint_selection,
            "--train-exclude-fold",
            str(fold),
            "--validation-include-fold",
            str(fold),
            "--log-every",
            "250",
        ]
        if args.stress_suite:
            command.append("--stress-suite")
        if args.eval_bidirectional:
            command.append("--eval-bidirectional")
        if args.gradient_checkpointing:
            command.append("--gradient-checkpointing")
        if args.no_prefetch:
            command.append("--no-prefetch")
        print(f"start fold={fold}", flush=True)
        subprocess.run(command, check=True)


if __name__ == "__main__":
    main()
