from __future__ import annotations

import argparse
import json
from pathlib import Path

import polars as pl

from ecup_matching.neural_ensemble import evaluate_neural_ensemble, load_ensemble_oof


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate a cross-fitted base/small ensemble")
    parser.add_argument("--base-root", type=Path, required=True)
    parser.add_argument("--small-oof", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--step", type=float, default=0.05)
    parser.add_argument("--minimum-uplift", type=float, default=0.002)
    parser.add_argument("--public-correction", type=float, default=0.279)
    parser.add_argument("--forecast-uncertainty", type=float, default=0.02)
    parser.add_argument("--candidate-name", default="small_v3")
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    report_path = args.output_dir / "evaluation.json"
    prediction_paths = {
        name: args.output_dir / f"{name}_oof_predictions.parquet"
        for name in ("raw_logit", "category_rank")
    }
    if not args.overwrite:
        existing = [path for path in (report_path, *prediction_paths.values()) if path.exists()]
        if existing:
            raise FileExistsError(existing[0])
    frame = load_ensemble_oof(args.base_root, args.small_oof)
    report, predictions = evaluate_neural_ensemble(
        frame=frame,
        step=args.step,
        minimum_uplift=args.minimum_uplift,
        public_correction=args.public_correction,
        forecast_uncertainty=args.forecast_uncertainty,
        candidate_name=args.candidate_name,
    )
    args.output_dir.mkdir(parents=True, exist_ok=True)
    for name, prediction in predictions.items():
        frame.select("id1", "id2", "target", "category", "fold").with_columns(
            pl.Series("predict", prediction)
        ).write_parquet(prediction_paths[name], compression="zstd", statistics=True)
    report["outputs"] = {
        name: str(path.resolve()) for name, path in prediction_paths.items()
    }
    report_path.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(report_path)


if __name__ == "__main__":
    main()
