from __future__ import annotations

import argparse
import json
from pathlib import Path

import polars as pl

from ecup_matching.release_ensemble_search import (
    BASE_MODEL,
    evaluate_release_ensemble_search,
    load_release_ensemble_oof,
)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate all bounded no-training release ensembles"
    )
    parser.add_argument("--base-cv-root", type=Path, required=True)
    parser.add_argument("--lamar-oof", type=Path, required=True)
    parser.add_argument("--small-oof", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--step", type=float, default=0.05)
    parser.add_argument("--minimum-uplift", type=float, default=0.002)
    parser.add_argument("--forecast-uncertainty", type=float, default=0.015)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--base-public", type=float, required=True)
    parser.add_argument("--lamar-public", type=float, required=True)
    parser.add_argument("--small-public", type=float, required=True)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    report_path = args.output_dir / "evaluation.json"
    if report_path.exists() and not args.overwrite:
        raise FileExistsError(report_path)

    model_names = (BASE_MODEL, "lamar_600m", "rumodernbert_small_v3")
    frame = load_release_ensemble_oof(
        base_cv_root=args.base_cv_root,
        model_oof_paths={
            "lamar_600m": args.lamar_oof,
            "rumodernbert_small_v3": args.small_oof,
        },
    )
    report, predictions = evaluate_release_ensemble_search(
        frame=frame,
        model_names=model_names,
        official_public_scores={
            BASE_MODEL: args.base_public,
            "lamar_600m": args.lamar_public,
            "rumodernbert_small_v3": args.small_public,
        },
        step=args.step,
        minimum_uplift=args.minimum_uplift,
        forecast_uncertainty=args.forecast_uncertainty,
        workers=args.workers,
    )
    args.output_dir.mkdir(parents=True, exist_ok=True)
    output_paths: dict[str, str] = {}
    for method_name in report["selection"]["quality_passed"]:
        path = args.output_dir / f"{method_name}_oof_predictions.parquet"
        frame.select("id1", "id2", "target", "category", "fold").with_columns(
            pl.Series("predict", predictions[method_name])
        ).write_parquet(path, compression="zstd", statistics=True)
        output_paths[method_name] = str(path.resolve())
    report["outputs"] = output_paths
    report_path.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(report_path)


if __name__ == "__main__":
    main()
