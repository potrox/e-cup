from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np
import polars as pl

from ecup_matching.inference_variant_routing import evaluate_cross_fitted_category_route
from ecup_matching.metrics import macro_average_precision
from ecup_matching.neural_ensemble import load_ensemble_oof

K16_METHOD = "rumodernbert_base+lamar_600m__raw_logit__top_16_category_router"


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate leakage-safe max-length routing on immutable K16"
    )
    parser.add_argument("--base-cv-root", type=Path, required=True)
    parser.add_argument("--lamar-oof", type=Path, required=True)
    parser.add_argument("--length-oof", type=Path, required=True)
    parser.add_argument("--k16-report", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--minimum-category-uplift", type=float, default=0.003)
    parser.add_argument("--minimum-training-positive-folds", type=int, default=3)
    parser.add_argument("--minimum-deployment-positive-folds", type=int, default=4)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _method(path: Path) -> dict[str, Any]:
    report = json.loads(path.read_text(encoding="utf-8"))
    method = report.get("methods", {}).get(K16_METHOD)
    if not isinstance(method, dict):
        raise ValueError(f"canonical K16 method is missing from {path}")
    if method.get("feature_order") != ["rumodernbert_base", "lamar_600m"]:
        raise ValueError("K16 feature order changed")
    return method


def _join_length(frame: pl.DataFrame, path: Path) -> pl.DataFrame:
    required = {"id1", "id2", "target", "category", "fold", "predict"}
    missing = sorted(required - set(pl.read_parquet_schema(path)))
    if missing:
        raise ValueError(f"length OOF is missing columns: {missing}")
    length = pl.read_parquet(path, columns=sorted(required)).rename(
        {
            "target": "length_target",
            "category": "length_category",
            "fold": "length_fold",
            "predict": "base_l384",
        }
    )
    joined = frame.join(length, on=["id1", "id2"], how="left", validate="1:1")
    if joined["base_l384"].null_count() or joined.height != frame.height:
        raise ValueError("length OOF does not exactly cover K16 rows")
    mismatches = joined.filter(
        (pl.col("target") != pl.col("length_target"))
        | (pl.col("category") != pl.col("length_category"))
        | (pl.col("fold") != pl.col("length_fold"))
    ).height
    if mismatches:
        raise ValueError(f"length OOF contract mismatches: {mismatches}")
    return joined.drop("length_target", "length_category", "length_fold")


def main() -> None:
    args = _parse_args()
    output_predictions = args.output_dir / "oof_predictions.parquet"
    output_report = args.output_dir / "evaluation.json"
    if not args.overwrite and (output_predictions.exists() or output_report.exists()):
        raise FileExistsError(args.output_dir)
    for path in (args.lamar_oof, args.length_oof, args.k16_report):
        if not path.is_file():
            raise FileNotFoundError(path)

    frame = load_ensemble_oof(args.base_cv_root, args.lamar_oof).rename(
        {"small_score": "lamar_score"}
    )
    frame = _join_length(frame, args.length_oof)
    method = _method(args.k16_report)
    weights = method.get("blend_weights") or [0.55, 0.45]
    if weights != [0.55, 0.45]:
        raise ValueError(f"unexpected K16 weights: {weights}")
    base = frame["base_score"].to_numpy()
    base_l384 = frame["base_l384"].to_numpy()
    lamar = frame["lamar_score"].to_numpy()
    categories = frame["category"].to_numpy()
    folds = frame["fold"].to_numpy()
    k16 = base.copy()
    k16_l384 = base_l384.copy()
    selections = method.get("cross_fitted_selections")
    if not isinstance(selections, list) or len(selections) != 5:
        raise ValueError("K16 must contain five cross-fitted selections")
    for record in selections:
        fold = int(record["validation_fold"])
        selected = tuple(str(value) for value in record["selected_categories"])
        mask = (folds == fold) & np.isin(categories, selected)
        k16[mask] = 0.55 * base[mask] + 0.45 * lamar[mask]
        k16_l384[mask] = 0.55 * base_l384[mask] + 0.45 * lamar[mask]
    target = frame["target"].to_numpy().astype(np.int8, copy=False)
    reconstructed = macro_average_precision(target, k16, categories).score
    canonical = float(method["local_oof"]["score"])
    if not np.isclose(reconstructed, canonical, rtol=0.0, atol=1e-12):
        raise ValueError(f"K16 reconstruction mismatch: {reconstructed} != {canonical}")

    evaluation_frame = frame.with_columns(
        pl.Series("k16", k16),
        pl.Series("k16_l384", k16_l384),
    )
    report, prediction, routed_mask = evaluate_cross_fitted_category_route(
        evaluation_frame,
        base_column="k16",
        candidate_column="k16_l384",
        minimum_category_uplift=args.minimum_category_uplift,
        minimum_training_positive_folds=args.minimum_training_positive_folds,
        minimum_deployment_positive_folds=args.minimum_deployment_positive_folds,
    )
    output = frame.select("id1", "id2", "target", "category", "fold").with_columns(
        pl.Series("predict", prediction),
        pl.Series("length_routed", routed_mask),
    )
    args.output_dir.mkdir(parents=True, exist_ok=True)
    output.write_parquet(output_predictions, compression="zstd", statistics=True)
    report["composition"] = {
        "reference": K16_METHOD,
        "candidate_variant": "K16 with RuModernBERT-base component scored at max_length=384",
        "k16_reconstructed_metric": reconstructed,
        "weights": [0.55, 0.45],
    }
    report["sources"] = {
        "lamar_oof": {"path": str(args.lamar_oof.resolve()), "sha256": _sha256(args.lamar_oof)},
        "length_oof": {"path": str(args.length_oof.resolve()), "sha256": _sha256(args.length_oof)},
        "k16_report": {"path": str(args.k16_report.resolve()), "sha256": _sha256(args.k16_report)},
    }
    report["output"] = {
        "path": str(output_predictions.resolve()),
        "sha256": _sha256(output_predictions),
    }
    output_report.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(output_report)


if __name__ == "__main__":
    main()
