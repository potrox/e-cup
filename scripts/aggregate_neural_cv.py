from __future__ import annotations

import argparse
import json
from dataclasses import asdict
from pathlib import Path
from typing import Any

import polars as pl

from ecup_matching.metrics import macro_average_precision


def _parse_folds(raw: str) -> tuple[int, ...]:
    folds = tuple(sorted({int(value) for value in raw.split(",")}))
    if not folds or any(fold not in range(5) for fold in folds):
        raise argparse.ArgumentTypeError("folds must be a subset of 0,1,2,3,4")
    return folds


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Aggregate paired neural cross-validation")
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument(
        "--base-oof",
        type=Path,
        default=Path("artifacts/category_baseline_v1/oof_predictions.parquet"),
    )
    parser.add_argument("--folds", type=_parse_folds, default=(0, 1, 2, 3, 4))
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def _fold_scores(frame: pl.DataFrame) -> dict[str, float]:
    scores: dict[str, float] = {}
    for fold in sorted(frame["fold"].unique().to_list()):
        selected = frame.filter(pl.col("fold") == fold)
        scores[str(fold)] = macro_average_precision(
            selected["target"].to_numpy(),
            selected["predict"].to_numpy(),
            selected["category"].to_numpy(),
        ).score
    return scores


def _report(frame: pl.DataFrame) -> dict[str, Any]:
    all_report = macro_average_precision(
        frame["target"].to_numpy(),
        frame["predict"].to_numpy(),
        frame["category"].to_numpy(),
    )
    folds_1_4 = frame.filter(pl.col("fold") != 0)
    uncontaminated = macro_average_precision(
        folds_1_4["target"].to_numpy(),
        folds_1_4["predict"].to_numpy(),
        folds_1_4["category"].to_numpy(),
    )
    return {
        "all_folds": asdict(all_report),
        "folds_1_4": asdict(uncontaminated),
        "per_fold": _fold_scores(frame),
    }


def _load_arm(root: Path, arm: str, folds: tuple[int, ...]) -> pl.DataFrame:
    frames: list[pl.DataFrame] = []
    for fold in folds:
        run_dir = root / arm / f"fold_{fold}"
        manifest_path = run_dir / "training_manifest.json"
        if not manifest_path.is_file():
            raise FileNotFoundError(manifest_path)
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        best_epoch = int(manifest["best_epoch"])
        prediction_path = run_dir / f"predictions_epoch_{best_epoch}.parquet"
        frame = pl.read_parquet(prediction_path).select(
            "id1", "id2", "target", "category", pl.col("score").alias("predict")
        )
        frames.append(frame.with_columns(pl.lit(fold, dtype=pl.Int8).alias("fold")))
    result = pl.concat(frames, how="vertical")
    if result.select(pl.struct("id1", "id2").n_unique()).item() != result.height:
        raise ValueError(f"{arm} contains duplicate pair keys")
    return result


def _validate_contract(frame: pl.DataFrame, truth: pl.DataFrame, arm: str) -> pl.DataFrame:
    scored = frame.rename(
        {"target": "target_score", "category": "category_score", "fold": "fold_score"}
    )
    joined = truth.join(scored, on=["id1", "id2"], how="left", validate="1:1")
    if joined["predict"].null_count():
        raise ValueError(f"{arm} does not score every requested pair")
    mismatch = joined.filter(
        (pl.col("target") != pl.col("target_score"))
        | (pl.col("category") != pl.col("category_score"))
        | (pl.col("fold") != pl.col("fold_score"))
    ).height
    if mismatch:
        raise ValueError(f"{arm} predictions violate the human OOF contract")
    return joined.select("id1", "id2", "target", "category", "fold", "predict")


def main() -> None:
    args = _parse_args()
    args.root.mkdir(parents=True, exist_ok=True)
    report_path = args.root / "paired_cv_report.json"
    if report_path.exists() and not args.overwrite:
        raise FileExistsError(report_path)

    truth = (
        pl.read_parquet(
            args.base_oof,
            columns=["id1", "id2", "target", "category", "fold"],
        )
        .filter(pl.col("fold").is_in(args.folds))
        .sort("id1", "id2")
    )
    arm_frames: dict[str, pl.DataFrame] = {}
    reports: dict[str, dict[str, Any]] = {}
    for arm in ("human_only", "llm_gold"):
        frame = _validate_contract(_load_arm(args.root, arm, args.folds), truth, arm)
        arm_frames[arm] = frame
        reports[arm] = _report(frame)
        frame.write_parquet(
            args.root / f"{arm}_oof.parquet",
            compression="zstd",
            statistics=True,
        )

    human = reports["human_only"]
    llm_gold = reports["llm_gold"]
    human_categories = human["all_folds"]["per_category"]
    llm_categories = llm_gold["all_folds"]["per_category"]
    per_category_delta = {
        category: float(llm_categories[category] - human_categories[category])
        for category in sorted(human_categories)
    }
    per_fold_delta = {
        fold: float(llm_gold["per_fold"][fold] - human["per_fold"][fold])
        for fold in human["per_fold"]
    }
    payload = {
        "schema_version": 1,
        "folds": list(args.folds),
        "rows": truth.height,
        "protocol": (
            "Paired 5-fold comparison. Both arms use identical architecture, seed, "
            "serialization, optimizer, and gold-stage settings. Only the LLM pre-stage differs."
        ),
        "arms": reports,
        "llm_gold_minus_human_only": {
            "all_folds": float(llm_gold["all_folds"]["score"] - human["all_folds"]["score"]),
            "folds_1_4": float(llm_gold["folds_1_4"]["score"] - human["folds_1_4"]["score"]),
            "per_fold": per_fold_delta,
            "positive_folds": int(sum(value > 0.0 for value in per_fold_delta.values())),
            "per_category": per_category_delta,
            "positive_categories": int(sum(value > 0.0 for value in per_category_delta.values())),
        },
    }
    report_path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(report_path)


if __name__ == "__main__":
    main()
