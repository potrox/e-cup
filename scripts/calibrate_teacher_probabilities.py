from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import polars as pl
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import brier_score_loss, log_loss


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Fit leakage-safe teacher probability calibration and apply it"
    )
    parser.add_argument("--fit", type=Path, required=True)
    parser.add_argument("--validation", type=Path, required=True)
    parser.add_argument("--apply", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--minimum-class-rows", type=int, default=10)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def _sha256(path: Path, chunk_size: int = 8 * 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while block := handle.read(chunk_size):
            digest.update(block)
    return digest.hexdigest()


def _validate(frame: pl.DataFrame, name: str) -> None:
    required = {"target", "category", "teacher_logit", "teacher_probability"}
    missing = required - set(frame.columns)
    if missing:
        raise ValueError(f"{name} is missing columns: {sorted(missing)}")
    target = frame["target"].to_numpy()
    if not np.isin(target, (0.0, 1.0)).all():
        raise ValueError(f"{name} targets must be binary human labels")
    if not np.isfinite(frame["teacher_logit"].to_numpy()).all():
        raise ValueError(f"{name} teacher logits must be finite")


def _fit_logistic(logit: np.ndarray, target: np.ndarray) -> tuple[float, float]:
    estimator = LogisticRegression(C=1.0, solver="lbfgs", max_iter=1_000)
    estimator.fit(logit.reshape(-1, 1), target)
    coefficient = float(estimator.coef_[0, 0])
    intercept = float(estimator.intercept_[0])
    if not math.isfinite(coefficient) or not math.isfinite(intercept):
        raise RuntimeError("teacher calibration returned non-finite parameters")
    return coefficient, intercept


def _probability(logit: np.ndarray, coefficient: float, intercept: float) -> np.ndarray:
    calibrated_logit = np.clip(coefficient * logit + intercept, -40.0, 40.0)
    return 1.0 / (1.0 + np.exp(-calibrated_logit))


def _metrics(target: np.ndarray, probability: np.ndarray) -> dict[str, float]:
    clipped = np.clip(probability, 1e-7, 1.0 - 1e-7)
    return {
        "brier": float(brier_score_loss(target, clipped)),
        "log_loss": float(log_loss(target, clipped, labels=[0, 1])),
        "mean_probability": float(clipped.mean()),
        "positive_rate": float(target.mean()),
    }


def _calibrate_by_category(
    frame: pl.DataFrame,
    parameters: dict[str, tuple[float, float]],
    global_parameters: tuple[float, float],
) -> np.ndarray:
    result = np.empty(frame.height, dtype=np.float64)
    categories = frame["category"].to_numpy()
    logits = frame["teacher_logit"].to_numpy().astype(np.float64)
    for category in sorted(set(categories.tolist())):
        mask = categories == category
        coefficient, intercept = parameters.get(str(category), global_parameters)
        result[mask] = _probability(logits[mask], coefficient, intercept)
    return result


def main() -> None:
    args = _parse_args()
    for path in (args.fit, args.validation, args.apply):
        if not path.is_file():
            raise FileNotFoundError(path)
    if args.minimum_class_rows < 2:
        raise ValueError("minimum-class-rows must be at least 2")
    if not args.overwrite and (args.output.exists() or args.report.exists()):
        raise FileExistsError(args.output)

    fit = pl.read_parquet(args.fit)
    validation = pl.read_parquet(args.validation)
    apply = pl.read_parquet(args.apply)
    _validate(fit, "fit")
    _validate(validation, "validation")
    required_apply = {"category", "teacher_logit", "teacher_probability"}
    missing_apply = required_apply - set(apply.columns)
    if missing_apply:
        raise ValueError(f"apply is missing columns: {sorted(missing_apply)}")

    fit_target = fit["target"].to_numpy().astype(np.int64)
    fit_logit = fit["teacher_logit"].to_numpy().astype(np.float64)
    global_parameters = _fit_logistic(fit_logit, fit_target)
    category_parameters: dict[str, tuple[float, float]] = {}
    fit_categories = fit["category"].to_numpy()
    for category in sorted(set(fit_categories.tolist())):
        mask = fit_categories == category
        category_target = fit_target[mask]
        counts = np.bincount(category_target, minlength=2)
        if int(counts.min()) < args.minimum_class_rows:
            continue
        coefficient, intercept = _fit_logistic(fit_logit[mask], category_target)
        if coefficient > 0.0:
            category_parameters[str(category)] = (coefficient, intercept)

    validation_target = validation["target"].to_numpy().astype(np.int64)
    validation_logit = validation["teacher_logit"].to_numpy().astype(np.float64)
    raw_validation = validation["teacher_probability"].to_numpy().astype(np.float64)
    global_validation = _probability(validation_logit, *global_parameters)
    validated_categories = set(validation["category"].unique().to_list())
    promoted_category_parameters = {
        category: values
        for category, values in category_parameters.items()
        if category in validated_categories
    }
    category_validation = _calibrate_by_category(
        validation,
        promoted_category_parameters,
        global_parameters,
    )
    validation_metrics = {
        "raw": _metrics(validation_target, raw_validation),
        "global_platt": _metrics(validation_target, global_validation),
        "category_platt": _metrics(validation_target, category_validation),
    }
    chosen = min(
        validation_metrics,
        key=lambda name: (
            validation_metrics[name]["log_loss"],
            validation_metrics[name]["brier"],
        ),
    )

    apply_logit = apply["teacher_logit"].to_numpy().astype(np.float64)
    apply_categories = apply["category"].to_numpy()
    validated_mask = np.isin(
        apply_categories,
        np.asarray(sorted(validated_categories), dtype=object),
    )
    calibrated = apply["teacher_probability"].to_numpy().astype(np.float64).copy()
    if chosen == "raw":
        pass
    elif chosen == "global_platt":
        calibrated[validated_mask] = _probability(
            apply_logit[validated_mask],
            *global_parameters,
        )
    else:
        calibrated_values = _calibrate_by_category(
            apply.filter(pl.Series(validated_mask)),
            promoted_category_parameters,
            global_parameters,
        )
        calibrated[validated_mask] = calibrated_values
    if not np.isfinite(calibrated).all():
        raise RuntimeError("calibrated teacher probabilities are non-finite")

    output = apply.with_columns(
        pl.col("teacher_probability").alias("teacher_probability_raw"),
        pl.Series(
            "teacher_probability_calibrated",
            calibrated.astype(np.float32),
        ),
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.report.parent.mkdir(parents=True, exist_ok=True)
    if args.overwrite:
        args.output.unlink(missing_ok=True)
        args.report.unlink(missing_ok=True)
    temporary = args.output.with_suffix(".tmp.parquet")
    output.write_parquet(temporary, compression="zstd", statistics=True)
    os.replace(temporary, args.output)

    report = {
        "schema_version": 1,
        "created_at_utc": datetime.now(UTC).isoformat(),
        "protocol": (
            "Fit calibration only on registered human training rows, choose the probability "
            "mapping on the frozen inner holdout, then apply the frozen mapping to item-disjoint "
            "organizer-LLM hard pairs."
        ),
        "sources": {
            "fit": {"path": str(args.fit.resolve()), "sha256": _sha256(args.fit)},
            "validation": {
                "path": str(args.validation.resolve()),
                "sha256": _sha256(args.validation),
            },
            "apply": {"path": str(args.apply.resolve()), "sha256": _sha256(args.apply)},
        },
        "fit_rows": fit.height,
        "validation_rows": validation.height,
        "global_parameters": {
            "coefficient": global_parameters[0],
            "intercept": global_parameters[1],
        },
        "category_parameters": {
            category: {"coefficient": values[0], "intercept": values[1]}
            for category, values in sorted(category_parameters.items())
        },
        "promoted_category_parameters": sorted(promoted_category_parameters),
        "unvalidated_categories_use_raw_probability": True,
        "minimum_class_rows": args.minimum_class_rows,
        "validation_metrics": validation_metrics,
        "chosen": chosen,
        "artifact": {
            "path": str(args.output.resolve()),
            "bytes": args.output.stat().st_size,
            "sha256": _sha256(args.output),
        },
    }
    args.report.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(args.report)


if __name__ == "__main__":
    main()
