from __future__ import annotations

import argparse
import json
import math
import os
from pathlib import Path

import numpy as np
import polars as pl


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Apply the frozen teacher calibration contract")
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--calibration", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def _probability(logit: np.ndarray, coefficient: float, intercept: float) -> np.ndarray:
    calibrated = np.clip(coefficient * logit + intercept, -40.0, 40.0)
    return 1.0 / (1.0 + np.exp(-calibrated))


def main() -> None:
    args = _parse_args()
    if not args.input.is_file() or not args.calibration.is_file():
        raise FileNotFoundError(args.input if not args.input.is_file() else args.calibration)
    if args.output.exists() and not args.overwrite:
        raise FileExistsError(args.output)
    contract = json.loads(args.calibration.read_text(encoding="utf-8"))
    if contract.get("schema_version") != 1 or contract.get("chosen") != "category_platt":
        raise ValueError("unsupported teacher-calibration contract")
    parameters = contract.get("category_parameters")
    if not isinstance(parameters, dict) or not parameters:
        raise ValueError("teacher-calibration contract has no category parameters")
    frame = pl.read_parquet(args.input)
    required = {"category", "teacher_logit", "teacher_probability"}
    missing = required - set(frame.columns)
    if missing:
        raise ValueError(f"teacher scores are missing columns: {sorted(missing)}")
    calibrated = frame["teacher_probability"].to_numpy().astype(np.float64).copy()
    categories = frame["category"].to_numpy()
    logits = frame["teacher_logit"].to_numpy().astype(np.float64)
    for category, values in parameters.items():
        coefficient = float(values["coefficient"])
        intercept = float(values["intercept"])
        if not math.isfinite(coefficient) or not math.isfinite(intercept):
            raise ValueError(f"non-finite calibration parameters for {category}")
        mask = categories == category
        calibrated[mask] = _probability(logits[mask], coefficient, intercept)
    if not np.isfinite(calibrated).all():
        raise RuntimeError("calibrated teacher probabilities are non-finite")
    output = frame.with_columns(
        pl.col("teacher_probability").alias("teacher_probability_raw"),
        pl.Series("teacher_probability_calibrated", calibrated.astype(np.float32)),
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.output.with_suffix(".tmp.parquet")
    output.write_parquet(temporary, compression="zstd", statistics=True)
    if args.overwrite:
        args.output.unlink(missing_ok=True)
    os.replace(temporary, args.output)
    print(args.output)


if __name__ == "__main__":
    main()
