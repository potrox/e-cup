from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import polars as pl
from scipy.stats import spearmanr


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--expected", type=Path, required=True)
    parser.add_argument("--actual", type=Path, required=True)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--min-spearman", type=float, default=0.999)
    parser.add_argument("--min-sign-agreement", type=float, default=0.99)
    parser.add_argument("--max-mean-probability-delta", type=float, default=0.005)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    expected = pl.read_parquet(args.expected.resolve(strict=True))
    actual = pl.read_parquet(args.actual.resolve(strict=True))
    key = "teacher_source_row"
    required = {key, "category", "teacher_logit", "teacher_probability"}
    for label, frame in (("expected", expected), ("actual", actual)):
        missing = sorted(required.difference(frame.columns))
        if missing:
            raise ValueError(f"{label} artifact is missing columns: {missing}")
        if frame[key].n_unique() != frame.height:
            raise ValueError(f"{label} artifact contains duplicate source rows")

    comparison = expected.select(
        key,
        "category",
        pl.col("teacher_logit").alias("expected_logit"),
        pl.col("teacher_probability").alias("expected_probability"),
    ).join(
        actual.select(
            key,
            pl.col("teacher_logit").alias("actual_logit"),
            pl.col("teacher_probability").alias("actual_probability"),
        ),
        on=key,
        how="inner",
        validate="1:1",
    )
    if comparison.height != expected.height or comparison.height != actual.height:
        raise RuntimeError("expected and actual sample coverage differs")

    expected_logit = comparison["expected_logit"].to_numpy()
    actual_logit = comparison["actual_logit"].to_numpy()
    expected_probability = comparison["expected_probability"].to_numpy()
    actual_probability = comparison["actual_probability"].to_numpy()
    if not all(
        np.isfinite(values).all()
        for values in (expected_logit, actual_logit, expected_probability, actual_probability)
    ):
        raise RuntimeError("non-finite teacher values")

    probability_delta = np.abs(expected_probability - actual_probability)
    correlation = float(spearmanr(expected_logit, actual_logit).statistic)
    sign_agreement = float(np.mean((expected_logit >= 0.0) == (actual_logit >= 0.0)))
    report = {
        "schema_version": 1,
        "rows": comparison.height,
        "categories": comparison["category"].n_unique(),
        "metrics": {
            "spearman_logit": correlation,
            "sign_agreement": sign_agreement,
            "mean_absolute_probability_delta": float(probability_delta.mean()),
            "max_absolute_probability_delta": float(probability_delta.max()),
        },
        "gates": {
            "min_spearman": args.min_spearman,
            "min_sign_agreement": args.min_sign_agreement,
            "max_mean_probability_delta": args.max_mean_probability_delta,
        },
    }
    passed = (
        correlation >= args.min_spearman
        and sign_agreement >= args.min_sign_agreement
        and float(probability_delta.mean()) <= args.max_mean_probability_delta
    )
    report["status"] = "pass" if passed else "fail"
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(
            json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
            newline="\n",
        )
    print(json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True))
    if not passed:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
