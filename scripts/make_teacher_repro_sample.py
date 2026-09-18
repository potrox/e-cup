from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path

import polars as pl


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--rows-per-category", type=int, default=100)
    parser.add_argument("--seed", type=int, default=2026)
    return parser.parse_args()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while chunk := stream.read(8 * 1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> None:
    args = parse_args()
    source = args.input.resolve(strict=True)
    output = args.output.resolve(strict=False)
    if args.rows_per_category < 1:
        raise ValueError("rows-per-category must be positive")
    if output.exists():
        raise FileExistsError(output)

    frame = pl.read_parquet(source)
    required = {
        "teacher_source_row",
        "id1",
        "id2",
        "category",
        "left_name",
        "left_attributes",
        "right_name",
        "right_attributes",
        "teacher_logit",
        "teacher_probability",
        "teacher_logit_forward",
        "teacher_logit_backward",
    }
    missing = sorted(required.difference(frame.columns))
    if missing:
        raise ValueError(f"teacher artifact is missing columns: {missing}")

    categories = sorted(frame["category"].unique().to_list())
    selected = (
        frame.with_columns(
            pl.struct("category", "id1", "id2", "teacher_source_row")
            .hash(seed=args.seed)
            .alias("_sample_hash")
        )
        .sort("category", "_sample_hash", "teacher_source_row")
        .group_by("category", maintain_order=True)
        .head(args.rows_per_category)
        .drop("_sample_hash")
        .sort("category", "teacher_source_row")
    )
    expected_rows = len(categories) * args.rows_per_category
    if selected.height != expected_rows:
        raise RuntimeError(f"sample row count mismatch: {selected.height} != {expected_rows}")
    counts = selected.group_by("category").len().sort("category")
    if counts["len"].min() != args.rows_per_category:
        raise RuntimeError("at least one category has an incomplete sample")

    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_suffix(".tmp.parquet")
    selected.write_parquet(temporary, compression="zstd", statistics=True)
    os.replace(temporary, output)
    report = {
        "schema_version": 1,
        "protocol": "deterministic category-balanced teacher reproducibility sample",
        "source": {
            "rows": frame.height,
            "sha256": sha256_file(source),
        },
        "sample": {
            "path": output.name,
            "rows": selected.height,
            "categories": len(categories),
            "rows_per_category": args.rows_per_category,
            "seed": args.seed,
            "bytes": output.stat().st_size,
            "sha256": sha256_file(output),
        },
    }
    report_path = output.with_suffix(".manifest.json")
    report_path.write_text(
        json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
        newline="\n",
    )
    print(json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
