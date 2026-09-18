from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import polars as pl

from ecup_matching.teacher_distillation import corrected_teacher_target

DEFAULT_WEAK_CATEGORIES = ("Одежда", "Обувь", "Ювелирные изделия")


def _parse_categories(raw: str) -> tuple[str, ...]:
    values = tuple(value.strip() for value in raw.split(",") if value.strip())
    if not values or len(values) != len(set(values)):
        raise argparse.ArgumentTypeError("categories must be unique and non-empty")
    return values


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build high-confidence corrected LLM replay")
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--rows", type=int, default=58_506)
    parser.add_argument(
        "--weak-categories",
        type=_parse_categories,
        default=DEFAULT_WEAK_CATEGORIES,
    )
    parser.add_argument("--weak-fraction", type=float, default=0.75)
    parser.add_argument("--teacher-weight", type=float, default=0.8)
    parser.add_argument(
        "--teacher-probability-column",
        default="teacher_probability",
    )
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def _sha256(path: Path, chunk_size: int = 8 * 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while block := handle.read(chunk_size):
            digest.update(block)
    return digest.hexdigest()


def _quotas(
    categories: list[str],
    weak: tuple[str, ...],
    rows: int,
    weak_fraction: float,
) -> dict[str, int]:
    strong = sorted(set(categories) - set(weak))
    weak_rows = round(rows * weak_fraction)
    weak_base, weak_extra = divmod(weak_rows, len(weak))
    strong_base, strong_extra = divmod(rows - weak_rows, len(strong))
    result = {
        category: weak_base + int(index < weak_extra)
        for index, category in enumerate(sorted(weak))
    }
    result.update(
        {
            category: strong_base + int(index < strong_extra)
            for index, category in enumerate(strong)
        }
    )
    return result


def main() -> None:
    args = _parse_args()
    if not args.input.is_file():
        raise FileNotFoundError(args.input)
    if args.rows < 20 or not 0.0 < args.weak_fraction < 1.0:
        raise ValueError("invalid rows or weak fraction")
    if not 0.0 <= args.teacher_weight <= 1.0:
        raise ValueError("teacher-weight must be in [0, 1]")
    manifest_path = args.output.with_suffix(".manifest.json")
    if not args.overwrite and (args.output.exists() or manifest_path.exists()):
        raise FileExistsError(args.output)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    if args.overwrite:
        args.output.unlink(missing_ok=True)
        manifest_path.unlink(missing_ok=True)

    frame = pl.read_parquet(args.input)
    required = {
        "id1",
        "id2",
        "target",
        "category",
        args.teacher_probability_column,
    }
    missing = required - set(frame.columns)
    if missing:
        raise ValueError(f"teacher scores are missing columns: {sorted(missing)}")
    categories = sorted(frame["category"].unique().to_list())
    if len(categories) != 20:
        raise ValueError(f"expected 20 categories, got {len(categories)}")
    unknown = sorted(set(args.weak_categories) - set(categories))
    if unknown:
        raise ValueError(f"unknown weak categories: {unknown}")
    quotas = _quotas(categories, args.weak_categories, args.rows, args.weak_fraction)

    prepared = frame.with_columns(
        pl.col(args.teacher_probability_column).alias("teacher_probability_used"),
        pl.col("target").alias("organizer_target"),
        (2.0 * (pl.col(args.teacher_probability_column) - 0.5).abs()).alias(
            "teacher_confidence"
        ),
        (pl.col(args.teacher_probability_column) - pl.col("target"))
        .abs()
        .alias("teacher_correction_magnitude"),
    ).with_columns(
        (pl.col("teacher_confidence") * pl.col("teacher_correction_magnitude")).alias(
            "teacher_correction_priority"
        )
    )
    selected = pl.concat(
        [
            prepared.filter(pl.col("category") == category)
            .sort(
                "teacher_correction_priority",
                pl.struct("id1", "id2").hash(seed=args.seed),
                descending=[True, False],
            )
            .head(quota)
            for category, quota in sorted(quotas.items())
        ]
    ).sort(pl.struct("id1", "id2").hash(seed=args.seed + 1))
    if selected.height != args.rows:
        raise RuntimeError(f"expected {args.rows} replay rows, selected {selected.height}")

    corrected = [
        corrected_teacher_target(
            float(organizer),
            float(teacher),
            teacher_weight=args.teacher_weight,
        )
        for organizer, teacher in zip(
            selected["organizer_target"],
            selected["teacher_probability_used"],
            strict=True,
        )
    ]
    selected = selected.with_columns(pl.Series("target", corrected, dtype=pl.Float64))
    if selected["target"].is_null().any() or not selected["target"].is_between(0.0, 1.0).all():
        raise RuntimeError("corrected targets are invalid")
    selected.write_parquet(args.output, compression="zstd", statistics=True)

    summary = (
        selected.group_by("category")
        .agg(
            pl.len().alias("rows"),
            pl.col("organizer_target").mean().alias("organizer_target_mean"),
            pl.col("teacher_probability_used")
            .mean()
            .alias("teacher_probability_mean"),
            pl.col("target").mean().alias("corrected_target_mean"),
            pl.col("teacher_confidence").mean().alias("teacher_confidence_mean"),
            pl.col("teacher_correction_magnitude")
            .mean()
            .alias("correction_magnitude_mean"),
        )
        .sort("category")
        .to_dicts()
    )
    payload = {
        "schema_version": 1,
        "protocol": (
            "Within each fixed category quota, retain the highest product of teacher confidence "
            "and absolute disagreement with the organizer soft target. Corrected targets are a "
            "fixed convex blend; human labels are never read."
        ),
        "source": {"path": str(args.input.resolve()), "sha256": _sha256(args.input)},
        "config": {
            "rows": args.rows,
            "weak_categories": list(args.weak_categories),
            "weak_fraction": args.weak_fraction,
            "teacher_weight": args.teacher_weight,
            "teacher_probability_column": args.teacher_probability_column,
            "category_quotas": quotas,
            "seed": args.seed,
        },
        "audit": {
            "rows": selected.height,
            "unique_pairs": selected.select(pl.struct("id1", "id2").n_unique()).item(),
            "categories": selected["category"].n_unique(),
            "target_min": selected["target"].min(),
            "target_max": selected["target"].max(),
            "summary": summary,
        },
        "output": {
            "path": str(args.output.resolve()),
            "bytes": args.output.stat().st_size,
            "sha256": _sha256(args.output),
        },
    }
    manifest_path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(manifest_path)


if __name__ == "__main__":
    main()
