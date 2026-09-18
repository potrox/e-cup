from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import polars as pl

DEFAULT_WEAK_CATEGORIES = ("Одежда", "Обувь", "Ювелирные изделия")


def _parse_categories(raw: str) -> tuple[str, ...]:
    values = tuple(value.strip() for value in raw.split(",") if value.strip())
    if not values or len(values) != len(set(values)):
        raise argparse.ArgumentTypeError("categories must be unique and non-empty")
    return values


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Select model-disagreement pairs for expensive teacher scoring"
    )
    parser.add_argument("--candidates", type=Path, required=True)
    parser.add_argument("--base", type=Path, required=True)
    parser.add_argument("--lamar", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--rows", type=int, default=80_000)
    parser.add_argument(
        "--weak-categories",
        type=_parse_categories,
        default=DEFAULT_WEAK_CATEGORIES,
    )
    parser.add_argument("--weak-fraction", type=float, default=0.75)
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
    unknown = sorted(set(weak) - set(categories))
    if unknown:
        raise ValueError(f"unknown weak categories: {unknown}")
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
    if sum(result.values()) != rows:
        raise AssertionError("category quota calculation lost rows")
    return result


def _prediction_frame(path: Path, name: str) -> pl.DataFrame:
    frame = pl.read_parquet(path)
    required = {"id1", "id2", "target", "category", "score"}
    missing = required - set(frame.columns)
    if missing:
        raise ValueError(f"{name} predictions are missing columns: {sorted(missing)}")
    return frame.select(
        "id1",
        "id2",
        pl.col("target").alias(f"{name}_target"),
        pl.col("category").alias(f"{name}_category"),
        pl.col("score").alias(f"{name}_score"),
    )


def main() -> None:
    args = _parse_args()
    for path in (args.candidates, args.base, args.lamar):
        if not path.is_file():
            raise FileNotFoundError(path)
    if args.rows < 20 or not 0.0 < args.weak_fraction < 1.0:
        raise ValueError("invalid rows or weak fraction")
    manifest_path = args.output.with_suffix(".manifest.json")
    if not args.overwrite and (args.output.exists() or manifest_path.exists()):
        raise FileExistsError(args.output)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    if args.overwrite:
        args.output.unlink(missing_ok=True)
        manifest_path.unlink(missing_ok=True)

    candidates = pl.read_parquet(args.candidates)
    base = _prediction_frame(args.base, "base")
    lamar = _prediction_frame(args.lamar, "lamar")
    joined = candidates.join(base, on=["id1", "id2"], how="inner", validate="1:1").join(
        lamar,
        on=["id1", "id2"],
        how="inner",
        validate="1:1",
    )
    if joined.height != candidates.height:
        raise ValueError("model predictions do not cover the candidate pool")
    for name in ("base", "lamar"):
        target_delta = (
            joined["target"].cast(pl.Float64)
            - joined[f"{name}_target"].cast(pl.Float64)
        ).abs()
        if target_delta.null_count() or target_delta.max() > 1e-7:
            raise ValueError(f"{name} targets differ from candidate targets")
        if not joined["category"].equals(joined[f"{name}_category"], null_equal=True):
            raise ValueError(f"{name} categories differ from candidate categories")

    ranked = joined.with_columns(
        (
            (pl.col("base_score").rank("average").over("category") - 1)
            / (pl.len().over("category") - 1).clip(lower_bound=1)
        ).alias("base_category_rank"),
        (
            (pl.col("lamar_score").rank("average").over("category") - 1)
            / (pl.len().over("category") - 1).clip(lower_bound=1)
        ).alias("lamar_category_rank"),
    ).with_columns(
        ((pl.col("base_category_rank") + pl.col("lamar_category_rank")) * 0.5).alias(
            "model_mean_rank"
        ),
        (pl.col("base_category_rank") - pl.col("lamar_category_rank"))
        .abs()
        .alias("model_disagreement"),
    )
    false_negative = (1.0 - pl.col("target")) * pl.col("model_mean_rank")
    false_positive = pl.col("target") * (1.0 - pl.col("model_mean_rank"))
    ranked = ranked.with_columns(
        pl.max_horizontal(
            false_negative,
            false_positive,
            pl.col("model_disagreement") * 0.8,
        ).alias("teacher_priority"),
        pl.when(false_negative >= false_positive)
        .then(pl.lit("possible_false_negative"))
        .otherwise(pl.lit("possible_false_positive"))
        .alias("teacher_reason"),
    )

    categories = sorted(ranked["category"].unique().to_list())
    if len(categories) != 20:
        raise ValueError(f"expected 20 categories, got {len(categories)}")
    quotas = _quotas(categories, args.weak_categories, args.rows, args.weak_fraction)
    selected = pl.concat(
        [
            ranked.filter(pl.col("category") == category)
            .sort(
                "teacher_priority",
                pl.struct("id1", "id2").hash(seed=args.seed),
                descending=[True, False],
            )
            .head(quota)
            for category, quota in sorted(quotas.items())
        ]
    ).sort(pl.struct("id1", "id2").hash(seed=args.seed + 1))
    if selected.height != args.rows or selected["category"].n_unique() != 20:
        raise RuntimeError("teacher scoring subset coverage mismatch")
    if selected.select(pl.struct("id1", "id2").n_unique()).item() != selected.height:
        raise RuntimeError("teacher scoring subset contains duplicate pairs")
    selected.write_parquet(args.output, compression="zstd", statistics=True)

    summary = (
        selected.group_by("category", "teacher_reason")
        .agg(
            pl.len().alias("rows"),
            pl.col("teacher_priority").mean().alias("mean_priority"),
            pl.col("target").mean().alias("mean_organizer_target"),
        )
        .sort("category", "teacher_reason")
        .to_dicts()
    )
    payload = {
        "schema_version": 1,
        "protocol": (
            "Within-category percentile ranks from immutable base and LAMAR scores. "
            "Priority is the maximum suspected false-negative, suspected false-positive, "
            "or model-disagreement signal. Human labels are never read."
        ),
        "sources": {
            "candidates": {
                "path": str(args.candidates.resolve()),
                "sha256": _sha256(args.candidates),
            },
            "base": {"path": str(args.base.resolve()), "sha256": _sha256(args.base)},
            "lamar": {"path": str(args.lamar.resolve()), "sha256": _sha256(args.lamar)},
        },
        "config": {
            "rows": args.rows,
            "weak_categories": list(args.weak_categories),
            "weak_fraction": args.weak_fraction,
            "category_quotas": quotas,
            "seed": args.seed,
        },
        "audit": {
            "rows": selected.height,
            "unique_pairs": selected.select(pl.struct("id1", "id2").n_unique()).item(),
            "categories": selected["category"].n_unique(),
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
