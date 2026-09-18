from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import polars as pl

from ecup_matching.splits import stratified_component_folds


def _sha256(path: Path, chunk_size: int = 8 * 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Add a deterministic component-disjoint inner fold to human pair text"
    )
    parser.add_argument(
        "--input",
        type=Path,
        default=Path("data_derived/neural/pairs_v2/human_all.parquet"),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("data_derived/neural/pairs_v2/human_all_nested.parquet"),
    )
    parser.add_argument("--inner-folds", type=int, default=5)
    parser.add_argument("--inner-validation-fold", type=int, default=0)
    parser.add_argument("--seed", type=int, default=2027)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    if not args.input.is_file():
        raise FileNotFoundError(args.input)
    manifest_path = args.output.with_suffix(".manifest.json")
    if not args.overwrite and (args.output.exists() or manifest_path.exists()):
        raise FileExistsError(args.output)
    if args.inner_folds < 2:
        raise ValueError("inner_folds must be at least 2")
    if not 0 <= args.inner_validation_fold < args.inner_folds:
        raise ValueError("inner_validation_fold is outside the configured folds")

    frame = pl.read_parquet(args.input)
    required = {"id1", "id2", "target", "category", "component_id", "fold"}
    missing = required - set(frame.columns)
    if missing:
        raise ValueError(f"human pair source is missing columns: {sorted(missing)}")
    inner = stratified_component_folds(
        frame["target"],
        frame["category"],
        frame["component_id"],
        n_splits=args.inner_folds,
        random_state=args.seed,
    )
    result = frame.with_columns(pl.Series("inner_fold", inner))
    component_leakage = (
        result.group_by("component_id")
        .agg(
            pl.col("fold").n_unique().alias("outer_folds"),
            pl.col("inner_fold").n_unique().alias("inner_folds"),
        )
        .filter((pl.col("outer_folds") != 1) | (pl.col("inner_folds") != 1))
        .height
    )
    if component_leakage:
        raise RuntimeError(f"components span validation folds: {component_leakage}")

    inner_summary = (
        result.group_by("inner_fold", "category")
        .agg(
            pl.len().alias("rows"),
            pl.col("target").sum().cast(pl.Int64).alias("positives"),
            pl.col("component_id").n_unique().alias("components"),
        )
        .sort("inner_fold", "category")
    )
    if inner_summary["category"].n_unique() != 20:
        raise ValueError("nested split does not contain all 20 categories")
    if inner_summary.height != args.inner_folds * 20:
        raise ValueError("at least one inner fold is missing a category")

    def item_ids(selected: pl.DataFrame) -> set[int]:
        return set(
            pl.concat(
                (
                    selected.select(pl.col("id1").alias("id")),
                    selected.select(pl.col("id2").alias("id")),
                ),
                how="vertical",
            )
            .unique()["id"]
            .to_list()
        )

    item_overlap_audit: list[dict[str, int | str]] = []
    for outer_fold in sorted(result["fold"].unique().to_list()):
        outer_validation = result.filter(pl.col("fold") == outer_fold)
        outer_training = result.filter(pl.col("fold") != outer_fold)
        inner_validation = outer_training.filter(pl.col("inner_fold") == args.inner_validation_fold)
        inner_training = outer_training.filter(pl.col("inner_fold") != args.inner_validation_fold)
        outer_ids = item_ids(outer_validation)
        inner_validation_ids = item_ids(inner_validation)
        inner_training_ids = item_ids(inner_training)
        audit = {
            "outer_fold": int(outer_fold),
            "outer_validation_items": len(outer_ids),
            "inner_validation_items": len(inner_validation_ids),
            "inner_training_items": len(inner_training_ids),
            "inner_train_validation_overlap": len(inner_training_ids & inner_validation_ids),
            "inner_train_outer_overlap": len(inner_training_ids & outer_ids),
            "inner_validation_outer_overlap": len(inner_validation_ids & outer_ids),
        }
        if any(
            audit[key]
            for key in (
                "inner_train_validation_overlap",
                "inner_train_outer_overlap",
                "inner_validation_outer_overlap",
            )
        ):
            raise RuntimeError(f"item leakage detected for outer fold {outer_fold}: {audit}")
        item_overlap_audit.append(audit)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    result.write_parquet(args.output, compression="zstd", statistics=True)
    manifest = {
        "schema_version": 1,
        "strategy": "StratifiedGroupKFold over frozen human component_id for inner selection",
        "outer_fold_column": "fold",
        "inner_fold_column": "inner_fold",
        "inner_folds": args.inner_folds,
        "inner_validation_fold": args.inner_validation_fold,
        "random_state": args.seed,
        "rows": result.height,
        "components": result["component_id"].n_unique(),
        "component_leakage": component_leakage,
        "item_overlap_audit": item_overlap_audit,
        "source": {
            "path": str(args.input.resolve()),
            "sha256": _sha256(args.input),
        },
        "output": {
            "path": str(args.output.resolve()),
            "bytes": args.output.stat().st_size,
            "sha256": _sha256(args.output),
        },
        "inner_summary": inner_summary.to_dicts(),
    }
    manifest_path.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(args.output)
    print(manifest_path)


if __name__ == "__main__":
    main()
