"""Generate deterministic, product-disjoint folds for human-labeled pairs."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import polars as pl

from ecup_matching.splits import connected_component_ids, stratified_component_folds


def _sha256(path: Path, chunk_size: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        while chunk := source.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=Path("."))
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("data_derived/splits"),
    )
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--seed", type=int, default=2026)
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    matches_path = args.data_dir / "matches.parquet"
    items_path = args.data_dir / "items_human.parquet"
    for path in (matches_path, items_path):
        if not path.is_file():
            raise FileNotFoundError(path)

    pairs = pl.read_parquet(matches_path)
    items = pl.read_parquet(items_path, columns=["id", "category"])
    if pairs.select(pl.col("target").is_in([0.0, 1.0]).all()).item() is not True:
        raise ValueError("human labels must be strictly binary")

    joined = pairs.join(
        items.rename({"id": "id1"}),
        on="id1",
        how="left",
        validate="m:1",
    )
    if joined["category"].null_count() != 0:
        raise ValueError("some id1 values are missing from items_human.parquet")

    component_ids = connected_component_ids(joined["id1"], joined["id2"])
    folds = stratified_component_folds(
        joined["target"],
        joined["category"],
        component_ids,
        n_splits=args.folds,
        random_state=args.seed,
    )
    result = joined.with_columns(
        pl.Series("component_id", component_ids),
        pl.Series("fold", folds),
    ).select("id1", "id2", "target", "category", "component_id", "fold")

    leakage = (
        result.group_by("component_id")
        .agg(pl.col("fold").n_unique().alias("fold_count"))
        .filter(pl.col("fold_count") != 1)
        .height
    )
    if leakage != 0:
        raise RuntimeError(f"detected {leakage} components spanning multiple folds")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    split_path = args.output_dir / "human_folds.parquet"
    manifest_path = args.output_dir / "human_folds.manifest.json"
    result.write_parquet(split_path, compression="zstd", statistics=True)

    summary = (
        result.group_by("fold", "category")
        .agg(
            pl.len().alias("rows"),
            pl.col("target").sum().cast(pl.Int64).alias("positives"),
            pl.col("target").mean().alias("positive_rate"),
            pl.col("component_id").n_unique().alias("components"),
        )
        .sort("fold", "category")
    )
    manifest = {
        "schema_version": 1,
        "strategy": "StratifiedGroupKFold over all-edge product connected components",
        "n_splits": args.folds,
        "random_state": args.seed,
        "rows": result.height,
        "components": result["component_id"].n_unique(),
        "sources": {
            str(matches_path): _sha256(matches_path),
            str(items_path): _sha256(items_path),
        },
        "fold_summary": summary.to_dicts(),
    }
    manifest_path.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )

    overall = (
        result.group_by("fold")
        .agg(
            pl.len().alias("rows"),
            pl.col("target").sum().cast(pl.Int64).alias("positives"),
            pl.col("target").mean().alias("positive_rate"),
            pl.col("component_id").n_unique().alias("components"),
        )
        .sort("fold")
    )
    print(overall)
    print(f"wrote {split_path}")
    print(f"wrote {manifest_path}")


if __name__ == "__main__":
    main()
