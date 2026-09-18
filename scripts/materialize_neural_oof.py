from __future__ import annotations

import argparse
import hashlib
import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np
import polars as pl


def _parse_folds(raw: str) -> tuple[int, ...]:
    folds = tuple(sorted({int(value) for value in raw.split(",")}))
    if not folds or any(fold not in range(5) for fold in folds):
        raise argparse.ArgumentTypeError("folds must be a non-empty subset of 0,1,2,3,4")
    return folds


def _sha256(path: Path, chunk_size: int = 8 * 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while chunk := stream.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Materialize one deterministic OOF parquet from frozen neural fold outputs"
    )
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--folds", type=_parse_folds, default=(0, 1, 2, 3, 4))
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def load_recipe_oof(
    root: Path,
    folds: tuple[int, ...],
) -> tuple[pl.DataFrame, list[dict[str, Any]]]:
    frames: list[pl.DataFrame] = []
    inputs: list[dict[str, Any]] = []
    required = {"id1", "id2", "target", "category", "score"}
    for fold in folds:
        fold_dir = root / f"fold_{fold}"
        manifest_path = fold_dir / "training_manifest.json"
        if not manifest_path.is_file():
            raise FileNotFoundError(manifest_path)
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        best_epoch = int(manifest["best_epoch"])
        prediction_path = fold_dir / f"predictions_epoch_{best_epoch}.parquet"
        if not prediction_path.is_file():
            raise FileNotFoundError(prediction_path)
        schema = pl.read_parquet_schema(prediction_path)
        missing = sorted(required - set(schema))
        if missing:
            raise ValueError(f"fold {fold} predictions are missing columns: {missing}")
        frame = (
            pl.read_parquet(prediction_path, columns=sorted(required))
            .select(
                "id1",
                "id2",
                "target",
                "category",
                pl.col("score").cast(pl.Float32).alias("predict"),
            )
            .with_columns(pl.lit(fold, dtype=pl.Int8).alias("fold"))
        )
        if frame.select(pl.any_horizontal(pl.all().is_null()).any()).item():
            raise ValueError(f"fold {fold} predictions contain null values")
        if not np.isfinite(frame["predict"].to_numpy()).all():
            raise ValueError(f"fold {fold} predictions contain non-finite scores")
        if frame.select(pl.struct("id1", "id2").n_unique()).item() != frame.height:
            raise ValueError(f"fold {fold} predictions contain duplicate pair keys")
        frames.append(frame)
        inputs.append(
            {
                "fold": fold,
                "best_epoch": best_epoch,
                "rows": frame.height,
                "manifest": str(manifest_path.resolve()),
                "manifest_sha256": _sha256(manifest_path),
                "predictions": str(prediction_path.resolve()),
                "predictions_sha256": _sha256(prediction_path),
            }
        )

    result = pl.concat(frames, how="vertical").sort("fold", "id1", "id2")
    if result.select(pl.struct("id1", "id2").n_unique()).item() != result.height:
        raise ValueError("OOF predictions contain pair keys in more than one fold")
    if result["category"].n_unique() != 20:
        raise ValueError("OOF predictions must cover exactly 20 categories")
    return result, inputs


def main() -> None:
    args = _parse_args()
    if not args.root.is_dir():
        raise FileNotFoundError(args.root)
    manifest_path = args.output.with_suffix(".manifest.json")
    for path in (args.output, manifest_path):
        if path.exists() and not args.overwrite:
            raise FileExistsError(path)

    frame, inputs = load_recipe_oof(args.root, args.folds)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    frame.write_parquet(args.output, compression="zstd", statistics=True)
    payload = {
        "schema_version": 1,
        "created_at_utc": datetime.now(UTC).isoformat(),
        "protocol": "Exact concatenation of frozen fold predictions; no score transformation.",
        "root": str(args.root.resolve()),
        "folds": list(args.folds),
        "rows": frame.height,
        "categories": frame["category"].n_unique(),
        "columns": frame.columns,
        "inputs": inputs,
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
    print(args.output)


if __name__ == "__main__":
    main()
