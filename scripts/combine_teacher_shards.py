from __future__ import annotations

import argparse
import hashlib
import json
import os
from datetime import UTC, datetime
from pathlib import Path

import polars as pl


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Combine and validate deterministic causal-teacher shards"
    )
    parser.add_argument("--input-dir", type=Path, required=True)
    parser.add_argument("--pattern", required=True)
    parser.add_argument("--shard-count", type=int, required=True)
    parser.add_argument("--expected-rows", type=int, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def _sha256(path: Path, chunk_size: int = 8 * 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while block := handle.read(chunk_size):
            digest.update(block)
    return digest.hexdigest()


def main() -> None:
    args = _parse_args()
    if args.shard_count < 1 or args.expected_rows < 1:
        raise ValueError("shard-count and expected-rows must be positive")
    manifest_path = args.output.with_suffix(".manifest.json")
    if not args.overwrite and (args.output.exists() or manifest_path.exists()):
        raise FileExistsError(args.output)

    shard_paths = [
        args.input_dir / args.pattern.format(index=index)
        for index in range(args.shard_count)
    ]
    missing = [str(path) for path in shard_paths if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"missing teacher shards: {missing}")

    frames: list[pl.DataFrame] = []
    source_hashes: set[str] = set()
    shard_records: list[dict[str, object]] = []
    for expected_index, path in enumerate(shard_paths):
        shard_manifest_path = path.with_suffix(".manifest.json")
        if not shard_manifest_path.is_file():
            raise FileNotFoundError(shard_manifest_path)
        shard_manifest = json.loads(shard_manifest_path.read_text(encoding="utf-8"))
        config = shard_manifest["config"]
        if config["shard_count"] != args.shard_count:
            raise ValueError(f"wrong shard_count in {shard_manifest_path}")
        if config["shard_index"] != expected_index:
            raise ValueError(f"wrong shard_index in {shard_manifest_path}")
        artifact_sha256 = _sha256(path)
        if artifact_sha256 != shard_manifest["artifact"]["sha256"]:
            raise ValueError(f"artifact hash mismatch for {path}")
        source_hashes.add(shard_manifest["data"]["source_sha256"])
        frame = pl.read_parquet(path)
        if frame.height != shard_manifest["data"]["rows"]:
            raise ValueError(f"row count mismatch for {path}")
        frames.append(frame)
        shard_records.append(
            {
                "index": expected_index,
                "path": str(path.resolve()),
                "rows": frame.height,
                "sha256": artifact_sha256,
            }
        )
    if len(source_hashes) != 1:
        raise ValueError("teacher shards do not share one immutable source")

    combined = pl.concat(frames, how="vertical_relaxed").sort("teacher_source_row")
    if combined.height != args.expected_rows:
        raise ValueError(
            f"expected {args.expected_rows} rows, combined {combined.height}"
        )
    if combined["teacher_source_row"].n_unique() != combined.height:
        raise ValueError("duplicate teacher_source_row across shards")
    if combined.select(pl.struct("id1", "id2").n_unique()).item() != combined.height:
        raise ValueError("duplicate pair key across shards")
    if combined["teacher_source_row"].min() != 0:
        raise ValueError("teacher_source_row does not start at zero")
    if combined["teacher_source_row"].max() != args.expected_rows - 1:
        raise ValueError("teacher_source_row does not cover the full source")
    if combined.null_count().select(pl.sum_horizontal(pl.all())).item() != 0:
        raise ValueError("combined teacher output contains nulls")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    if args.overwrite:
        args.output.unlink(missing_ok=True)
        manifest_path.unlink(missing_ok=True)
    temporary = args.output.with_suffix(".tmp.parquet")
    combined.write_parquet(temporary, compression="zstd", statistics=True)
    os.replace(temporary, args.output)

    payload = {
        "schema_version": 1,
        "created_at_utc": datetime.now(UTC).isoformat(),
        "protocol": "Exact concatenation of complete deterministic teacher shards.",
        "source_sha256": next(iter(source_hashes)),
        "shards": shard_records,
        "audit": {
            "rows": combined.height,
            "unique_pairs": combined.select(
                pl.struct("id1", "id2").n_unique()
            ).item(),
            "unique_source_rows": combined["teacher_source_row"].n_unique(),
            "categories": combined["category"].n_unique(),
            "finite_teacher_logits": bool(
                combined["teacher_logit"].is_finite().all()
            ),
        },
        "artifact": {
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
