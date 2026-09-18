from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import polars as pl

CHUNK_BYTES = 8 * 1024 * 1024


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=Path.cwd())
    return parser.parse_args()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while chunk := stream.read(CHUNK_BYTES):
            digest.update(chunk)
    return digest.hexdigest()


def verify_file(root: Path, record: dict[str, object]) -> None:
    path = (root / str(record["path"])).resolve(strict=True)
    if root not in path.parents:
        raise ValueError(f"artifact escaped repository root: {path}")
    if path.stat().st_size != int(record["bytes"]):
        raise RuntimeError(f"artifact byte count mismatch: {path}")
    if sha256_file(path) != record["sha256"]:
        raise RuntimeError(f"artifact SHA-256 mismatch: {path}")


def main() -> None:
    args = parse_args()
    root = args.root.resolve(strict=True)
    label_manifest = json.loads(
        (root / "artifacts/llm_labels/ARTIFACT_MANIFEST.json").read_text(encoding="utf-8")
    )
    checked_labels = []
    for record in label_manifest["artifacts"]:
        verify_file(root, record)
        path = root / record["path"]
        summary = pl.scan_parquet(path).select(
            pl.len().alias("rows"),
            pl.col("category").n_unique().alias("categories"),
        ).collect()
        if int(summary["rows"][0]) != record["rows"]:
            raise RuntimeError(f"artifact row count mismatch: {path}")
        if int(summary["categories"][0]) != record["categories"]:
            raise RuntimeError(f"artifact category count mismatch: {path}")
        checked_labels.append(record["path"])

    sample = label_manifest["reproducibility_sample"]
    verify_file(root, sample)
    sample_summary = pl.scan_parquet(root / sample["path"]).select(
        pl.len().alias("rows"),
        pl.col("category").n_unique().alias("categories"),
    ).collect()
    if int(sample_summary["rows"][0]) != sample["rows"]:
        raise RuntimeError("teacher sample row count mismatch")
    if int(sample_summary["categories"][0]) != sample["categories"]:
        raise RuntimeError("teacher sample category count mismatch")

    model_manifest = json.loads(
        (root / "artifacts/trained_models/ARTIFACT_MANIFEST.json").read_text(
            encoding="utf-8"
        )
    )
    reconstructed_digest = hashlib.sha256()
    reconstructed_bytes = 0
    model_root = root / "artifacts/trained_models"
    for record in model_manifest["parts"]:
        part = model_root / record["path"]
        if part.stat().st_size != record["bytes"] or sha256_file(part) != record["sha256"]:
            raise RuntimeError(f"trained release part failed integrity: {part}")
        with part.open("rb") as stream:
            while chunk := stream.read(CHUNK_BYTES):
                reconstructed_digest.update(chunk)
                reconstructed_bytes += len(chunk)
    expected = model_manifest["reconstructed_artifact"]
    if reconstructed_bytes != expected["bytes"]:
        raise RuntimeError("trained release reconstructed byte count mismatch")
    if reconstructed_digest.hexdigest() != expected["sha256"]:
        raise RuntimeError("trained release reconstructed SHA-256 mismatch")

    print(
        json.dumps(
            {
                "status": "pass",
                "llm_label_artifacts": checked_labels,
                "teacher_sample_rows": int(sample_summary["rows"][0]),
                "trained_release_parts": len(model_manifest["parts"]),
                "trained_release_sha256": reconstructed_digest.hexdigest(),
            },
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
