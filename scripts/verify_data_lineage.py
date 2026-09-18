from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import pyarrow.parquet as pq


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Verify frozen Task 1 data lineage")
    parser.add_argument("--manifest", type=Path, default=Path("evidence/data_lineage.json"))
    parser.add_argument("--group", required=True)
    parser.add_argument("--root", type=Path, default=Path.cwd())
    return parser.parse_args()


def _sha256(path: Path, chunk_size: int = 8 * 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> None:
    args = _parse_args()
    root = args.root.resolve(strict=True)
    manifest_path = args.manifest if args.manifest.is_absolute() else root / args.manifest
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("schema_version") != 2:
        raise ValueError("data-lineage schema_version must be 2")
    artifacts = manifest.get("artifacts")
    groups = manifest.get("groups")
    if not isinstance(artifacts, dict) or not isinstance(groups, dict):
        raise ValueError("data-lineage manifest requires artifacts and groups")
    selected = groups.get(args.group)
    if not isinstance(selected, list) or not selected:
        raise ValueError(f"unknown or empty data-lineage group: {args.group}")
    results = []
    for relative in selected:
        contract = artifacts.get(relative)
        if not isinstance(relative, str) or not isinstance(contract, dict):
            raise ValueError(f"invalid artifact contract: {relative!r}")
        path = root / relative
        if not path.is_file():
            raise FileNotFoundError(path)
        rows = pq.ParquetFile(path).metadata.num_rows
        digest = _sha256(path)
        if rows != int(contract["rows"]):
            raise ValueError(f"row count mismatch for {relative}")
        if digest != contract["sha256"]:
            raise ValueError(f"SHA-256 mismatch for {relative}")
        results.append({"path": relative, "rows": rows, "sha256": digest})
    print(json.dumps({"group": args.group, "verified": results}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
