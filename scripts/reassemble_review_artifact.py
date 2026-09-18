from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
from pathlib import Path

CHUNK_BYTES = 8 * 1024 * 1024


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--manifest",
        type=Path,
        default=Path("artifacts/trained_models/ARTIFACT_MANIFEST.json"),
    )
    parser.add_argument("--output", type=Path)
    parser.add_argument("--check-only", action="store_true")
    return parser.parse_args()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while chunk := stream.read(CHUNK_BYTES):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> None:
    args = parse_args()
    manifest_path = args.manifest.resolve(strict=True)
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    root = manifest_path.parent
    reconstructed = manifest["reconstructed_artifact"]
    parts = manifest["parts"]
    digest = hashlib.sha256()
    total = 0
    for record in parts:
        part = root / record["path"]
        if not part.is_file():
            raise FileNotFoundError(part)
        if part.stat().st_size != record["bytes"]:
            raise RuntimeError(f"part size mismatch: {part}")
        actual_hash = sha256_file(part)
        if actual_hash != record["sha256"]:
            raise RuntimeError(f"part hash mismatch: {part}")
        with part.open("rb") as stream:
            while chunk := stream.read(CHUNK_BYTES):
                digest.update(chunk)
                total += len(chunk)
    if total != reconstructed["bytes"]:
        raise RuntimeError("reconstructed byte count mismatch")
    if digest.hexdigest() != reconstructed["sha256"]:
        raise RuntimeError("reconstructed SHA-256 mismatch")

    result = {
        "status": "pass",
        "parts": len(parts),
        "bytes": total,
        "sha256": digest.hexdigest(),
    }
    if not args.check_only:
        output = (
            args.output.resolve(strict=False)
            if args.output is not None
            else (root / reconstructed["name"]).resolve(strict=False)
        )
        if output.exists():
            if output.stat().st_size == total and sha256_file(output) == digest.hexdigest():
                result["output"] = str(output)
                print(json.dumps(result, indent=2, sort_keys=True))
                return
            raise FileExistsError(output)
        output.parent.mkdir(parents=True, exist_ok=True)
        temporary = output.with_suffix(output.suffix + ".tmp")
        if temporary.exists():
            temporary.unlink()
        with temporary.open("wb") as target:
            for record in parts:
                with (root / record["path"]).open("rb") as source:
                    shutil.copyfileobj(source, target, length=CHUNK_BYTES)
            target.flush()
            os.fsync(target.fileno())
        if temporary.stat().st_size != total or sha256_file(temporary) != digest.hexdigest():
            temporary.unlink(missing_ok=True)
            raise RuntimeError("materialized artifact failed integrity verification")
        os.replace(temporary, output)
        result["output"] = str(output)
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
