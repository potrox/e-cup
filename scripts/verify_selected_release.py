from __future__ import annotations

import argparse
import hashlib
import json
import zipfile
from pathlib import Path
from typing import Any


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Verify organizer-selected Task 1 archives")
    parser.add_argument("--registry", type=Path, default=Path("SELECTED_RELEASES.json"))
    parser.add_argument("--archive-dir", type=Path, required=True)
    parser.add_argument("--profile-root", type=Path, default=Path("release_profiles"))
    parser.add_argument("--root", type=Path, default=Path.cwd())
    parser.add_argument(
        "--release",
        action="append",
        dest="releases",
        help="Verify only the named archive; may be repeated",
    )
    return parser.parse_args()


def _sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _sha256(path: Path, chunk_size: int = 8 * 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


def _object(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"expected JSON object: {path}")
    return value


def main() -> None:
    args = _parse_args()
    root = args.root.resolve(strict=True)
    registry_path = args.registry if args.registry.is_absolute() else root / args.registry
    archive_dir = args.archive_dir.resolve(strict=True)
    profile_root = (
        args.profile_root if args.profile_root.is_absolute() else root / args.profile_root
    )
    registry = _object(registry_path.resolve(strict=True))
    if registry.get("schema_version") != 1:
        raise ValueError("selected-release registry schema_version must be 1")
    selected = registry.get("selected_releases", [])
    if args.releases:
        requested = set(args.releases)
        selected = [release for release in selected if release["archive"] in requested]
        found = {release["archive"] for release in selected}
        missing = sorted(requested.difference(found))
        if missing:
            raise ValueError(f"unknown selected release(s): {missing}")
    results = []
    for release in selected:
        archive_path = archive_dir / release["archive"]
        if not archive_path.is_file():
            raise FileNotFoundError(archive_path)
        if archive_path.stat().st_size != release["bytes"]:
            raise ValueError(f"archive byte count mismatch: {archive_path.name}")
        archive_hash = _sha256(archive_path)
        if archive_hash != release["sha256"]:
            raise ValueError(f"archive SHA-256 mismatch: {archive_path.name}")
        with zipfile.ZipFile(archive_path) as archive:
            if archive.testzip() is not None:
                raise ValueError(f"archive CRC failed: {archive_path.name}")
            runtime_manifest = archive.read("ensemble_manifest.json")
        runtime_hash = _sha256_bytes(runtime_manifest)
        if runtime_hash != release["runtime_manifest_sha256"]:
            raise ValueError(f"runtime manifest mismatch: {archive_path.name}")
        profile_manifest = profile_root / release["profile"] / "ensemble_manifest.json"
        if profile_manifest.read_bytes() != runtime_manifest:
            raise ValueError(f"release profile differs from archive: {release['profile']}")
        results.append(
            {
                "profile": release["profile"],
                "archive": archive_path.name,
                "bytes": archive_path.stat().st_size,
                "sha256": archive_hash,
                "runtime_manifest_sha256": runtime_hash,
            }
        )
    print(json.dumps({"verified": results}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
