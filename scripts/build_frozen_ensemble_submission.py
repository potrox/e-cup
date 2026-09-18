from __future__ import annotations

import argparse
import copy
import hashlib
import json
import os
import shutil
import tempfile
import zipfile
from pathlib import Path, PurePosixPath
from typing import Any

from build_submission import _artifact, _validate_zip, _write_deterministic_zip

_MODEL_FILES = (
    "config.json",
    "model.safetensors",
    "tokenizer.json",
    "tokenizer_config.json",
)
_RUNTIME_FILES = (
    "run.py",
    "metadata.json",
    "RUNTIME_DEPENDENCIES.json",
    "ecup_matching/__init__.py",
    "ecup_matching/serialization.py",
    "ecup_matching/neural_inference.py",
    "ecup_matching/ensemble_inference.py",
)
_MAX_ARCHIVE_BYTES = 5_000_000_000


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build an ensemble from a frozen release profile and fresh component archives"
    )
    parser.add_argument("--profile-dir", type=Path, required=True)
    parser.add_argument("--base-archive", type=Path, required=True)
    parser.add_argument("--lamar-archive", type=Path, required=True)
    parser.add_argument("--teacher-archive", type=Path)
    parser.add_argument("--output-path", type=Path, required=True)
    parser.add_argument("--strict-source-hashes", action="store_true")
    parser.add_argument("--check-only", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def _sha256(path: Path, chunk_size: int = 8 * 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


def _read_object(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"expected a JSON object: {path}")
    return value


def _safe_members(archive: zipfile.ZipFile) -> set[str]:
    names: set[str] = set()
    for info in archive.infolist():
        path = PurePosixPath(info.filename)
        if path.is_absolute() or ".." in path.parts or info.filename in names:
            raise ValueError(f"unsafe or duplicate ZIP entry: {info.filename}")
        names.add(info.filename)
    if archive.testzip() is not None:
        raise ValueError("component archive CRC validation failed")
    return names


def _archive_object(archive: zipfile.ZipFile, name: str) -> dict[str, Any]:
    try:
        value = json.loads(archive.read(name).decode("utf-8"))
    except KeyError as error:
        raise ValueError(f"component archive is missing {name}") from error
    if not isinstance(value, dict):
        raise ValueError(f"component archive contains invalid {name}")
    return value


def _extract(archive: zipfile.ZipFile, source: str, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    with archive.open(source) as reader, destination.open("wb") as writer:
        shutil.copyfileobj(reader, writer)


def _copy_component(
    *,
    archive_path: Path,
    expected: dict[str, Any],
    target_root: Path,
    strict_source_hashes: bool,
) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    component, dependencies, metadata, actual_archive = _check_component(
        archive_path=archive_path,
        expected=expected,
        strict_source_hashes=strict_source_hashes,
    )
    source_model = component["model"]
    model_root = target_root / expected["path"]
    with zipfile.ZipFile(archive_path) as archive:
        for filename in _MODEL_FILES:
            _extract(archive, f"neural_model/{filename}", model_root / filename)

    rebuilt = copy.deepcopy(expected)
    rebuilt["model_type"] = source_model.get("model_type")
    rebuilt["source_archive"] = actual_archive
    rebuilt["artifacts"] = [
        _artifact(model_root / filename, target_root) for filename in _MODEL_FILES
    ]
    rebuilt["training_lineage"] = component.get("training_lineage")
    return rebuilt, dependencies, metadata


def _check_component(
    *,
    archive_path: Path,
    expected: dict[str, Any],
    strict_source_hashes: bool,
) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any], dict[str, Any]]:
    if not archive_path.is_file():
        raise FileNotFoundError(archive_path)
    actual_archive = {
        "sha256": _sha256(archive_path),
        "bytes": archive_path.stat().st_size,
    }
    if strict_source_hashes and actual_archive != expected.get("source_archive"):
        raise ValueError(f"component archive hash mismatch for {expected['name']}")
    with zipfile.ZipFile(archive_path) as archive:
        names = _safe_members(archive)
        component = _archive_object(archive, "neural_manifest.json")
        dependencies = _archive_object(archive, "RUNTIME_DEPENDENCIES.json")
        metadata = _archive_object(archive, "metadata.json")
        source_model = component.get("model")
        source_inference = component.get("inference")
        if not isinstance(source_model, dict) or not isinstance(source_inference, dict):
            raise ValueError(f"invalid neural_manifest.json in {archive_path.name}")
        expected_pretrained = expected.get("pretrained_model")
        if source_model.get("pretrained_model") != expected_pretrained:
            raise ValueError(f"pretrained model lineage mismatch for {expected['name']}")
        for field in ("serialization_mode", "serialization_version", "bidirectional"):
            if source_inference.get(field) != expected["inference"].get(field):
                raise ValueError(f"inference contract mismatch for {expected['name']}: {field}")
        for filename in _MODEL_FILES:
            source = f"neural_model/{filename}"
            if source not in names:
                raise ValueError(f"component archive is missing {source}")
    return component, dependencies, metadata, actual_archive


def _copy_vendor(source_archive: Path, target_root: Path) -> None:
    with zipfile.ZipFile(source_archive) as archive:
        names = _safe_members(archive)
        vendor = sorted(
            name for name in names if name.startswith("vendor/") and not name.endswith("/")
        )
        if not vendor:
            raise ValueError("base component archive has no vendored runtime dependencies")
        for name in vendor:
            _extract(archive, name, target_root / PurePosixPath(name))


def _validate_profile(profile_dir: Path, profile: dict[str, Any]) -> list[dict[str, Any]]:
    if profile.get("schema_version") != 1:
        raise ValueError("release profile schema_version must be 1")
    models = profile.get("models")
    if not isinstance(models, list):
        raise ValueError("release profile has no model list")
    names = [model.get("name") for model in models if isinstance(model, dict)]
    if names not in (
        ["rumodernbert_base", "lamar_600m"],
        ["rumodernbert_base", "lamar_600m", "teacher_student"],
    ):
        raise ValueError(f"unsupported release-profile model order: {names}")
    for relative in _RUNTIME_FILES:
        if not (profile_dir / relative).is_file():
            raise FileNotFoundError(profile_dir / relative)
    return models


def main() -> None:
    args = _parse_args()
    if args.output_path.suffix.lower() != ".zip":
        raise ValueError("output path must end with .zip")
    if args.output_path.exists() and not args.overwrite:
        raise FileExistsError(args.output_path)
    profile_dir = args.profile_dir.resolve(strict=True)
    profile = _read_object(profile_dir / "ensemble_manifest.json")
    expected_models = _validate_profile(profile_dir, profile)
    archive_by_name = {
        "rumodernbert_base": args.base_archive.resolve(strict=True),
        "lamar_600m": args.lamar_archive.resolve(strict=True),
    }
    if len(expected_models) == 3:
        if args.teacher_archive is None:
            raise ValueError("this release profile requires --teacher-archive")
        archive_by_name["teacher_student"] = args.teacher_archive.resolve(strict=True)
    elif args.teacher_archive is not None:
        raise ValueError("this release profile does not use a teacher component")

    if args.check_only:
        checked = []
        images = set()
        digests = set()
        for expected in expected_models:
            _, dependencies, metadata, archive_record = _check_component(
                archive_path=archive_by_name[expected["name"]],
                expected=expected,
                strict_source_hashes=args.strict_source_hashes,
            )
            images.add(metadata.get("image"))
            digests.add(dependencies.get("container_digest"))
            checked.append({"name": expected["name"], **archive_record})
        if len(images) != 1 or len(digests) != 1:
            raise ValueError("component archives use different runtime containers")
        print(json.dumps({"profile": profile_dir.name, "checked": checked}, indent=2))
        return

    args.output_path.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix="ecup-frozen-", dir=args.output_path.parent))
    temporary_zip = args.output_path.with_name(f".{args.output_path.name}.{os.getpid()}.tmp")
    try:
        for relative in _RUNTIME_FILES:
            destination = staging / relative
            destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(profile_dir / relative, destination)
        _copy_vendor(archive_by_name["rumodernbert_base"], staging)

        rebuilt_models: list[dict[str, Any]] = []
        dependency_records: list[dict[str, Any]] = []
        metadata_records: list[dict[str, Any]] = []
        for expected in expected_models:
            rebuilt, dependencies, metadata = _copy_component(
                archive_path=archive_by_name[expected["name"]],
                expected=expected,
                target_root=staging,
                strict_source_hashes=args.strict_source_hashes,
            )
            rebuilt_models.append(rebuilt)
            dependency_records.append(dependencies)
            metadata_records.append(metadata)

        images = {record.get("image") for record in metadata_records}
        digests = {record.get("container_digest") for record in dependency_records}
        if len(images) != 1 or len(digests) != 1:
            raise ValueError("component archives use different runtime containers")

        rebuilt_profile = copy.deepcopy(profile)
        rebuilt_profile["release_name"] = args.output_path.stem
        rebuilt_profile["models"] = rebuilt_models
        (staging / "ensemble_manifest.json").write_text(
            json.dumps(rebuilt_profile, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
            newline="\n",
        )

        dependencies = _read_object(staging / "RUNTIME_DEPENDENCIES.json")
        dependencies["source_archives"] = {
            model["name"]: model["source_archive"] for model in rebuilt_models
        }
        (staging / "RUNTIME_DEPENDENCIES.json").write_text(
            json.dumps(dependencies, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
            newline="\n",
        )

        _write_deterministic_zip(staging, temporary_zip)
        names = _validate_zip(temporary_zip)
        if temporary_zip.stat().st_size > _MAX_ARCHIVE_BYTES:
            raise ValueError("ensemble archive exceeds the official 5 GB limit")
        if args.output_path.exists():
            args.output_path.unlink()
        os.replace(temporary_zip, args.output_path)
        sidecar = {
            "schema_version": 1,
            "archive": {
                "name": args.output_path.name,
                "bytes": args.output_path.stat().st_size,
                "sha256": _sha256(args.output_path),
                "entries": len(names),
            },
            "profile_sha256": _sha256(profile_dir / "ensemble_manifest.json"),
            "components": {
                model["name"]: model["source_archive"] for model in rebuilt_models
            },
        }
        args.output_path.with_suffix(".manifest.json").write_text(
            json.dumps(sidecar, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
            newline="\n",
        )
        print(args.output_path)
    finally:
        if temporary_zip.exists():
            temporary_zip.unlink()
        if staging.exists():
            shutil.rmtree(staging)


if __name__ == "__main__":
    main()
