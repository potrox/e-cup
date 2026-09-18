"""Build and validate a minimal, deterministic Task 1 submission ZIP."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import tempfile
import zipfile
from datetime import datetime
from pathlib import Path
from typing import Any

_DEFAULT_IMAGE = "odsai/ecup26-matching-baseline:1.0"
_ENTRY_POINT = "python -u run.py"
_ZIP_TIMESTAMP = (2026, 8, 22, 0, 0, 0)


def _sha256(path: Path, chunk_size: int = 8 * 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        while chunk := source.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


def _artifact(path: Path, root: Path) -> dict[str, object]:
    return {
        "path": path.relative_to(root).as_posix(),
        "bytes": path.stat().st_size,
        "sha256": _sha256(path),
    }


def _validate_model_artifacts(model_dir: Path) -> dict[str, Any]:
    manifest_path = model_dir / "model_manifest.json"
    if not manifest_path.is_file():
        raise FileNotFoundError(manifest_path)
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("schema_version") != 1:
        raise ValueError("unsupported model manifest schema_version")
    artifact_records = [manifest["artifacts"]["global_model"]]
    artifact_records.extend(manifest["artifacts"]["category_models"].values())
    for artifact in artifact_records:
        path = (model_dir / artifact["path"]).resolve()
        try:
            path.relative_to(model_dir.resolve())
        except ValueError as error:
            raise ValueError(f"model artifact escapes model directory: {path}") from error
        if not path.is_file() or _sha256(path) != artifact["sha256"]:
            raise ValueError(f"model artifact failed integrity check: {path}")
    return manifest


def _write_deterministic_zip(staging_dir: Path, destination: Path) -> None:
    with zipfile.ZipFile(
        destination,
        mode="w",
        compression=zipfile.ZIP_DEFLATED,
        compresslevel=9,
        allowZip64=True,
    ) as archive:
        for path in sorted(staging_dir.rglob("*")):
            if not path.is_file():
                continue
            relative = path.relative_to(staging_dir).as_posix()
            info = zipfile.ZipInfo(relative, date_time=_ZIP_TIMESTAMP)
            info.compress_type = zipfile.ZIP_DEFLATED
            info._compresslevel = 9
            info.external_attr = 0o100644 << 16
            info.file_size = path.stat().st_size
            with path.open("rb") as source, archive.open(info, "w", force_zip64=True) as target:
                shutil.copyfileobj(source, target, length=8 * 1024 * 1024)


def _validate_zip(path: Path) -> list[str]:
    with zipfile.ZipFile(path) as archive:
        if archive.testzip() is not None:
            raise ValueError("ZIP CRC validation failed")
        names = archive.namelist()
        if len(names) != len(set(names)):
            raise ValueError("ZIP contains duplicate paths")
        for name in names:
            pure = Path(name)
            if pure.is_absolute() or ".." in pure.parts or "\\" in name:
                raise ValueError(f"unsafe ZIP path: {name}")
        if "metadata.json" not in names or "run.py" not in names:
            raise ValueError("metadata.json and run.py must be at ZIP root")
        metadata = json.loads(archive.read("metadata.json"))
        if set(metadata) != {"image", "entry_point"}:
            raise ValueError("metadata.json has an unexpected schema")
        if metadata["entry_point"] != _ENTRY_POINT:
            raise ValueError("metadata.json entry_point is invalid")
        forbidden_parts = {
            "__pycache__",
            ".cache",
            ".venv",
            ".git",
            "artifacts",
            "data_derived",
            "tests",
        }
        for name in names:
            if forbidden_parts.intersection(Path(name).parts):
                raise ValueError(f"forbidden development path in ZIP: {name}")
            if Path(name).suffix in {".parquet", ".ipynb", ".log", ".pyc"}:
                raise ValueError(f"forbidden development file in ZIP: {name}")
    return names


def build_submission(
    *,
    project_root: Path,
    model_dir: Path,
    output_path: Path,
    image: str,
    vendor_dir: Path | None,
    overwrite: bool,
) -> tuple[Path, Path]:
    if not image.strip():
        raise ValueError("container image must not be empty")
    if output_path.suffix.lower() != ".zip":
        raise ValueError("output path must end with .zip")
    if output_path.exists() and not overwrite:
        raise FileExistsError(output_path)
    _validate_model_artifacts(model_dir)

    runtime_dir = project_root / "submission_runtime"
    runtime_sources = {
        runtime_dir / "run.py": Path("run.py"),
        runtime_dir / "RUNTIME_DEPENDENCIES.json": Path("RUNTIME_DEPENDENCIES.json"),
        runtime_dir / "ecup_matching/__init__.py": Path("ecup_matching/__init__.py"),
        project_root / "src/ecup_matching/features.py": Path("ecup_matching/features.py"),
        project_root / "src/ecup_matching/inference.py": Path("ecup_matching/inference.py"),
    }
    for source in runtime_sources:
        if not source.is_file():
            raise FileNotFoundError(source)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    staging_dir = Path(tempfile.mkdtemp(prefix="ecup-submit-", dir=output_path.parent))
    temporary_zip = output_path.with_name(f".{output_path.name}.{os.getpid()}.tmp")
    try:
        for source, relative in runtime_sources.items():
            destination = staging_dir / relative
            destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(source, destination)
        shutil.copy2(model_dir / "model_manifest.json", staging_dir / "model_manifest.json")
        shutil.copytree(model_dir / "models", staging_dir / "models")
        if vendor_dir is not None:
            if not vendor_dir.is_dir():
                raise FileNotFoundError(vendor_dir)
            shutil.copytree(
                vendor_dir,
                staging_dir / "vendor",
                ignore=shutil.ignore_patterns("__pycache__", "*.pyc"),
            )

        metadata = {"image": image, "entry_point": _ENTRY_POINT}
        (staging_dir / "metadata.json").write_text(
            json.dumps(metadata, indent=2) + "\n",
            encoding="utf-8",
        )
        _write_deterministic_zip(staging_dir, temporary_zip)
        names = _validate_zip(temporary_zip)
        if output_path.exists():
            if not overwrite:
                raise FileExistsError(output_path)
            output_path.unlink()
        os.replace(temporary_zip, output_path)

        release_manifest = {
            "schema_version": 1,
            "created_at": datetime.now().astimezone().isoformat(),
            "archive": {
                "path": str(output_path),
                "bytes": output_path.stat().st_size,
                "sha256": _sha256(output_path),
                "entries": len(names),
            },
            "container_image": image,
            "entry_point": _ENTRY_POINT,
            "vendor_included": vendor_dir is not None,
            "contents": [
                _artifact(path, staging_dir)
                for path in sorted(staging_dir.rglob("*"))
                if path.is_file()
            ],
        }
        release_manifest_path = output_path.with_suffix(".manifest.json")
        release_manifest_path.write_text(
            json.dumps(release_manifest, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        return output_path, release_manifest_path
    finally:
        if temporary_zip.exists():
            temporary_zip.unlink()
        if staging_dir.exists():
            shutil.rmtree(staging_dir)


def _parse_args() -> argparse.Namespace:
    project_root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--model-dir",
        type=Path,
        default=project_root / "artifacts/final_structured_v1",
    )
    parser.add_argument(
        "--output-path",
        type=Path,
        default=project_root / "artifacts/submissions/ECUP2026_task1_structured_v1.zip",
    )
    parser.add_argument("--image", default=_DEFAULT_IMAGE)
    parser.add_argument("--vendor-dir", type=Path)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    project_root = Path(__file__).resolve().parents[1]
    archive_path, manifest_path = build_submission(
        project_root=project_root,
        model_dir=args.model_dir,
        output_path=args.output_path,
        image=args.image,
        vendor_dir=args.vendor_dir,
        overwrite=args.overwrite,
    )
    print(f"wrote {archive_path}", flush=True)
    print(f"wrote {manifest_path}", flush=True)


if __name__ == "__main__":
    main()
