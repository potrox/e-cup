from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Any

import pyarrow.parquet as pq


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Validate and execute an immutable neural-training configuration"
    )
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--root", type=Path, default=Path.cwd())
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--check-only", action="store_true")
    return parser.parse_args()


def _sha256(path: Path, chunk_size: int = 8 * 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


def _load_object(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"expected a JSON object: {path}")
    return value


def _resolve(root: Path, value: str) -> Path:
    path = Path(value)
    return path if path.is_absolute() else root / path


def _verify_inputs(root: Path, records: object) -> list[dict[str, object]]:
    if not isinstance(records, list) or not records:
        raise ValueError("expected_inputs must be a non-empty list")
    results: list[dict[str, object]] = []
    for record in records:
        if not isinstance(record, dict):
            raise ValueError("each expected input must be a JSON object")
        relative = record.get("path")
        expected_hash = record.get("sha256")
        if not isinstance(relative, str) or not isinstance(expected_hash, str):
            raise ValueError("each expected input requires path and sha256")
        path = _resolve(root, relative)
        if not path.is_file():
            raise FileNotFoundError(path)
        actual_hash = _sha256(path)
        if actual_hash != expected_hash:
            raise ValueError(f"input SHA-256 mismatch: {relative}")
        result: dict[str, object] = {
            "path": relative,
            "bytes": path.stat().st_size,
            "sha256": actual_hash,
        }
        expected_rows = record.get("rows")
        if expected_rows is not None:
            actual_rows = pq.ParquetFile(path).metadata.num_rows
            if actual_rows != int(expected_rows):
                raise ValueError(
                    f"input row count mismatch for {relative}: {actual_rows} != {expected_rows}"
                )
            result["rows"] = actual_rows
        results.append(result)
    return results


def _source_root(root: Path, config: dict[str, Any]) -> Path:
    profile = config.get("source_profile", "current")
    if profile == "current":
        return root
    if not isinstance(profile, str) or not profile or Path(profile).name != profile:
        raise ValueError("source_profile must be 'current' or a simple directory name")
    return root / "source_profiles" / profile


def _model_path(root: Path, model: object) -> Path:
    if isinstance(model, str):
        return _resolve(root, model)
    if not isinstance(model, dict) or not isinstance(model.get("local_path"), str):
        raise ValueError("model must be a path or an object containing local_path")
    return _resolve(root, model["local_path"])


def _append_argument(command: list[str], name: str, value: object) -> None:
    option = "--" + name.replace("_", "-")
    if isinstance(value, bool):
        if value:
            command.append(option)
        return
    if value is None:
        return
    if isinstance(value, (str, int, float, os.PathLike)):
        command.extend((option, str(value)))
        return
    raise ValueError(f"unsupported argument value for {name}: {value!r}")


def build_command(root: Path, config: dict[str, Any]) -> tuple[list[str], Path]:
    if config.get("schema_version") != 2:
        raise ValueError("training config schema_version must be 2")
    source_root = _source_root(root, config)
    entry_point = config.get("entry_point")
    if not isinstance(entry_point, str):
        raise ValueError("entry_point is required")
    script = _resolve(source_root, entry_point)
    if not script.is_file():
        raise FileNotFoundError(script)

    required = ("model", "train", "validation", "validation_kind", "output_dir")
    missing = [name for name in required if name not in config]
    if missing:
        raise ValueError(f"training config is missing fields: {missing}")
    command = [sys.executable, str(script)]
    fixed = {
        "model": _model_path(root, config["model"]),
        "train": _resolve(root, str(config["train"])),
        "validation": _resolve(root, str(config["validation"])),
        "validation_kind": config["validation_kind"],
        "output_dir": _resolve(root, str(config["output_dir"])),
    }
    for name, value in fixed.items():
        _append_argument(command, name, value)
    for name in ("replay_train", "replay_fraction"):
        if name in config:
            value = config[name]
            if name.endswith("_train"):
                value = _resolve(root, str(value))
            _append_argument(command, name, value)
    arguments = config.get("arguments", {})
    if not isinstance(arguments, dict):
        raise ValueError("arguments must be a JSON object")
    for name, value in arguments.items():
        _append_argument(command, name, value)
    return command, source_root


def main() -> None:
    args = _parse_args()
    root = args.root.resolve(strict=True)
    config_path = args.config if args.config.is_absolute() else root / args.config
    config = _load_object(config_path.resolve(strict=True))
    command, source_root = build_command(root, config)
    checked_inputs: list[dict[str, object]] = []
    checked_sources: list[dict[str, object]] = []
    if not args.dry_run:
        checked_inputs = _verify_inputs(root, config.get("expected_inputs"))
        checked_sources = _verify_inputs(root, config.get("expected_source_files"))
        model = _model_path(root, config["model"])
        if not model.is_dir():
            raise FileNotFoundError(model)
    payload = {
        "config": config_path.relative_to(root).as_posix(),
        "source_profile": config.get("source_profile", "current"),
        "command": command,
        "checked_inputs": checked_inputs,
        "checked_source_files": checked_sources,
        "execute": not (args.dry_run or args.check_only),
    }
    print(json.dumps(payload, ensure_ascii=False, indent=2))
    if args.dry_run or args.check_only:
        return
    environment = os.environ.copy()
    python_paths = [str(source_root / "src")]
    if source_root != root:
        python_paths.append(str(root / "src"))
    if environment.get("PYTHONPATH"):
        python_paths.append(environment["PYTHONPATH"])
    environment["PYTHONPATH"] = os.pathsep.join(python_paths)
    subprocess.run(command, cwd=root, env=environment, check=True)


if __name__ == "__main__":
    main()
