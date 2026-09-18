from __future__ import annotations

import argparse
import json
import os
import shutil
import tempfile
from datetime import datetime
from pathlib import Path
from typing import Any

from build_submission import _artifact, _sha256, _validate_zip, _write_deterministic_zip

from ecup_matching.serialization import ITEM_SERIALIZER_VERSION, PAIR_SERIALIZER_VERSION

_DEFAULT_IMAGE = "odsai/ecup26-matching-baseline:1.0"
_ENTRY_POINT = "python -u run.py"
_MODEL_ARTIFACT_NAMES = (
    "config.json",
    "model.safetensors",
    "tokenizer.json",
    "tokenizer_config.json",
)
_SUPPORTED_MODEL_ARCHITECTURES = {
    "modernbert": "ModernBertForSequenceClassification",
    "xlm-roberta": "XLMRobertaForSequenceClassification",
}


def _parse_args() -> argparse.Namespace:
    project_root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description="Build a clean neural-only submission ZIP")
    parser.add_argument("--neural-model-dir", type=Path, required=True)
    parser.add_argument(
        "--lineage-manifest",
        type=Path,
        action="append",
        required=True,
        help="Ordered training manifests from base-model stage to final gold stage",
    )
    parser.add_argument(
        "--runtime-dependencies",
        type=Path,
        default=project_root / "submission_runtime/RUNTIME_DEPENDENCIES.json",
    )
    parser.add_argument(
        "--vendor-dir",
        type=Path,
        default=project_root / "artifacts/runtime_vendor/base_py312_v1",
    )
    parser.add_argument("--output-path", type=Path, required=True)
    parser.add_argument("--image", default=_DEFAULT_IMAGE)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument(
        "--bidirectional",
        action=argparse.BooleanOptionalAction,
        default=False,
    )
    parser.add_argument(
        "--serialization-mode",
        choices=("item_v1", "pair_v2"),
        default="pair_v2",
    )
    parser.add_argument("--max-length", type=int, default=256)
    parser.add_argument("--max-attribute-characters", type=int, default=2048)
    parser.add_argument("--serialization-chunk-size", type=int, default=8192)
    parser.add_argument("--length-bucketing", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def _validate_neural_model(path: Path) -> tuple[tuple[Path, ...], str]:
    required = tuple(path / name for name in _MODEL_ARTIFACT_NAMES)
    for artifact in required:
        if not artifact.is_file():
            raise FileNotFoundError(artifact)
    unexpected = sorted(
        candidate.name
        for candidate in path.iterdir()
        if candidate.is_file() and candidate.name not in _MODEL_ARTIFACT_NAMES
    )
    if unexpected:
        raise ValueError(f"neural checkpoint has unexpected files: {unexpected}")
    config = json.loads((path / "config.json").read_text(encoding="utf-8"))
    model_type = config.get("model_type")
    label_count = len(config.get("id2label", {}))
    expected_architecture = _SUPPORTED_MODEL_ARCHITECTURES.get(model_type)
    architectures = config.get("architectures")
    if (
        expected_architecture is None
        or architectures != [expected_architecture]
        or label_count != 1
    ):
        raise ValueError(
            "neural checkpoint is not a supported single-logit sequence classifier"
        )
    return required, model_type


def _load_runtime_dependencies(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(path)
    manifest = json.loads(path.read_text(encoding="utf-8"))
    pretrained_model = manifest.get("pretrained_model")
    if not isinstance(pretrained_model, dict):
        raise ValueError("runtime dependencies must declare pretrained_model provenance")
    required = ("name", "source", "revision", "license", "license_evidence")
    missing = [field for field in required if not pretrained_model.get(field)]
    if missing:
        raise ValueError(f"pretrained_model provenance is missing: {', '.join(missing)}")
    for field in ("container_digest", "container_license", "container_dependencies"):
        if not manifest.get(field):
            raise ValueError(f"runtime dependencies are missing: {field}")
    return manifest


def _validate_orjson_vendor(path: Path) -> tuple[Path, Path]:
    package = path / "orjson"
    if not package.is_dir():
        raise FileNotFoundError(package)
    binaries = list(package.glob("orjson.cpython-312-*-linux-gnu.so"))
    if len(binaries) != 1:
        raise ValueError("vendored orjson must contain one CPython 3.12 Linux binary")
    metadata_dirs = sorted(path.glob("orjson-*.dist-info"))
    if len(metadata_dirs) != 1:
        raise ValueError("vendored orjson must contain one dist-info directory")
    license_dir = metadata_dirs[0] / "licenses"
    if not license_dir.is_dir() or not any(license_dir.iterdir()):
        raise ValueError("vendored orjson license evidence is missing")
    return package, metadata_dirs[0]


def _model_source_matches_pretrained(source: str, pretrained: dict[str, Any]) -> bool:
    normalized_source = source.replace("\\", "/").lower()
    name = str(pretrained["name"]).lower()
    cache_name = f"models--{name.replace('/', '--')}"
    revision = str(pretrained["revision"]).lower()
    return (name in normalized_source or cache_name in normalized_source) and (
        revision in normalized_source
    )


def _checkpoint_artifacts(manifest: dict[str, Any]) -> dict[str, dict[str, Any]]:
    artifacts = manifest.get("best_checkpoint_artifacts")
    if not isinstance(artifacts, list) or not artifacts:
        raise ValueError("training manifest is missing best_checkpoint_artifacts")
    result: dict[str, dict[str, Any]] = {}
    for artifact in artifacts:
        name = Path(str(artifact["path"])).name
        if name in result:
            raise ValueError(f"duplicate checkpoint artifact in training manifest: {name}")
        result[name] = artifact
    return result


def _resolved_manifest_path(value: str, manifest_path: Path) -> Path:
    candidate = Path(value)
    if not candidate.is_absolute():
        candidate = manifest_path.parent / candidate
    return candidate.resolve()


def _portable_lineage_entry(path: Path, manifest: dict[str, Any]) -> dict[str, Any]:
    config = manifest.get("config")
    if not isinstance(config, dict):
        raise ValueError(f"training manifest config is missing: {path}")
    train = manifest.get("train")
    validation = manifest.get("validation")
    if not isinstance(train, dict) or not isinstance(validation, dict):
        raise ValueError(f"training data provenance is missing: {path}")
    replay = manifest.get("replay")
    if replay is not None and not isinstance(replay, dict):
        raise ValueError(f"training replay provenance is invalid: {path}")
    return {
        "manifest_sha256": _sha256(path),
        "best_epoch": manifest.get("best_epoch"),
        "checkpoint_selection": config.get("checkpoint_selection"),
        "serializer": {
            "mode": config.get("serialization_mode", "item_v1"),
            "version": manifest.get("serialization_version"),
            "max_length": config.get("max_length"),
            "max_attribute_characters": config.get("max_attribute_characters"),
        },
        "objective": {
            "loss_mode": config.get("loss_mode", "row_mean"),
            "confidence_gamma": config.get("confidence_gamma", 0.0),
            "hard_replay_fraction": config.get("hard_replay_fraction", 0.0),
        },
        "seed": config.get("seed"),
        "train": {
            "sha256": train.get("sha256"),
            "available_rows": train.get("available_rows"),
            "used_rows": train.get("used_rows"),
            "include_fold": config.get("train_include_fold"),
            "exclude_fold": config.get("train_exclude_fold"),
            "include_inner_fold": config.get("train_include_inner_fold"),
            "exclude_inner_fold": config.get("train_exclude_inner_fold"),
        },
        "replay": (
            {
                "sha256": replay.get("sha256"),
                "available_rows": replay.get("available_rows"),
                "planned_rows_per_epoch": replay.get("planned_rows_per_epoch"),
                "used_rows": replay.get("used_rows"),
                "fraction": replay.get("fraction"),
            }
            if replay is not None
            else None
        ),
        "validation": {
            "sha256": validation.get("sha256"),
            "kind": config.get("validation_kind"),
            "include_fold": config.get("validation_include_fold"),
            "exclude_fold": config.get("validation_exclude_fold"),
            "include_inner_fold": config.get("validation_include_inner_fold"),
            "exclude_inner_fold": config.get("validation_exclude_inner_fold"),
        },
        "training_code": manifest.get("training_code"),
    }


def _validate_training_lineage(
    manifest_paths: list[Path],
    *,
    final_model_dir: Path,
    pretrained_model: dict[str, Any],
) -> list[dict[str, Any]]:
    if not manifest_paths:
        raise ValueError("at least one lineage manifest is required")
    loaded: list[tuple[Path, dict[str, Any]]] = []
    for path in manifest_paths:
        if not path.is_file():
            raise FileNotFoundError(path)
        manifest = json.loads(path.read_text(encoding="utf-8"))
        if manifest.get("schema_version") != 1:
            raise ValueError(f"unsupported training manifest schema: {path}")
        loaded.append((path.resolve(), manifest))

    first_path, first = loaded[0]
    first_source = str(first.get("model_source", ""))
    if not _model_source_matches_pretrained(first_source, pretrained_model):
        raise ValueError(
            "first training stage does not match pinned pretrained model name/revision"
        )

    previous_checkpoint: Path | None = None
    for index, (path, manifest) in enumerate(loaded):
        checkpoint_value = manifest.get("best_checkpoint")
        model_source_value = manifest.get("model_source")
        if not checkpoint_value or not model_source_value:
            raise ValueError(f"training lineage paths are missing: {path}")
        checkpoint = _resolved_manifest_path(str(checkpoint_value), path)
        model_source = _resolved_manifest_path(str(model_source_value), path)
        if index > 0 and model_source != previous_checkpoint:
            raise ValueError(
                f"training lineage is disconnected at stage {index + 1}: "
                f"{model_source} != {previous_checkpoint}"
            )
        artifacts = _checkpoint_artifacts(manifest)
        for required_name in _MODEL_ARTIFACT_NAMES:
            try:
                artifact = artifacts[required_name]
            except KeyError as error:
                raise ValueError(
                    f"training manifest lacks checkpoint artifact: {required_name}"
                ) from error
            artifact_path = _resolved_manifest_path(str(artifact["path"]), path)
            if artifact_path.parent != checkpoint:
                raise ValueError(f"checkpoint artifact is outside best checkpoint: {artifact_path}")
            if not artifact_path.is_file():
                raise FileNotFoundError(artifact_path)
            if artifact_path.stat().st_size != int(artifact["bytes"]):
                raise ValueError(f"checkpoint artifact size mismatch: {artifact_path}")
            if _sha256(artifact_path) != str(artifact["sha256"]):
                raise ValueError(f"checkpoint artifact SHA-256 mismatch: {artifact_path}")
        previous_checkpoint = checkpoint

    if previous_checkpoint != final_model_dir.resolve():
        raise ValueError(
            f"final lineage checkpoint does not match release model: "
            f"{previous_checkpoint} != {final_model_dir.resolve()}"
        )
    return [_portable_lineage_entry(path, manifest) for path, manifest in loaded]


def _validate_release_training_contract(
    lineage: list[dict[str, Any]],
    *,
    serialization_mode: str,
    serialization_version: str,
    max_length: int,
    max_attribute_characters: int,
) -> None:
    expected = {
        "mode": serialization_mode,
        "version": serialization_version,
        "max_length": max_length,
        "max_attribute_characters": max_attribute_characters,
    }
    for index, stage in enumerate(lineage, start=1):
        serializer = stage.get("serializer")
        if serializer != expected:
            raise ValueError(
                f"release/training serializer contract mismatch at stage {index}: "
                f"{serializer} != {expected}"
            )


def main() -> None:
    args = _parse_args()
    project_root = Path(__file__).resolve().parents[1]
    if args.output_path.suffix.lower() != ".zip":
        raise ValueError("output path must end with .zip")
    if args.output_path.exists() and not args.overwrite:
        raise FileExistsError(args.output_path)
    if not args.image.strip() or args.batch_size < 1:
        raise ValueError("image and batch size must be valid")
    if args.max_length < 8 or args.max_attribute_characters < 1:
        raise ValueError("neural text limits are invalid")
    if args.serialization_mode == "pair_v2" and args.bidirectional:
        raise ValueError("pair_v2 is canonical and cannot use bidirectional inference")
    if args.serialization_chunk_size < args.batch_size:
        raise ValueError("serialization chunk size must be at least batch size")

    neural_artifacts, model_type = _validate_neural_model(args.neural_model_dir)
    dependencies = _load_runtime_dependencies(args.runtime_dependencies)
    orjson_package, orjson_metadata = _validate_orjson_vendor(args.vendor_dir)
    pretrained_model = dict(dependencies["pretrained_model"])
    pretrained_model.pop("derivative_training", None)
    lineage = _validate_training_lineage(
        args.lineage_manifest,
        final_model_dir=args.neural_model_dir,
        pretrained_model=pretrained_model,
    )
    serialization_version = (
        PAIR_SERIALIZER_VERSION if args.serialization_mode == "pair_v2" else ITEM_SERIALIZER_VERSION
    )
    _validate_release_training_contract(
        lineage,
        serialization_mode=args.serialization_mode,
        serialization_version=serialization_version,
        max_length=args.max_length,
        max_attribute_characters=args.max_attribute_characters,
    )

    runtime_sources = {
        project_root / "submission_runtime/neural_only_run.py": Path("run.py"),
        project_root / "submission_runtime/ecup_matching/__init__.py": Path(
            "ecup_matching/__init__.py"
        ),
        project_root / "src/ecup_matching/serialization.py": Path("ecup_matching/serialization.py"),
        project_root / "src/ecup_matching/neural_inference.py": Path(
            "ecup_matching/neural_inference.py"
        ),
    }
    for source in runtime_sources:
        if not source.is_file():
            raise FileNotFoundError(source)

    args.output_path.parent.mkdir(parents=True, exist_ok=True)
    staging_dir = Path(tempfile.mkdtemp(prefix="ecup-neural-", dir=args.output_path.parent))
    temporary_zip = args.output_path.with_name(f".{args.output_path.name}.{os.getpid()}.tmp")
    try:
        for source, relative in runtime_sources.items():
            destination = staging_dir / relative
            destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(source, destination)
        model_target = staging_dir / "neural_model"
        model_target.mkdir()
        for source in neural_artifacts:
            shutil.copy2(source, model_target / source.name)
        vendor_target = staging_dir / "vendor"
        shutil.copytree(
            orjson_package,
            vendor_target / orjson_package.name,
            ignore=shutil.ignore_patterns("__pycache__", "*.pyc"),
        )
        shutil.copytree(
            orjson_metadata,
            vendor_target / orjson_metadata.name,
            ignore=shutil.ignore_patterns("__pycache__", "*.pyc"),
        )

        model_entries = [
            _artifact(model_target / source.name, staging_dir) for source in neural_artifacts
        ]
        neural_manifest = {
            "schema_version": 1,
            "release_name": "ecup-task1-neural-model-first-v2",
            "model": {
                "path": "neural_model",
                "model_type": model_type,
                "pretrained_model": pretrained_model,
                "artifacts": model_entries,
            },
            "inference": {
                "serialization_mode": args.serialization_mode,
                "serialization_version": serialization_version,
                "max_length": args.max_length,
                "max_attribute_characters": args.max_attribute_characters,
                "batch_size": args.batch_size,
                "serialization_chunk_size": args.serialization_chunk_size,
                "bidirectional": args.bidirectional,
                "length_bucketing": args.length_bucketing,
                "score": (
                    "mean of forward and reverse raw classifier logits"
                    if args.bidirectional
                    else "single raw classifier logit"
                ),
            },
            "training_lineage": lineage,
        }
        (staging_dir / "neural_manifest.json").write_text(
            json.dumps(neural_manifest, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        runtime_dependencies = {
            "schema_version": 2,
            "container_image": args.image,
            "container_digest": dependencies["container_digest"],
            "container_license": dependencies["container_license"],
            "container_dependencies": dependencies["container_dependencies"],
            "vendored_dependencies": [
                dependency
                for dependency in dependencies.get("vendored_dependencies", [])
                if dependency.get("name") == "orjson"
            ],
            "archive_dependencies": [
                {
                    "name": "ecup_matching neural runtime",
                    "origin": "project-owned",
                    "license": "competition submission code",
                },
                {
                    "name": f"single-logit {model_type} sequence-classification checkpoint",
                    "origin": "derived from organizer labels",
                    "license": f"{pretrained_model['license']} base plus dataset-derived weights",
                },
            ],
            "checkpoint_model_type": model_type,
            "pretrained_model": pretrained_model,
            "training_lineage": lineage,
        }
        (staging_dir / "RUNTIME_DEPENDENCIES.json").write_text(
            json.dumps(runtime_dependencies, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        (staging_dir / "metadata.json").write_text(
            json.dumps({"image": args.image, "entry_point": _ENTRY_POINT}, indent=2) + "\n",
            encoding="utf-8",
        )

        _write_deterministic_zip(staging_dir, temporary_zip)
        names = _validate_zip(temporary_zip)
        if args.output_path.exists():
            args.output_path.unlink()
        os.replace(temporary_zip, args.output_path)
        release_manifest = {
            "schema_version": 1,
            "created_at": datetime.now().astimezone().isoformat(),
            "archive": {
                "path": str(args.output_path.resolve()),
                "bytes": args.output_path.stat().st_size,
                "sha256": _sha256(args.output_path),
                "entries": len(names),
            },
            "container_image": args.image,
            "entry_point": _ENTRY_POINT,
            "contents": [
                _artifact(path, staging_dir)
                for path in sorted(staging_dir.rglob("*"))
                if path.is_file()
            ],
        }
        manifest_path = args.output_path.with_suffix(".manifest.json")
        manifest_path.write_text(
            json.dumps(release_manifest, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        print(args.output_path)
        print(manifest_path)
    finally:
        if temporary_zip.exists():
            temporary_zip.unlink()
        if staging_dir.exists():
            shutil.rmtree(staging_dir)


if __name__ == "__main__":
    main()
