from __future__ import annotations

import argparse
import json
import os
import shutil
import tempfile
import zipfile
from datetime import datetime
from pathlib import Path, PurePosixPath
from typing import Any

from build_submission import _artifact, _sha256, _validate_zip, _write_deterministic_zip

from ecup_matching.release_evidence import (
    resolve_category_router,
    resolve_length_router,
)

_ENTRY_POINT = "python -u run.py"
_MODEL_FILES = (
    "config.json",
    "model.safetensors",
    "tokenizer.json",
    "tokenizer_config.json",
)
_MAX_ARCHIVE_BYTES = 5_000_000_000
_QWEN_README_SHA256 = "c5f5a8c2dddab69cfbf05279235aa5fddb137939a06539c4c7637aa900fef6d0"
_QWEN_LICENSE_SHA256 = "bbedc3fda3305820b977265f01b8619d87570a6739de3a5582c3464840f1e57a"


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build an allowlist neural-ensemble Task 1 submission"
    )
    parser.add_argument("--base-archive", type=Path, required=True)
    parser.add_argument("--lamar-archive", type=Path, required=True)
    parser.add_argument("--teacher-archive", type=Path)
    parser.add_argument("--output-path", type=Path, required=True)
    parser.add_argument("--base-weight", type=float, default=0.55)
    parser.add_argument("--lamar-weight", type=float, default=0.45)
    parser.add_argument("--base-batch-size", type=int, default=512)
    parser.add_argument("--lamar-batch-size", type=int, default=192)
    parser.add_argument("--teacher-batch-size", type=int, default=512)
    parser.add_argument("--length-bucketing", action="store_true")
    parser.add_argument("--category-router-report", type=Path)
    parser.add_argument("--category-router-k", type=int)
    parser.add_argument("--base-length-router-report", type=Path)
    parser.add_argument("--base-routed-max-length", type=int)
    parser.add_argument("--teacher-residual-contract", type=Path)
    parser.add_argument("--teacher-residual-report", type=Path)
    parser.add_argument("--teacher-selection-report", type=Path)
    parser.add_argument("--post-blend-report", type=Path)
    parser.add_argument("--post-blend-route-contract", type=Path)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def _safe_members(archive: zipfile.ZipFile) -> set[str]:
    names: set[str] = set()
    for info in archive.infolist():
        path = PurePosixPath(info.filename)
        if path.is_absolute() or ".." in path.parts or info.filename in names:
            raise ValueError(f"unsafe or duplicate ZIP entry: {info.filename}")
        names.add(info.filename)
    if archive.testzip() is not None:
        raise ValueError("source archive CRC validation failed")
    return names


def _read_json(archive: zipfile.ZipFile, name: str) -> dict[str, Any]:
    try:
        payload = archive.read(name)
    except KeyError as error:
        raise ValueError(f"source archive is missing {name}") from error
    value = json.loads(payload.decode("utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"invalid JSON object: {name}")
    return value


def _extract_file(
    archive: zipfile.ZipFile,
    source: str,
    destination: Path,
) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    with archive.open(source) as reader, destination.open("wb") as writer:
        shutil.copyfileobj(reader, writer)


def _copy_model(
    *,
    archive_path: Path,
    name: str,
    target_root: Path,
    batch_size: int,
    length_bucketing: bool,
) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    if not archive_path.is_file():
        raise FileNotFoundError(archive_path)
    with zipfile.ZipFile(archive_path) as archive:
        names = _safe_members(archive)
        neural_manifest = _read_json(archive, "neural_manifest.json")
        dependencies = _read_json(archive, "RUNTIME_DEPENDENCIES.json")
        metadata = _read_json(archive, "metadata.json")
        inference = neural_manifest.get("inference")
        if not isinstance(inference, dict):
            raise ValueError(f"source inference config is missing: {archive_path}")
        if (
            inference.get("serialization_mode") != "item_v1"
            or inference.get("serialization_version") != "item_v1"
            or inference.get("bidirectional") is not False
        ):
            raise ValueError("ensemble sources must use one-direction item_v1 inference")
        source_model = neural_manifest.get("model")
        if not isinstance(source_model, dict):
            raise ValueError(f"source model manifest is missing: {archive_path}")
        model_target = target_root / "models" / name
        for filename in _MODEL_FILES:
            source_name = f"neural_model/{filename}"
            if source_name not in names:
                raise ValueError(f"source model artifact is missing: {source_name}")
            _extract_file(archive, source_name, model_target / filename)

    model_entry = {
        "name": name,
        "path": f"models/{name}",
        "model_type": source_model.get("model_type"),
        "pretrained_model": source_model.get("pretrained_model"),
        "source_archive": {
            "sha256": _sha256(archive_path),
            "bytes": archive_path.stat().st_size,
        },
        "artifacts": [
            _artifact(model_target / filename, target_root) for filename in _MODEL_FILES
        ],
        "inference": {
            **inference,
            "batch_size": batch_size,
            "bidirectional": False,
            "load_in_bf16": False,
            "length_bucketing": length_bucketing,
            "score": "single raw classifier logit",
        },
        "training_lineage": neural_manifest.get("training_lineage"),
    }
    return model_entry, dependencies, metadata


def _copy_vendor(source_archive: Path, target_root: Path) -> None:
    with zipfile.ZipFile(source_archive) as archive:
        names = _safe_members(archive)
        vendor_files = sorted(
            name for name in names if name.startswith("vendor/") and not name.endswith("/")
        )
        if not vendor_files:
            raise ValueError("source archive has no vendored runtime dependencies")
        for name in vendor_files:
            _extract_file(archive, name, target_root / PurePosixPath(name))


def main() -> None:
    args = _parse_args()
    if args.output_path.suffix.lower() != ".zip":
        raise ValueError("output path must end with .zip")
    if args.output_path.exists() and not args.overwrite:
        raise FileExistsError(args.output_path)
    if min(args.base_batch_size, args.lamar_batch_size, args.teacher_batch_size) < 1:
        raise ValueError("batch sizes must be positive")
    weights = (args.base_weight, args.lamar_weight)
    if any(value < 0.0 for value in weights) or abs(sum(weights) - 1.0) > 1e-12:
        raise ValueError("ensemble weights must be nonnegative and sum to one")
    if (args.category_router_report is None) != (args.category_router_k is None):
        raise ValueError("category router report and K must be provided together")
    if args.category_router_k is not None and not 1 <= args.category_router_k <= 20:
        raise ValueError("category router K must be in [1, 20]")
    if (args.base_length_router_report is None) != (
        args.base_routed_max_length is None
    ):
        raise ValueError(
            "base length router report and routed max length must be provided together"
        )
    teacher_inputs = (
        args.teacher_archive,
        args.teacher_residual_contract,
        args.teacher_residual_report,
        args.teacher_selection_report,
    )
    if any(value is not None for value in teacher_inputs) and not all(
        value is not None for value in teacher_inputs
    ):
        raise ValueError(
            "teacher archive, residual contract, and residual report must be provided together"
        )

    residual = None
    residual_source = None
    if args.teacher_residual_contract is not None:
        contract = json.loads(
            args.teacher_residual_contract.read_text(encoding="utf-8")
        )
        report = json.loads(args.teacher_residual_report.read_text(encoding="utf-8"))
        selection = json.loads(
            args.teacher_selection_report.read_text(encoding="utf-8")
        )
        route = contract.get("route")
        if not isinstance(route, dict):
            raise ValueError("teacher residual contract has no route")
        if report.get("route") != route or report.get("gate", {}).get("pass") is not True:
            raise ValueError("teacher residual report does not pass its immutable contract")
        residual = {
            "model_name": "teacher_student",
            "representation": "category_rank",
            "scope": "category_subset",
            "categories": route["categories"],
            "reference_weight": float(route["reference_weight"]),
            "model_weight": float(route["teacher_student_weight"]),
        }
        residual_weights = (
            residual["reference_weight"],
            residual["model_weight"],
        )
        if any(value < 0.0 for value in residual_weights) or not abs(
            sum(residual_weights) - 1.0
        ) <= 1e-12:
            raise ValueError("teacher residual weights must be nonnegative and sum to one")
        residual_source = {
            "contract_sha256": _sha256(args.teacher_residual_contract),
            "report_sha256": _sha256(args.teacher_residual_report),
            "local_oof": report["candidate"]["score"],
            "uplift_vs_k17": report["candidate_minus_reference"][
                "macro_average_precision"
            ],
            "positive_folds": report["candidate_minus_reference"]["positive_folds"],
            "label_teacher": {
                "name": "Qwen/Qwen3.5-9B",
                "revision": "c202236235762e1c871ad0ccb60c8ee5ba337b9a",
                "license": "Apache-2.0",
                "not_shipped": True,
                "selection_report_sha256": _sha256(args.teacher_selection_report),
                "license_evidence": {
                    "readme_sha256": _QWEN_README_SHA256,
                    "license_sha256": _QWEN_LICENSE_SHA256,
                },
            },
        }
        if selection.get("winner") != "Qwen3.5-9B" or selection.get(
            "candidates", {}
        ).get("Qwen3.5-9B", {}).get("revision") != residual_source["label_teacher"][
            "revision"
        ]:
            raise ValueError("teacher selection report does not identify pinned Qwen3.5-9B")

    post_blend = None
    post_blend_source = None
    if args.post_blend_report is not None:
        if residual_source is None:
            raise ValueError("post-blend report requires the frozen teacher residual reference")
        report = json.loads(args.post_blend_report.read_text(encoding="utf-8"))
        blend = report.get("blend")
        standalone = report.get("standalone")
        if report.get("schema_version") != 1 or not isinstance(blend, dict):
            raise ValueError("invalid post-blend report")
        if report.get("rows") != 365654 or report.get("score_columns") != [
            "predict_structured",
            "predict_old_lamar",
        ]:
            raise ValueError("post-blend report does not match the frozen K17+LAMAR contract")
        if (
            blend.get("mode") != "category"
            or blend.get("protocol", {}).get("name") != "source_fold_cross_fit"
            or blend.get("protocol", {}).get("source_folds") != [0, 1, 2, 3, 4]
        ):
            raise ValueError("post-blend report is not frozen five-fold category cross-fit")
        if not isinstance(standalone, dict) or not abs(
            float(standalone["predict_structured"]["macro_pr_auc"])
            - float(residual_source["local_oof"])
        ) <= 1e-12:
            raise ValueError("post-blend reference is not the frozen teacher K17 release")
        deployment_weights = blend.get("final_weights")
        if not isinstance(deployment_weights, dict) or len(deployment_weights) != 20:
            raise ValueError("post-blend report must contain 20 category weight vectors")
        for values in deployment_weights.values():
            vector = tuple(float(value) for value in values)
            if len(vector) != 2 or any(value < 0.0 for value in vector) or not abs(
                sum(vector) - 1.0
            ) <= 1e-12:
                raise ValueError("invalid post-blend category weight vector")
        fold_uplift = blend.get("per_fold_uplift_over_structured")
        if not isinstance(fold_uplift, dict) or set(fold_uplift) != {
            "0",
            "1",
            "2",
            "3",
            "4",
        }:
            raise ValueError("post-blend report has incomplete fold evidence")
        positive_folds = sum(float(value) > 0.0 for value in fold_uplift.values())
        if positive_folds < 4 or float(blend["uplift_over_structured"]) <= 0.0:
            raise ValueError("post-blend report has no stable positive signal")
        post_blend = {
            "model_name": "lamar_600m",
            "representation": "category_rank",
            "scope": "category",
            "weights": deployment_weights,
        }
        post_blend_source = {
            "report_sha256": _sha256(args.post_blend_report),
            "reference_local_oof": standalone["predict_structured"]["macro_pr_auc"],
            "local_oof": blend["macro_pr_auc"],
            "uplift_vs_teacher_k17": blend["uplift_over_structured"],
            "positive_folds": positive_folds,
            "selection_protocol": blend["protocol"],
        }
    elif args.post_blend_route_contract is not None:
        raise ValueError("post-blend route contract requires a post-blend report")

    if args.category_router_report is None:
        ensemble = {
            "feature_order": ["rumodernbert_base", "lamar_600m"],
            "representation": "raw_logit",
            "scope": "global",
            "weights": list(weights),
        }
        router_source = None
    else:
        router_report = json.loads(
            args.category_router_report.read_text(encoding="utf-8")
        )
        method_name, method = resolve_category_router(
            router_report,
            selected_count=args.category_router_k,
        )
        deployment_weights = method.get("final_deployment_weights")
        if not isinstance(deployment_weights, dict) or len(deployment_weights) != 20:
            raise ValueError("category router must contain exactly 20 deployment weights")
        allowed_weights = {weights, (1.0, 0.0)}
        if any(tuple(values) not in allowed_weights for values in deployment_weights.values()):
            raise ValueError("category router contains an unexpected weight vector")
        ensemble = {
            "feature_order": ["rumodernbert_base", "lamar_600m"],
            "representation": "raw_logit",
            "scope": "category",
            "weights": deployment_weights,
        }
        router_source = {
            "report_sha256": _sha256(args.category_router_report),
            "method": method_name,
            "selected_categories": method["final_selected_categories"],
            "local_oof": method["local_oof"]["score"],
            "uplift_vs_base": method["candidate_minus_base"]["macro_pr_auc"],
        }

    if args.post_blend_route_contract is not None:
        contract = json.loads(
            args.post_blend_route_contract.read_text(encoding="utf-8")
        )
        if contract.get("schema_version") != 1 or post_blend is None:
            raise ValueError("invalid post-blend route contract")
        if contract.get("source_report_sha256") != _sha256(args.post_blend_report):
            raise ValueError("post-blend route contract source hash mismatch")
        route = contract.get("route")
        if not isinstance(route, dict) or route.get("model_name") != "lamar_600m":
            raise ValueError("post-blend route contract references an unexpected model")
        contract_weights = route.get("weights")
        if not isinstance(contract_weights, dict) or len(contract_weights) != 20:
            raise ValueError("post-blend route contract must contain 20 category weights")
        original_weights = post_blend["weights"]
        for category, values in contract_weights.items():
            vector = tuple(float(value) for value in values)
            _original = tuple(float(value) for value in original_weights[category])
            if len(vector) != 2 or any(value < 0.0 for value in vector) or not abs(
                sum(vector) - 1.0
            ) <= 1e-12:
                raise ValueError("invalid post-blend route-contract weight vector")
            if vector[1] > _original[1] + 1e-12:
                raise ValueError("runtime route contract may not increase LAMAR weight")
        base_lamar_categories = sorted(
            category
            for category, values in ensemble["weights"].items()
            if float(values[1]) > 0.0
        )
        workload = contract.get("runtime_workload")
        if not isinstance(workload, dict) or sorted(
            workload.get("base_lamar_categories", [])
        ) != base_lamar_categories:
            raise ValueError("post-blend route contract disagrees with base LAMAR routing")
        extra_lamar_categories = sorted(workload.get("extra_lamar_categories", []))
        deployed_lamar_categories = sorted(
            set(base_lamar_categories)
            | {
                category
                for category, values in contract_weights.items()
                if float(values[1]) > 0.0
            }
        )
        if deployed_lamar_categories != sorted(
            set(base_lamar_categories) | set(extra_lamar_categories)
        ) or int(workload.get("total_lamar_categories", -1)) != len(
            deployed_lamar_categories
        ):
            raise ValueError("post-blend route contract has inconsistent LAMAR workload")
        for stage in ("public", "private"):
            stage_workload = workload.get(stage)
            if (
                not isinstance(stage_workload, dict)
                or float(stage_workload["projected_container_seconds"])
                >= float(stage_workload["official_limit_seconds"])
                or float(stage_workload["projected_margin_seconds"]) <= 0.0
            ):
                raise ValueError(f"post-blend route contract exceeds {stage} runtime")
        quality = contract.get("quality")
        external_signal = contract.get("external_official_signal")
        strict_runtime_fallback = (
            isinstance(external_signal, dict)
            and int(quality.get("positive_folds", 0)) >= 3
            and float(external_signal.get("full_route_macro_pr_auc", 0.0))
            > float(external_signal.get("reference_macro_pr_auc", 0.0))
            and deployed_lamar_categories == base_lamar_categories
        ) if isinstance(quality, dict) else False
        if (
            not isinstance(quality, dict)
            or float(quality.get("uplift_vs_reference", 0.0)) <= 0.0
            or (
                int(quality.get("positive_folds", 0)) < 4
                and not strict_runtime_fallback
            )
        ):
            raise ValueError("post-blend route contract has no stable positive OOF signal")
        post_blend["weights"] = contract_weights
        post_blend_source = {
            **post_blend_source,
            "route_contract_sha256": _sha256(args.post_blend_route_contract),
            "full_route_local_oof": post_blend_source["local_oof"],
            "full_route_uplift_vs_teacher_k17": post_blend_source[
                "uplift_vs_teacher_k17"
            ],
            "local_oof": quality["macro_pr_auc"],
            "uplift_vs_teacher_k17": quality["uplift_vs_reference"],
            "positive_folds": quality["positive_folds"],
            "runtime_workload": workload,
            "external_official_signal": external_signal,
            "strict_runtime_fallback": strict_runtime_fallback,
        }

    project_root = Path(__file__).resolve().parents[1]
    runtime_sources = {
        project_root / "submission_runtime/ensemble_run.py": Path("run.py"),
        project_root / "submission_runtime/ecup_matching/__init__.py": Path(
            "ecup_matching/__init__.py"
        ),
        project_root / "src/ecup_matching/serialization.py": Path(
            "ecup_matching/serialization.py"
        ),
        project_root / "src/ecup_matching/neural_inference.py": Path(
            "ecup_matching/neural_inference.py"
        ),
        project_root / "src/ecup_matching/ensemble_inference.py": Path(
            "ecup_matching/ensemble_inference.py"
        ),
    }
    for source in runtime_sources:
        if not source.is_file():
            raise FileNotFoundError(source)

    args.output_path.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix="ecup-ensemble-", dir=args.output_path.parent))
    temporary_zip = args.output_path.with_name(f".{args.output_path.name}.{os.getpid()}.tmp")
    try:
        for source, relative in runtime_sources.items():
            destination = staging / relative
            destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(source, destination)
        _copy_vendor(args.base_archive, staging)
        base, base_dependencies, base_metadata = _copy_model(
            archive_path=args.base_archive,
            name="rumodernbert_base",
            target_root=staging,
            batch_size=args.base_batch_size,
            length_bucketing=args.length_bucketing,
        )
        lamar, lamar_dependencies, lamar_metadata = _copy_model(
            archive_path=args.lamar_archive,
            name="lamar_600m",
            target_root=staging,
            batch_size=args.lamar_batch_size,
            length_bucketing=args.length_bucketing,
        )
        teacher = None
        teacher_dependencies = None
        teacher_metadata = None
        if args.teacher_archive is not None:
            teacher, teacher_dependencies, teacher_metadata = _copy_model(
                archive_path=args.teacher_archive,
                name="teacher_student",
                target_root=staging,
                batch_size=args.teacher_batch_size,
                length_bucketing=args.length_bucketing,
            )
        if base_metadata.get("image") != lamar_metadata.get("image"):
            raise ValueError("source archives use different container images")
        if base_dependencies.get("container_digest") != lamar_dependencies.get(
            "container_digest"
        ):
            raise ValueError("source archives use different container digests")
        if teacher_metadata is not None and base_metadata.get("image") != teacher_metadata.get(
            "image"
        ):
            raise ValueError("teacher source archive uses a different container image")
        if teacher_dependencies is not None and base_dependencies.get(
            "container_digest"
        ) != teacher_dependencies.get("container_digest"):
            raise ValueError("teacher source archive uses a different container digest")

        length_router_source = None
        if args.base_length_router_report is not None:
            if args.category_router_report is None:
                raise ValueError("base length routing requires the frozen category ensemble")
            length_report = json.loads(
                args.base_length_router_report.read_text(encoding="utf-8")
            )
            length_evidence = resolve_length_router(length_report)
            categories = length_evidence["categories"]
            default_length = int(base["inference"]["max_length"])
            routed_length = int(args.base_routed_max_length)
            if routed_length <= default_length:
                raise ValueError("routed max length must exceed the base default")
            base["inference"]["max_length_by_category"] = {
                str(category): routed_length for category in categories
            }
            length_router_source = {
                "report_sha256": _sha256(args.base_length_router_report),
                "default_max_length": default_length,
                "routed_max_length": routed_length,
                "selected_categories": categories,
                "local_oof": length_evidence["local_oof"],
                "uplift_vs_k16": length_evidence["uplift"],
                "positive_folds": length_evidence["positive_folds"],
            }

        models = [base, lamar]
        if teacher is not None:
            models.append(teacher)
        manifest = {
            "schema_version": 1,
            "release_name": args.output_path.stem,
            "runtime_integrity": {
                "artifact_size": True,
                "artifact_sha256": False,
                "archive_crc_and_sha256_verified_offline": True,
            },
            "models": models,
            "ensemble": ensemble,
            "residual": residual,
            "post_blend": post_blend,
            "router_quality_source": router_source,
            "length_router_quality_source": length_router_source,
            "residual_quality_source": residual_source,
            "post_blend_quality_source": post_blend_source,
        }
        (staging / "ensemble_manifest.json").write_text(
            json.dumps(manifest, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        dependencies = {
            "schema_version": 1,
            "container_image": base_metadata["image"],
            "container_digest": base_dependencies["container_digest"],
            "container_license": base_dependencies["container_license"],
            "container_dependencies": base_dependencies["container_dependencies"],
            "vendored_dependencies": base_dependencies.get("vendored_dependencies", []),
            "archive_dependencies": [
                {
                    "name": "ecup_matching ensemble runtime",
                    "origin": "project-owned",
                    "license": "competition submission code"
                },
                {
                    "name": "RuModernBERT-base derived checkpoint",
                    "origin": "immutable audited Task 1 release",
                    "license": base["pretrained_model"]["license"]
                },
                {
                    "name": "LAMAR-600m derived checkpoint",
                    "origin": "immutable audited Task 1 release",
                    "license": lamar["pretrained_model"]["license"]
                }
            ]
            + (
                [
                    {
                        "name": "teacher-distilled RuModernBERT-base checkpoint",
                        "origin": "exact lineage-validated Task 1 release component",
                        "license": teacher["pretrained_model"]["license"],
                    },
                    {
                        "name": "Qwen/Qwen3.5-9B label teacher",
                        "origin": "local teacher scoring only; model weights are not shipped",
                        "revision": residual_source["label_teacher"]["revision"],
                        "license": residual_source["label_teacher"]["license"],
                    },
                ]
                if teacher is not None
                else []
            ),
            "source_archives": {
                "rumodernbert_base": base["source_archive"],
                "lamar_600m": lamar["source_archive"],
                **(
                    {"teacher_student": teacher["source_archive"]}
                    if teacher is not None
                    else {}
                ),
            },
        }
        (staging / "RUNTIME_DEPENDENCIES.json").write_text(
            json.dumps(dependencies, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        (staging / "metadata.json").write_text(
            json.dumps(
                {"image": base_metadata["image"], "entry_point": _ENTRY_POINT}, indent=2
            )
            + "\n",
            encoding="utf-8",
        )

        _write_deterministic_zip(staging, temporary_zip)
        names = _validate_zip(temporary_zip)
        if temporary_zip.stat().st_size > _MAX_ARCHIVE_BYTES:
            raise ValueError("ensemble archive exceeds the official 5 GB limit")
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
            "container_image": base_metadata["image"],
            "entry_point": _ENTRY_POINT,
            "contents": [
                _artifact(path, staging)
                for path in sorted(staging.rglob("*"))
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
        if staging.exists():
            shutil.rmtree(staging)


if __name__ == "__main__":
    main()
