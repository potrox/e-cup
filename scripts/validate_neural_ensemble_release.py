from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import time
import zipfile
from datetime import UTC, datetime
from pathlib import Path, PurePosixPath
from typing import Any

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

from ecup_matching.release_evidence import (
    resolve_category_router,
    resolve_length_router,
)

_OFFICIAL_LIMITS = {"check": 60.0, "public": 360.0, "private": 780.0}
_OFFICIAL_MAX_ARCHIVE_BYTES = 5_000_000_000
_OFFICIAL_HOST_MEMORY_BYTES = 200_000_000_000
_REQUIRED_ROOTS = {
    "RUNTIME_DEPENDENCIES.json",
    "ecup_matching",
    "ensemble_manifest.json",
    "metadata.json",
    "models",
    "run.py",
    "vendor",
}
_FORBIDDEN_SUFFIXES = {
    ".csv",
    ".ipynb",
    ".joblib",
    ".npy",
    ".npz",
    ".parquet",
    ".pickle",
    ".pkl",
    ".pt",
    ".pth",
}
_RUNTIME_PREFIX = "runtime_resources="
_QWEN_README_SHA256 = "c5f5a8c2dddab69cfbf05279235aa5fddb137939a06539c4c7637aa900fef6d0"
_QWEN_LICENSE_SHA256 = "bbedc3fda3305820b977265f01b8619d87570a6739de3a5582c3464840f1e57a"


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Validate the exact base+LAMAR ensemble archive against organizer limits"
    )
    parser.add_argument("--archive", type=Path, required=True)
    parser.add_argument("--release-manifest", type=Path, required=True)
    parser.add_argument("--base-archive", type=Path, required=True)
    parser.add_argument("--lamar-archive", type=Path, required=True)
    parser.add_argument("--teacher-archive", type=Path)
    parser.add_argument("--base-parity", type=Path, required=True)
    parser.add_argument("--lamar-parity", type=Path, required=True)
    parser.add_argument("--teacher-parity", type=Path)
    parser.add_argument("--check-input", type=Path, required=True)
    parser.add_argument("--public-input", type=Path, required=True)
    parser.add_argument("--private-input", type=Path, required=True)
    parser.add_argument("--category-router-report", type=Path)
    parser.add_argument("--category-router-k", type=int)
    parser.add_argument("--base-length-router-report", type=Path)
    parser.add_argument("--base-routed-max-length", type=int)
    parser.add_argument("--base-long-parity", type=Path)
    parser.add_argument("--teacher-residual-contract", type=Path)
    parser.add_argument("--teacher-residual-report", type=Path)
    parser.add_argument("--teacher-selection-report", type=Path)
    parser.add_argument("--reuse-stages-from", type=Path)
    parser.add_argument("--reuse-check-container-seconds", type=float)
    parser.add_argument("--reuse-check-repeat-container-seconds", type=float)
    parser.add_argument("--reuse-public-container-seconds", type=float)
    parser.add_argument("--reuse-private-container-seconds", type=float)
    parser.add_argument("--validation-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(8 * 1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _json(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(path)
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"expected JSON object: {path}")
    return value


def _extract_archive(archive_path: Path, destination: Path) -> dict[str, Any]:
    if destination.exists():
        raise FileExistsError(destination)
    destination.mkdir(parents=True)
    with zipfile.ZipFile(archive_path) as archive:
        if archive.testzip() is not None:
            raise ValueError("archive CRC validation failed")
        names: set[str] = set()
        roots: set[str] = set()
        forbidden: list[str] = []
        for info in archive.infolist():
            path = PurePosixPath(info.filename)
            if path.is_absolute() or ".." in path.parts or info.filename in names:
                raise ValueError(f"unsafe or duplicate archive entry: {info.filename}")
            names.add(info.filename)
            if path.parts:
                roots.add(path.parts[0])
            if path.suffix.lower() in _FORBIDDEN_SUFFIXES:
                forbidden.append(info.filename)
        if roots != _REQUIRED_ROOTS:
            raise ValueError(f"unexpected archive roots: {sorted(roots)}")
        if forbidden:
            raise ValueError(f"forbidden release files: {forbidden[:5]}")
        archive.extractall(destination)
    return {
        "entries": len(names),
        "roots": sorted(roots),
        "crc_valid": True,
        "unsafe_paths": 0,
        "forbidden_development_entries": 0,
    }


def _extract_source_archive(archive_path: Path, destination: Path) -> None:
    if destination.exists():
        raise FileExistsError(destination)
    destination.mkdir(parents=True)
    with zipfile.ZipFile(archive_path) as archive:
        if archive.testzip() is not None:
            raise ValueError("source archive CRC validation failed")
        names: set[str] = set()
        for info in archive.infolist():
            path = PurePosixPath(info.filename)
            if path.is_absolute() or ".." in path.parts or info.filename in names:
                raise ValueError(f"unsafe or duplicate source entry: {info.filename}")
            names.add(info.filename)
        archive.extractall(destination)


def _docker_mount(source: Path, target: str, *, read_only: bool) -> str:
    value = f"type=bind,source={source.resolve()},target={target}"
    return f"{value},readonly" if read_only else value


def _runtime_resources(log_path: Path, expected_pairs: int) -> dict[str, Any]:
    records = [
        json.loads(line.removeprefix(_RUNTIME_PREFIX))
        for line in log_path.read_text(encoding="utf-8", errors="replace").splitlines()
        if line.startswith(_RUNTIME_PREFIX)
    ]
    if len(records) != 1:
        raise ValueError(f"expected one runtime resource record: {log_path}")
    record = records[0]
    if int(record["pairs"]) != expected_pairs:
        raise ValueError("runtime resource pair count mismatch")
    for field in (
        "elapsed_seconds",
        "peak_process_rss_bytes",
        "peak_gpu_allocated_bytes",
        "peak_gpu_reserved_bytes",
        "gpu_total_memory_bytes",
        "gpu_name",
    ):
        if record.get(field) in (None, ""):
            raise ValueError(f"runtime resource field is missing: {field}")
    return record


def _audit_output(input_root: Path, output_path: Path) -> dict[str, Any]:
    matches = pd.read_parquet(input_root / "matches.parquet", columns=["id1", "id2"])
    result = pd.read_csv(output_path)
    if list(result.columns) != ["id1", "id2", "predict"]:
        raise ValueError(f"invalid output schema: {result.columns.tolist()}")
    if len(result) != len(matches):
        raise ValueError("output row count mismatch")
    id1_matches = np.array_equal(result["id1"].to_numpy(), matches["id1"].to_numpy())
    id2_matches = np.array_equal(result["id2"].to_numpy(), matches["id2"].to_numpy())
    if not id1_matches or not id2_matches:
        raise ValueError("output pair order or coverage mismatch")
    if not np.isfinite(result["predict"].to_numpy()).all():
        raise ValueError("output contains non-finite predictions")
    return {
        "pairs": len(result),
        "columns": result.columns.tolist(),
        "row_coverage": "exact",
        "input_order_preserved": True,
        "finite_predictions": True,
        "output_bytes": output_path.stat().st_size,
        "output_sha256": _sha256(output_path),
    }


def _reconstruct_stage(
    *,
    name: str,
    input_root: Path,
    source_root: Path,
    container_seconds: float,
) -> dict[str, Any]:
    stage_root = source_root / name
    output_path = stage_root / "predictions.csv"
    log_path = stage_root / "runtime.log"
    if not output_path.is_file() or not log_path.is_file():
        raise FileNotFoundError(stage_root)
    stage_kind = name.split("_", 1)[0]
    limit = _OFFICIAL_LIMITS[stage_kind]
    output = _audit_output(input_root, output_path)
    resources = _runtime_resources(log_path, output["pairs"])
    application_seconds = float(resources["elapsed_seconds"])
    if (
        not np.isfinite(container_seconds)
        or container_seconds < application_seconds
        or container_seconds - application_seconds > 60.0
    ):
        raise ValueError(f"invalid recovered container timing for stage: {name}")
    return {
        **output,
        "application_seconds": application_seconds,
        "container_seconds": container_seconds,
        "official_limit_seconds": limit,
        "within_official_limit": container_seconds <= limit,
        "resources": resources,
        "log_sha256": _sha256(log_path),
        "output_path": str(output_path.resolve()),
        "reused_after_post_inference_audit_failure": True,
    }


def _run_stage(
    *,
    name: str,
    input_root: Path,
    extracted: Path,
    image: str,
    validation_root: Path,
) -> dict[str, Any]:
    stage_root = validation_root / name
    if stage_root.exists():
        raise FileExistsError(stage_root)
    stage_root.mkdir(parents=True)
    output_path = stage_root / "predictions.csv"
    log_path = stage_root / "runtime.log"
    stage_kind = name.split("_", 1)[0]
    limit = _OFFICIAL_LIMITS[stage_kind]
    command = [
        "docker",
        "run",
        "--rm",
        "--pull",
        "never",
        "--network",
        "none",
        "--gpus",
        "all",
        "--cpus",
        "20",
        "--env",
        "HF_HUB_OFFLINE=1",
        "--env",
        "TRANSFORMERS_OFFLINE=1",
        "--env",
        "HF_DATASETS_OFFLINE=1",
        "--env",
        "PYTHONNOUSERSITE=1",
        "--mount",
        _docker_mount(extracted, "/workspace/solution", read_only=True),
        "--mount",
        _docker_mount(input_root, "/workspace/input", read_only=True),
        "--mount",
        _docker_mount(stage_root, "/workspace/output", read_only=False),
        "--workdir",
        "/workspace/solution",
        image,
        "python",
        "-u",
        "run.py",
        "--items_path",
        "/workspace/input/items.parquet",
        "--matches_path",
        "/workspace/input/matches.parquet",
        "--output_path",
        "/workspace/output/predictions.csv",
    ]
    print(f"ensemble_release_stage_start={name}", flush=True)
    started = time.perf_counter()
    with log_path.open("w", encoding="utf-8", newline="") as log:
        completed = subprocess.run(
            command,
            stdout=log,
            stderr=subprocess.STDOUT,
            check=False,
            timeout=limit + 60.0,
        )
    container_seconds = time.perf_counter() - started
    if completed.returncode != 0:
        tail = "\n".join(log_path.read_text(encoding="utf-8", errors="replace").splitlines()[-80:])
        raise RuntimeError(f"Docker stage failed: {name}\n{tail}")
    output = _audit_output(input_root, output_path)
    resources = _runtime_resources(log_path, output["pairs"])
    payload = {
        **output,
        "application_seconds": float(resources["elapsed_seconds"]),
        "container_seconds": container_seconds,
        "official_limit_seconds": limit,
        "within_official_limit": container_seconds <= limit,
        "resources": resources,
        "log_sha256": _sha256(log_path),
        "output_path": str(output_path.resolve()),
    }
    print(
        f"ensemble_release_stage_done={name} container_seconds={container_seconds:.3f}",
        flush=True,
    )
    return payload


def _parity(
    *,
    ensemble_output: Path,
    base_output: Path,
    lamar_output: Path,
    teacher_output: Path | None,
    input_root: Path,
    ensemble_config: dict[str, Any],
    base_long_output: Path | None = None,
    base_long_categories: set[str] | None = None,
    residual_config: dict[str, Any] | None = None,
) -> dict[str, Any]:
    def read_component(path: Path, output_name: str) -> pd.DataFrame:
        with path.open("rb") as source:
            is_parquet = source.read(4) == b"PAR1"
        frame = pd.read_parquet(path) if is_parquet else pd.read_csv(path)
        score_column = "predict" if "predict" in frame.columns else "score"
        required = ["id1", "id2", score_column]
        if any(column not in frame.columns for column in required):
            raise ValueError(f"invalid component predictions: {path}")
        return frame[required].rename(columns={score_column: output_name})

    ensemble = pd.read_csv(ensemble_output)
    base = read_component(base_output, "base")
    lamar = read_component(lamar_output, "lamar")
    keys = ["id1", "id2"]
    joined = ensemble.merge(
        base.rename(columns={"predict": "base"}), on=keys, how="left", validate="1:1"
    ).merge(
        lamar.rename(columns={"predict": "lamar"}), on=keys, how="left", validate="1:1"
    )
    if base_long_output is not None:
        base_long = read_component(base_long_output, "base_long")
        joined = joined.merge(
            base_long,
            on=keys,
            how="left",
            validate="1:1",
        )
        if joined["base_long"].isna().any():
            raise ValueError("long-context base parity predictions are incomplete")
    if joined[["base", "lamar"]].isna().any().any():
        raise ValueError("component parity predictions are incomplete")
    actual = joined["predict"].to_numpy()
    items = pd.read_parquet(input_root / "items.parquet", columns=["id", "category"])
    categories = items.set_index("id", verify_integrity=True)["category"]
    joined["category"] = categories.reindex(joined["id1"].to_numpy()).to_numpy()
    base_score = joined["base"].to_numpy(copy=True)
    if base_long_categories:
        long_mask = joined["category"].isin(base_long_categories).to_numpy()
        base_score[long_mask] = joined.loc[long_mask, "base_long"].to_numpy()
    if ensemble_config["scope"] == "global":
        base_weight, lamar_weight = ensemble_config["weights"]
        expected = (
            base_weight * base_score
            + lamar_weight * joined["lamar"].to_numpy()
        )
    elif ensemble_config["scope"] == "category":
        expected = np.empty(len(joined), dtype=np.float64)
        for category, group in joined.groupby("category", sort=False):
            base_weight, lamar_weight = ensemble_config["weights"][str(category)]
            group_base = base_score[group.index]
            expected[group.index] = (
                base_weight * group_base
                + lamar_weight * group["lamar"].to_numpy()
            )
    else:
        raise ValueError("unsupported ensemble scope in parity audit")
    if residual_config is not None:
        if teacher_output is None:
            raise ValueError("teacher residual parity requires teacher predictions")
        teacher = read_component(teacher_output, "teacher")
        joined = joined.merge(teacher, on=keys, how="left", validate="1:1")
        if joined["teacher"].isna().any():
            raise ValueError("teacher parity predictions are incomplete")
        if residual_config.get("representation") != "category_rank" or residual_config.get(
            "scope"
        ) != "category_subset":
            raise ValueError("unsupported residual parity contract")
        reference_rank = joined.assign(_reference=expected).groupby(
            "category", sort=False
        )["_reference"].rank(method="average")
        teacher_rank = joined.groupby("category", sort=False)["teacher"].rank(
            method="average"
        )
        counts = joined.groupby("category", sort=False)["teacher"].transform("size")
        reference_rank = reference_rank.to_numpy(dtype=np.float64) / (
            counts.to_numpy(dtype=np.float64) + 1.0
        )
        teacher_rank = teacher_rank.to_numpy(dtype=np.float64) / (
            counts.to_numpy(dtype=np.float64) + 1.0
        )
        routed = joined["category"].isin(residual_config["categories"]).to_numpy()
        residual_expected = expected.copy()
        residual_expected[routed] = (
            float(residual_config["reference_weight"]) * reference_rank[routed]
            + float(residual_config["model_weight"]) * teacher_rank[routed]
        )
        if not np.array_equal(residual_expected[~routed], expected[~routed]):
            raise RuntimeError("residual parity changed non-routed predictions")
        expected = residual_expected
    delta = actual - expected
    joined["expected"] = expected
    per_category = {
        str(category): float(spearmanr(group["predict"], group["expected"]).statistic)
        for category, group in joined.groupby("category", sort=True)
    }
    return {
        "max_abs_delta": float(np.max(np.abs(delta))),
        "mean_abs_delta": float(np.mean(np.abs(delta))),
        "global_spearman": float(spearmanr(actual, expected).statistic),
        "minimum_category_spearman": min(per_category.values()),
        "per_category_spearman": per_category,
        "passed": (
            float(np.max(np.abs(delta))) <= 0.125
            and float(np.mean(np.abs(delta))) <= 0.01
            and min(per_category.values()) >= 0.9999
        ),
    }


def main() -> None:
    args = _parse_args()
    for path in (
        args.archive,
        args.release_manifest,
        args.base_archive,
        args.lamar_archive,
        args.base_parity,
        args.lamar_parity,
    ):
        if not path.is_file():
            raise FileNotFoundError(path)
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
            "teacher archive, residual contract, residual report, and selection report "
            "are required together"
        )
    if args.teacher_parity is not None and args.teacher_archive is None:
        raise ValueError("teacher parity predictions require a teacher archive")
    for path in (*teacher_inputs, args.teacher_parity):
        if path is not None and not path.is_file():
            raise FileNotFoundError(path)
    if args.base_long_parity is not None and not args.base_long_parity.is_file():
        raise FileNotFoundError(args.base_long_parity)
    for input_root in (args.check_input, args.public_input, args.private_input):
        for name in ("items.parquet", "matches.parquet"):
            if not (input_root / name).is_file():
                raise FileNotFoundError(input_root / name)
    if args.validation_root.exists():
        raise FileExistsError(args.validation_root)
    args.validation_root.mkdir(parents=True)

    archive_sha = _sha256(args.archive)
    release_manifest = _json(args.release_manifest)
    if release_manifest["archive"]["sha256"] != archive_sha:
        raise ValueError("release manifest does not match exact archive")
    archive_audit = _extract_archive(args.archive, args.validation_root / "extracted")
    extracted = args.validation_root / "extracted"
    metadata = _json(extracted / "metadata.json")
    manifest = _json(extracted / "ensemble_manifest.json")
    dependencies = _json(extracted / "RUNTIME_DEPENDENCIES.json")
    source_hashes = dependencies.get("source_archives", {})
    if source_hashes.get("rumodernbert_base", {}).get("sha256") != _sha256(
        args.base_archive
    ):
        raise ValueError("base source archive hash mismatch")
    if source_hashes.get("lamar_600m", {}).get("sha256") != _sha256(args.lamar_archive):
        raise ValueError("LAMAR source archive hash mismatch")
    if args.teacher_archive is not None and source_hashes.get("teacher_student", {}).get(
        "sha256"
    ) != _sha256(args.teacher_archive):
        raise ValueError("teacher source archive hash mismatch")
    if (args.category_router_report is None) != (args.category_router_k is None):
        raise ValueError("category router report and K must be provided together")
    if (args.base_length_router_report is None) != (
        args.base_routed_max_length is None
    ):
        raise ValueError(
            "base length router report and routed max length must be provided together"
        )
    if args.base_long_parity is not None and args.base_length_router_report is None:
        raise ValueError("base long parity requires a base length router report")
    reuse_seconds = (
        args.reuse_check_container_seconds,
        args.reuse_check_repeat_container_seconds,
        args.reuse_public_container_seconds,
        args.reuse_private_container_seconds,
    )
    if args.reuse_stages_from is None and any(value is not None for value in reuse_seconds):
        raise ValueError("reused container timings require --reuse-stages-from")
    if args.reuse_stages_from is not None and (
        not args.reuse_stages_from.is_dir()
        or any(value is None for value in reuse_seconds)
    ):
        raise ValueError("reused stages require a source root and all four timings")
    if args.category_router_report is None:
        expected_ensemble = {
            "feature_order": ["rumodernbert_base", "lamar_600m"],
            "representation": "raw_logit",
            "scope": "global",
            "weights": [0.55, 0.45],
        }
    else:
        router_report = _json(args.category_router_report)
        _, method = resolve_category_router(
            router_report,
            selected_count=args.category_router_k,
        )
        expected_ensemble = {
            "feature_order": ["rumodernbert_base", "lamar_600m"],
            "representation": "raw_logit",
            "scope": "category",
            "weights": method["final_deployment_weights"],
        }
    if manifest.get("ensemble") != expected_ensemble:
        raise ValueError("exact ensemble configuration mismatch")

    expected_residual = None
    expected_residual_source = None
    if args.teacher_residual_contract is not None:
        contract = _json(args.teacher_residual_contract)
        residual_report = _json(args.teacher_residual_report)
        teacher_selection = _json(args.teacher_selection_report)
        route = contract.get("route")
        if not isinstance(route, dict):
            raise ValueError("teacher residual contract has no route")
        if residual_report.get("route") != route or residual_report.get("gate", {}).get(
            "pass"
        ) is not True:
            raise ValueError("teacher residual report does not pass its immutable contract")
        expected_residual = {
            "model_name": "teacher_student",
            "representation": "category_rank",
            "scope": "category_subset",
            "categories": route["categories"],
            "reference_weight": float(route["reference_weight"]),
            "model_weight": float(route["teacher_student_weight"]),
        }
        expected_residual_source = {
            "contract_sha256": _sha256(args.teacher_residual_contract),
            "report_sha256": _sha256(args.teacher_residual_report),
            "local_oof": residual_report["candidate"]["score"],
            "uplift_vs_k17": residual_report["candidate_minus_reference"][
                "macro_average_precision"
            ],
            "positive_folds": residual_report["candidate_minus_reference"][
                "positive_folds"
            ],
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
        if teacher_selection.get("winner") != "Qwen3.5-9B" or teacher_selection.get(
            "candidates", {}
        ).get("Qwen3.5-9B", {}).get("revision") != expected_residual_source[
            "label_teacher"
        ]["revision"]:
            raise ValueError("teacher selection report does not identify pinned Qwen3.5-9B")
    if manifest.get("residual") != expected_residual:
        raise ValueError("exact teacher residual configuration mismatch")
    if manifest.get("residual_quality_source") != expected_residual_source:
        raise ValueError("exact teacher residual quality source mismatch")

    base_long_categories: set[str] | None = None
    base_long_stage: dict[str, Any] | None = None
    if args.base_length_router_report is not None:
        length_report = _json(args.base_length_router_report)
        length_evidence = resolve_length_router(length_report)
        categories = length_evidence["categories"]
        base_long_categories = {str(category) for category in categories}
        routed_length = int(args.base_routed_max_length)
        expected_router = {
            "report_sha256": _sha256(args.base_length_router_report),
            "default_max_length": 256,
            "routed_max_length": routed_length,
            "selected_categories": categories,
            "local_oof": length_evidence["local_oof"],
            "uplift_vs_k16": length_evidence["uplift"],
            "positive_folds": length_evidence["positive_folds"],
        }
        if manifest.get("length_router_quality_source") != expected_router:
            raise ValueError("exact length router quality source mismatch")
        base_model = manifest.get("models", [None])[0]
        if not isinstance(base_model, dict):
            raise ValueError("base model manifest is missing")
        expected_lengths = {category: routed_length for category in categories}
        if base_model.get("inference", {}).get("max_length_by_category") != expected_lengths:
            raise ValueError("exact base length routing config mismatch")

        if args.base_long_parity is not None:
            base_long_stage = {
                **_audit_output(args.public_input, args.base_long_parity),
                "output_path": str(args.base_long_parity.resolve()),
                "source_archive_sha256": _sha256(args.base_archive),
                "routed_max_length": routed_length,
                "reused_after_exact_component_run": True,
            }
        else:
            parity_root = args.validation_root / "base_long_component"
            _extract_source_archive(args.base_archive, parity_root)
            parity_manifest_path = parity_root / "neural_manifest.json"
            parity_manifest = _json(parity_manifest_path)
            if int(parity_manifest["inference"]["max_length"]) != 256:
                raise ValueError("base parity source has unexpected default max length")
            parity_manifest["inference"]["max_length"] = routed_length
            parity_manifest_path.write_text(
                json.dumps(parity_manifest, ensure_ascii=False, indent=2) + "\n",
                encoding="utf-8",
            )
            base_long_stage = _run_stage(
                name="public_base_long",
                input_root=args.public_input,
                extracted=parity_root,
                image=metadata["image"],
                validation_root=args.validation_root,
            )
    elif manifest.get("length_router_quality_source") is not None:
        raise ValueError("unexpected length router in archive")

    teacher_component_stage: dict[str, Any] | None = None
    teacher_parity_path = args.teacher_parity
    if args.teacher_archive is not None:
        if teacher_parity_path is not None:
            teacher_component_stage = {
                **_audit_output(args.public_input, teacher_parity_path),
                "output_path": str(teacher_parity_path.resolve()),
                "source_archive_sha256": _sha256(args.teacher_archive),
                "reused_after_exact_component_run": True,
            }
        else:
            teacher_root = args.validation_root / "teacher_component"
            _extract_source_archive(args.teacher_archive, teacher_root)
            teacher_component_stage = _run_stage(
                name="public_teacher_component",
                input_root=args.public_input,
                extracted=teacher_root,
                image=metadata["image"],
                validation_root=args.validation_root,
            )
            teacher_parity_path = Path(teacher_component_stage["output_path"])

    if args.reuse_stages_from is None:
        stages = {
            "check": _run_stage(
                name="check",
                input_root=args.check_input,
                extracted=extracted,
                image=metadata["image"],
                validation_root=args.validation_root,
            ),
            "check_repeat": _run_stage(
                name="check_repeat",
                input_root=args.check_input,
                extracted=extracted,
                image=metadata["image"],
                validation_root=args.validation_root,
            ),
            "public": _run_stage(
                name="public",
                input_root=args.public_input,
                extracted=extracted,
                image=metadata["image"],
                validation_root=args.validation_root,
            ),
            "private": _run_stage(
                name="private",
                input_root=args.private_input,
                extracted=extracted,
                image=metadata["image"],
                validation_root=args.validation_root,
            ),
        }
        stage_execution = "fresh"
    else:
        stages = {
            "check": _reconstruct_stage(
                name="check",
                input_root=args.check_input,
                source_root=args.reuse_stages_from,
                container_seconds=float(args.reuse_check_container_seconds),
            ),
            "check_repeat": _reconstruct_stage(
                name="check_repeat",
                input_root=args.check_input,
                source_root=args.reuse_stages_from,
                container_seconds=float(args.reuse_check_repeat_container_seconds),
            ),
            "public": _reconstruct_stage(
                name="public",
                input_root=args.public_input,
                source_root=args.reuse_stages_from,
                container_seconds=float(args.reuse_public_container_seconds),
            ),
            "private": _reconstruct_stage(
                name="private",
                input_root=args.private_input,
                source_root=args.reuse_stages_from,
                container_seconds=float(args.reuse_private_container_seconds),
            ),
        }
        stage_execution = "recovered_from_immutable_outputs_after_post_inference_failure"
    deterministic = stages["check"]["output_sha256"] == stages["check_repeat"][
        "output_sha256"
    ]
    parity = _parity(
        ensemble_output=Path(stages["public"]["output_path"]),
        base_output=args.base_parity,
        lamar_output=args.lamar_parity,
        teacher_output=teacher_parity_path,
        input_root=args.public_input,
        ensemble_config=manifest["ensemble"],
        base_long_output=(
            Path(base_long_stage["output_path"]) if base_long_stage is not None else None
        ),
        base_long_categories=base_long_categories,
        residual_config=expected_residual,
    )
    resources = [stages[name]["resources"] for name in ("check", "public", "private")]
    gates = {
        "archive_within_official_size_limit": args.archive.stat().st_size
        <= _OFFICIAL_MAX_ARCHIVE_BYTES,
        "check_within_official_limit": stages["check"]["within_official_limit"],
        "public_within_official_limit": stages["public"]["within_official_limit"],
        "private_within_official_limit": stages["private"]["within_official_limit"],
        "peak_host_memory_within_official_limit": all(
            int(item["peak_process_rss_bytes"]) <= _OFFICIAL_HOST_MEMORY_BYTES
            for item in resources
        ),
        "peak_gpu_memory_within_measured_device_capacity": all(
            int(item["peak_gpu_reserved_bytes"]) <= int(item["gpu_total_memory_bytes"])
            for item in resources
        ),
        "deterministic_repeat": deterministic,
        "component_release_parity": parity["passed"],
        "schema_order_coverage_finite": True,
        "offline_no_network": True,
        "source_archives_and_licenses_traced": True,
    }
    report = {
        "schema_version": 1,
        "validated_at_utc": datetime.now(UTC).isoformat(),
        "verdict": "PASS" if all(gates.values()) else "FAIL",
        "archive": {
            "path": str(args.archive.resolve()),
            "bytes": args.archive.stat().st_size,
            "sha256": archive_sha,
            **archive_audit,
        },
        "official_contract": {
            "source": "docs/problem_statement.md#закрытый-запуск-и-ограничения",
            "limits_seconds": _OFFICIAL_LIMITS,
            "archive_bytes": _OFFICIAL_MAX_ARCHIVE_BYTES,
            "host_memory_bytes": _OFFICIAL_HOST_MEMORY_BYTES,
            "strict_internal_runtime_gate": None,
        },
        "gates": gates,
        "stages": stages,
        "stage_execution": stage_execution,
        "parity_component_stages": {
            **(
                {"public_base_long": base_long_stage}
                if base_long_stage is not None
                else {}
            ),
            **(
                {"public_teacher_component": teacher_component_stage}
                if teacher_component_stage is not None
                else {}
            ),
        },
        "parity": parity,
        "ensemble": manifest["ensemble"],
        "residual": expected_residual,
        "residual_quality_source": expected_residual_source,
        "submission_uploaded": False,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(args.output)
    if not all(gates.values()):
        raise RuntimeError(f"ensemble release gates failed: {gates}")


if __name__ == "__main__":
    main()
