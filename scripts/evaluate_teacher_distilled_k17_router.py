from __future__ import annotations

import argparse
import json
from dataclasses import asdict
from hashlib import sha256
from pathlib import Path
from typing import Any

import numpy as np
import polars as pl

from ecup_matching.blending import category_fold_percentile_ranks
from ecup_matching.metrics import macro_average_precision
from ecup_matching.neural_ensemble import load_ensemble_oof

PAIR_KEYS = ("id1", "id2")
FOLDS = (0, 1, 2, 3, 4)
WEAK_CATEGORIES = ("Обувь", "Одежда", "Ювелирные изделия")


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate the preregistered teacher-student residual on immutable K17"
    )
    parser.add_argument("--base-cv-root", type=Path, required=True)
    parser.add_argument("--teacher-student-oof", type=Path, required=True)
    parser.add_argument("--k17-oof", type=Path, required=True)
    parser.add_argument("--contract", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def _sha256(path: Path) -> str:
    digest = sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _score(
    target: np.ndarray,
    prediction: np.ndarray,
    categories: np.ndarray,
    *,
    expected_categories: int = 20,
) -> dict[str, Any]:
    report = macro_average_precision(target, prediction, categories)
    if len(report.per_category) != expected_categories:
        raise ValueError(
            "teacher-distilled router category count differs: "
            f"{len(report.per_category)} != {expected_categories}"
        )
    return asdict(report)


def _stress_scores(
    frame: pl.DataFrame,
    target: np.ndarray,
    prediction: np.ndarray,
    categories: np.ndarray,
) -> dict[str, float]:
    surface = frame["surface_similarity"].to_numpy()
    conflicts = frame["identity_conflicts"].to_numpy()
    hard_negative_min = frame["hard_negative_min_similarity"].to_numpy()
    hard_positive_max = frame["hard_positive_max_similarity"].to_numpy()
    positives = target == 1
    negatives = ~positives
    masks = {
        "hard_negative_challenge": positives
        | (negatives & (surface >= hard_negative_min)),
        "hard_positive_challenge": negatives
        | (positives & (surface <= hard_positive_max)),
        "identity_conflict_challenge": positives | (negatives & (conflicts > 0)),
    }
    return {
        name: _score(target[mask], prediction[mask], categories[mask])["score"]
        for name, mask in masks.items()
    }


def _load_k17(frame: pl.DataFrame, path: Path) -> pl.DataFrame:
    required = {*PAIR_KEYS, "target", "category", "fold", "predict"}
    schema = pl.read_parquet_schema(path)
    missing = sorted(required - set(schema))
    if missing:
        raise ValueError(f"K17 OOF is missing columns: {missing}")
    k17 = pl.read_parquet(path, columns=sorted(required)).rename(
        {
            "target": "k17_target",
            "category": "k17_category",
            "fold": "k17_fold",
            "predict": "k17_score",
        }
    )
    if k17.height != frame.height:
        raise ValueError(f"K17 row count differs: {k17.height} != {frame.height}")
    if k17.select(pl.struct(*PAIR_KEYS).n_unique()).item() != k17.height:
        raise ValueError("K17 OOF contains duplicate pair keys")
    joined = frame.join(k17, on=list(PAIR_KEYS), how="left", validate="1:1")
    if joined["k17_score"].null_count():
        raise ValueError("K17 OOF does not cover every teacher-student row")
    mismatches = joined.filter(
        (pl.col("target") != pl.col("k17_target"))
        | (pl.col("category") != pl.col("k17_category"))
        | (pl.col("fold") != pl.col("k17_fold"))
    ).height
    if mismatches:
        raise ValueError(f"K17/teacher-student contract mismatches: {mismatches}")
    return joined.drop("k17_target", "k17_category", "k17_fold")


def main() -> None:
    args = _parse_args()
    report_path = args.output_dir / "evaluation.json"
    prediction_path = args.output_dir / "oof_predictions.parquet"
    if not args.overwrite:
        for path in (report_path, prediction_path):
            if path.exists():
                raise FileExistsError(path)

    contract = json.loads(args.contract.read_text(encoding="utf-8"))
    route = contract["route"]
    reference_weight = float(route["reference_weight"])
    teacher_weight = float(route["teacher_student_weight"])
    categories_to_route = tuple(str(value) for value in route["categories"])
    if categories_to_route != WEAK_CATEGORIES:
        raise ValueError(f"unexpected routed categories: {categories_to_route}")
    if not np.isclose(reference_weight + teacher_weight, 1.0):
        raise ValueError("router weights do not sum to one")
    expected_k17_hash = str(contract["reference"]["sha256"])
    actual_k17_hash = _sha256(args.k17_oof)
    if actual_k17_hash != expected_k17_hash:
        raise ValueError(
            f"immutable K17 hash mismatch: {actual_k17_hash} != {expected_k17_hash}"
        )

    frame = load_ensemble_oof(args.base_cv_root, args.teacher_student_oof).rename(
        {"small_score": "teacher_student_score"}
    )
    frame = _load_k17(frame, args.k17_oof)
    target = frame["target"].to_numpy().astype(np.int8, copy=False)
    categories = frame["category"].to_numpy()
    folds = frame["fold"].to_numpy()
    teacher_student = frame["teacher_student_score"].to_numpy()
    k17 = frame["k17_score"].to_numpy()
    if tuple(sorted(int(value) for value in np.unique(folds))) != FOLDS:
        raise ValueError("OOF frame does not contain the frozen five folds")
    if not np.isfinite(np.column_stack((teacher_student, k17))).all():
        raise ValueError("OOF scores contain non-finite values")

    k17_rank = category_fold_percentile_ranks(k17, categories, folds)
    teacher_rank = category_fold_percentile_ranks(teacher_student, categories, folds)
    routed_mask = np.isin(categories, categories_to_route)
    candidate = k17.copy()
    candidate[routed_mask] = (
        reference_weight * k17_rank[routed_mask]
        + teacher_weight * teacher_rank[routed_mask]
    )
    if not np.array_equal(candidate[~routed_mask], k17[~routed_mask]):
        raise RuntimeError("non-routed rows lost exact K17 parity")

    reference_metric = _score(target, k17, categories)
    candidate_metric = _score(target, candidate, categories)
    per_fold: dict[str, dict[str, float]] = {}
    for fold in FOLDS:
        mask = folds == fold
        reference = _score(target[mask], k17[mask], categories[mask])["score"]
        routed = _score(target[mask], candidate[mask], categories[mask])["score"]
        per_fold[str(fold)] = {
            "reference": reference,
            "candidate": routed,
            "delta": routed - reference,
        }
    per_category = {
        category: candidate_metric["per_category"][category]
        - reference_metric["per_category"][category]
        for category in sorted(reference_metric["per_category"])
    }
    per_routed_category_fold: dict[str, dict[str, float]] = {}
    for category in categories_to_route:
        fold_deltas: dict[str, float] = {}
        for fold in FOLDS:
            mask = (categories == category) & (folds == fold)
            fold_deltas[str(fold)] = (
                _score(
                    target[mask],
                    candidate[mask],
                    categories[mask],
                    expected_categories=1,
                )["score"]
                - _score(
                    target[mask],
                    k17[mask],
                    categories[mask],
                    expected_categories=1,
                )["score"]
            )
        per_routed_category_fold[category] = fold_deltas

    reference_stress = _stress_scores(frame, target, k17, categories)
    candidate_stress = _stress_scores(frame, target, candidate, categories)
    stress_delta = {
        name: candidate_stress[name] - reference_stress[name]
        for name in reference_stress
    }
    uplift = candidate_metric["score"] - reference_metric["score"]
    positive_folds = sum(record["delta"] > 0.0 for record in per_fold.values())
    max_regression = float(contract["release_gate"]["maximum_registered_stress_regression"])
    gates = {
        "positive_aggregate_macro_average_precision": uplift > 0.0,
        "minimum_positive_folds": positive_folds
        >= int(contract["release_gate"]["minimum_positive_folds"]),
        "registered_stress_within_regression_budget": all(
            value >= -max_regression for value in stress_delta.values()
        ),
    }

    args.output_dir.mkdir(parents=True, exist_ok=True)
    frame.select("id1", "id2", "target", "category", "fold").with_columns(
        pl.Series("predict", candidate)
    ).write_parquet(prediction_path, compression="zstd", statistics=True)
    report = {
        "schema_version": 1,
        "protocol": (
            "Preregistered 75/25 fold-local category-rank residual on three weak "
            "categories; the other 17 categories preserve immutable K17 predictions exactly."
        ),
        "rows": frame.height,
        "routed_rows": int(routed_mask.sum()),
        "route": route,
        "reference": reference_metric,
        "candidate": candidate_metric,
        "candidate_minus_reference": {
            "macro_average_precision": uplift,
            "per_fold": per_fold,
            "positive_folds": positive_folds,
            "per_category": per_category,
            "per_routed_category_fold": per_routed_category_fold,
            "stress": stress_delta,
        },
        "gate": {**gates, "pass": all(gates.values())},
        "invariants": {
            "non_routed_exact_k17_parity": True,
            "outer_folds_used_for_selection": False,
            "public_used_for_selection": False,
        },
        "sources": {
            "contract": {
                "path": str(args.contract.resolve()),
                "sha256": _sha256(args.contract),
            },
            "k17_oof": {
                "path": str(args.k17_oof.resolve()),
                "sha256": actual_k17_hash,
            },
            "teacher_student_oof": {
                "path": str(args.teacher_student_oof.resolve()),
                "sha256": _sha256(args.teacher_student_oof),
            },
        },
        "output": {
            "path": str(prediction_path.resolve()),
            "sha256": _sha256(prediction_path),
        },
    }
    report_path.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(report_path)
    print(json.dumps(report["candidate_minus_reference"], ensure_ascii=False, indent=2))
    print(json.dumps(report["gate"], ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
