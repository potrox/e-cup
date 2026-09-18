"""Normalize and validate immutable ensemble release evidence."""

from __future__ import annotations

from typing import Any


def resolve_category_router(
    report: dict[str, Any],
    *,
    selected_count: int,
) -> tuple[str, dict[str, Any]]:
    method_name = (
        "rumodernbert_base+lamar_600m__raw_logit__"
        f"top_{selected_count:02d}_category_router"
    )
    method = report.get("methods", {}).get(method_name)
    if not isinstance(method, dict):
        best = report.get("selection", {}).get("best")
        if isinstance(best, dict) and int(best.get("method", {}).get(
            "selected_category_count", -1
        )) == selected_count:
            method = best.get("method")
    if not isinstance(method, dict):
        raise ValueError("requested category router is missing from frozen evidence")
    if method.get("quality_gates", {}).get("pass") is not True:
        raise ValueError("requested category router did not pass frozen quality gates")
    if method.get("feature_order") != ["rumodernbert_base", "lamar_600m"]:
        raise ValueError("category router feature order mismatch")
    if int(method.get("selected_category_count", -1)) != selected_count:
        raise ValueError("category router K mismatch")
    return method_name, method


def resolve_length_router(report: dict[str, Any]) -> dict[str, Any]:
    delta = report.get("candidate_minus_reference", {})
    stress = delta.get("stress", {})
    if not isinstance(delta, dict) or not isinstance(stress, dict) or not stress:
        raise ValueError("length router delta evidence is missing")

    if report.get("config", {}).get("base_column") == "k16":
        categories = report.get("final_deployment_categories")
        invariants = report.get("invariants", {})
        valid = (
            report.get("config", {}).get("candidate_column") == "k16_l384"
            and isinstance(categories, list)
            and bool(categories)
            and int(delta.get("positive_folds", 0)) >= 4
            and int(delta.get("nonnegative_categories", 0)) >= 16
            and all(float(value) >= 0.0 for value in stress.values())
            and invariants.get("non_routed_exact_base_parity") is True
            and invariants.get("public_used_for_selection") is False
        )
        local_oof = report.get("candidate", {}).get("score")
    else:
        length_route = report.get("length_route", {})
        invariants = report.get("invariants", {})
        categories = length_route.get("categories")
        valid = (
            int(length_route.get("max_length", 0)) == 384
            and length_route.get("fixed_before_evaluation") is True
            and isinstance(categories, list)
            and bool(categories)
            and len(categories) == len(set(categories))
            and report.get("gate", {}).get("pass") is True
            and int(delta.get("positive_folds", 0)) >= 4
            and int(delta.get("nonnegative_categories", 0)) >= 16
            and all(float(value) >= 0.0 for value in stress.values())
            and invariants.get("public_used_for_selection") is False
            and float(report.get("parity", {}).get(
                "max_abs_reconstruction_delta", 1.0
            )) <= 1e-12
        )
        local_oof = report.get("candidate", {}).get("score")
    if not valid or not isinstance(local_oof, (int, float)):
        raise ValueError("base length router did not pass frozen quality gates")
    return {
        "categories": [str(category) for category in categories],
        "local_oof": float(local_oof),
        "uplift": float(delta["macro_average_precision"]),
        "positive_folds": int(delta["positive_folds"]),
    }
