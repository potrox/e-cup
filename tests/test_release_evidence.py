from __future__ import annotations

from ecup_matching.release_evidence import (
    resolve_category_router,
    resolve_length_router,
)


def test_resolves_aggressive_router_evidence() -> None:
    method = {
        "quality_gates": {"pass": True},
        "feature_order": ["rumodernbert_base", "lamar_600m"],
        "selected_category_count": 17,
    }
    name, resolved = resolve_category_router(
        {"selection": {"best": {"method": method}}}, selected_count=17
    )

    assert name.endswith("top_17_category_router")
    assert resolved is method


def test_resolves_fixed_length_router_evidence() -> None:
    evidence = resolve_length_router(
        {
            "length_route": {
                "max_length": 384,
                "categories": ["Мебель", "Обувь", "Одежда"],
                "fixed_before_evaluation": True,
            },
            "candidate": {"score": 0.8},
            "candidate_minus_reference": {
                "macro_average_precision": 0.001,
                "positive_folds": 5,
                "nonnegative_categories": 20,
                "stress": {"a": 0.0},
            },
            "gate": {"pass": True},
            "invariants": {"public_used_for_selection": False},
            "parity": {"max_abs_reconstruction_delta": 0.0},
        }
    )

    assert evidence["categories"] == ["Мебель", "Обувь", "Одежда"]
    assert evidence["positive_folds"] == 5
