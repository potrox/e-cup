from __future__ import annotations

import polars as pl

from ecup_matching.neural_ensemble import evaluate_neural_ensemble


def _frame() -> pl.DataFrame:
    rows: list[dict[str, object]] = []
    pair = 0
    for fold in range(5):
        for category_index in range(20):
            category = f"category-{category_index:02d}"
            for within_category, target in enumerate((0.0, 1.0, 0.0, 1.0)):
                rows.append(
                    {
                        "id1": pair * 2,
                        "id2": pair * 2 + 1,
                        "target": target,
                        "category": category,
                        "fold": fold,
                        "base_score": (0.4, 0.6, 0.9, 0.1)[within_category],
                        "small_score": target,
                        "surface_similarity": 1.0 if target == 0.0 else 0.0,
                        "identity_conflicts": 1,
                        "hard_negative_min_similarity": 0.5,
                        "hard_positive_max_similarity": 0.5,
                    }
                )
                pair += 1
    return pl.DataFrame(rows)


def test_neural_ensemble_cross_fits_weights_and_keeps_forecast_separate() -> None:
    report, predictions = evaluate_neural_ensemble(
        frame=_frame(),
        step=0.25,
        minimum_uplift=0.001,
        public_correction=0.279,
        forecast_uncertainty=0.02,
        candidate_name="candidate_model",
    )

    assert set(predictions) == {"raw_logit", "category_rank"}
    assert report["selection"]["winner"] in predictions
    for method in report["methods"].values():
        assert method["feature_order"] == ["base", "candidate_model"]
        assert method["gates"]["promote"] is True
        assert method["candidate_minus_base"]["positive_folds"] == 5
        assert method["candidate_minus_base"]["nonnegative_categories"] == 20
        assert method["final_deployment_weights"][1] > 0.0
        assert method["public_forecast"]["used_for_promotion"] is False
        assert method["public_forecast"]["point"] == (
            method["local_oof"]["score"] - 0.279
        )
