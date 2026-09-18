import numpy as np
import pytest

from ecup_matching.metrics import macro_average_precision


def test_macro_average_precision_is_unweighted_across_categories() -> None:
    report = macro_average_precision(
        y_true=[1, 0, 1, 0, 1, 0],
        y_score=[1.0, 0.0, 0.9, 0.8, 0.7, 0.6],
        categories=["a", "a", "b", "b", "b", "b"],
    )

    assert report.per_category["a"] == pytest.approx(1.0)
    assert report.per_category["b"] == pytest.approx((1.0 + 2.0 / 3.0) / 2.0)
    assert report.score == pytest.approx((1.0 + report.per_category["b"]) / 2.0)


@pytest.mark.parametrize(
    ("y_true", "y_score", "categories"),
    [
        ([0, 1], [0.1], ["a", "a"]),
        ([0, 2], [0.1, 0.2], ["a", "a"]),
        ([0, 1], [0.1, np.nan], ["a", "a"]),
        ([0, 1], [0.1, 0.2], ["a", None]),
    ],
)
def test_macro_average_precision_rejects_invalid_inputs(
    y_true: list[object],
    y_score: list[float],
    categories: list[object],
) -> None:
    with pytest.raises(ValueError):
        macro_average_precision(y_true, y_score, categories)
