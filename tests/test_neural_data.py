from __future__ import annotations

import duckdb
import pytest

from ecup_matching.neural_data import (
    NeuralDataConfig,
    llm_split_expression,
    surface_similarity_expression,
)


def test_llm_split_is_item_disjoint() -> None:
    connection = duckdb.connect()
    connection.execute("CREATE TABLE pairs(id1 BIGINT, id2 BIGINT)")
    connection.execute("INSERT INTO pairs VALUES (1, 2), (2, 3), (3, 4), (4, 5), (5, 6), (6, 1)")
    expression = llm_split_expression(modulus=3, validation_bucket=0)
    rows = connection.execute(f"SELECT id1, id2, {expression} AS split FROM pairs m").fetchall()

    train_items: set[int] = set()
    validation_items: set[int] = set()
    for id1, id2, split in rows:
        if split == "train":
            train_items.update((id1, id2))
        elif split == "validation":
            validation_items.update((id1, id2))

    assert train_items.isdisjoint(validation_items)


def test_llm_split_validates_parameters() -> None:
    with pytest.raises(ValueError, match="at least 2"):
        llm_split_expression(modulus=1)
    with pytest.raises(ValueError, match="outside"):
        llm_split_expression(modulus=5, validation_bucket=5)


def test_surface_similarity_is_label_free_and_bounded() -> None:
    connection = duckdb.connect()
    expression = surface_similarity_expression(left_name="left_name", right_name="right_name")
    similarity = connection.execute(
        f"SELECT {expression} FROM (VALUES ('ACME X100', 'ACME X200')) AS t(left_name, right_name)"
    ).fetchone()[0]

    assert 0.0 < similarity < 1.0


def test_surface_similarity_handles_empty_and_single_character_names() -> None:
    connection = duckdb.connect()
    expression = surface_similarity_expression(left_name="left_name", right_name="right_name")
    rows = connection.execute(
        f"SELECT {expression} FROM (VALUES ('', 'A'), ('A', 'a'), ('A', 'B')) "
        "AS t(left_name, right_name)"
    ).fetchall()

    assert [row[0] for row in rows] == [0.0, 1.0, 0.0]


def test_neural_data_config_rejects_invalid_sampling_parameters() -> None:
    with pytest.raises(ValueError, match="at least 2"):
        NeuralDataConfig(llm_validation_modulus=1).validate()
    with pytest.raises(ValueError, match="positive"):
        NeuralDataConfig(llm_train_sample_modulus=0).validate()
