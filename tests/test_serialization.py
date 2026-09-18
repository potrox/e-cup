import pytest

from ecup_matching.serialization import (
    EXACT_PRODUCT_INSTRUCTION,
    build_model_query,
    normalize_neural_text,
    normalized_attributes,
    serialize_item,
    serialize_pair,
    serialize_pairs,
)


def test_neural_normalization_preserves_decimal_and_units() -> None:
    assert normalize_neural_text("  Ёмкость\t2.5 л\n") == "емкость 2.5 л"


def test_attribute_serialization_is_stable_and_prioritizes_identifiers() -> None:
    first = serialize_item(
        "Электроника",
        "Телефон X100",
        '{"цвет":"Черный","прочее":"NFC","модель":"X100","бренд":"Acme"}',
    )
    second = serialize_item(
        "Электроника",
        "Телефон X100",
        '{"прочее":"NFC","бренд":"Acme","модель":"X100","цвет":"Черный"}',
    )

    assert first == second
    assert first.index("бренд: acme") < first.index("модель: x100")
    assert first.index("модель: x100") < first.index("цвет: черный")
    assert first.index("цвет: черный") < first.index("прочее: nfc")


def test_attributes_fail_soft_on_invalid_or_non_object_json() -> None:
    assert normalized_attributes("not-json") == ()
    assert normalized_attributes("[]") == ()


def test_serialize_item_rejects_invalid_character_budget() -> None:
    with pytest.raises(ValueError):
        serialize_item("a", "b", "{}", max_attribute_characters=0)


def test_exact_query_is_explicit_and_native_query_is_unchanged() -> None:
    item = "категория: обувь\nназвание: ботинки"

    assert build_model_query(item, "native") == item
    assert EXACT_PRODUCT_INSTRUCTION in build_model_query(item, "exact")
    with pytest.raises(ValueError):
        build_model_query(item, "unknown")


def test_pair_serialization_is_symmetric_and_marks_identity_evidence() -> None:
    left = {
        "name": "Смартфон ACME X100 500 мл",
        "attributes": '{"Бренд":"ACME","Модель":"X100","Объём":"0.5 л"}',
    }
    right = {
        "name": "ACME X100 смартфон 500ml",
        "attributes": '{"Объем":"500 мл","Модель":"X100","Бренд":"acme"}',
    }

    forward = serialize_pair(
        "Электроника",
        left["name"],
        left["attributes"],
        right["name"],
        right["attributes"],
    )
    reverse = serialize_pair(
        "Электроника",
        right["name"],
        right["attributes"],
        left["name"],
        left["attributes"],
    )

    assert forward == reverse
    assert "brand=acme" in forward.text
    assert "model=x100" in forward.text
    assert "volume=500 ml" in forward.text
    assert forward.identity_matches >= 3
    assert forward.identity_conflicts == 0
    assert forward.text.index("[LEFT] [TITLE]") < forward.text.index("[LEFT] [ATTR]")
    assert forward.text.index("[RIGHT] [TITLE]") < forward.text.index("[LEFT] [ATTR]")


def test_pair_serialization_canonicalizes_conflicts_and_attribute_states() -> None:
    forward = serialize_pair(
        "Красота",
        "Крем ACME A100 50 мл",
        '{"модель":"A100","объем":"50 мл"}',
        "Крем ACME A200 100 мл",
        "not-json",
    )
    reverse = serialize_pair(
        "Красота",
        "Крем ACME A200 100 мл",
        "not-json",
        "Крем ACME A100 50 мл",
        '{"объем":"50 мл","модель":"A100"}',
    )

    assert forward == reverse
    assert "[CONFLICT]" in forward.text
    assert "title_id=" in forward.text
    assert "[ATTRIBUTES] malformed" in forward.text
    assert "[MISSING_IDENTITY] model,volume" in forward.text
    assert forward.identity_conflicts >= 1
    assert forward.missing_identity_fields == 2


def test_pair_serialization_validates_alignment_and_character_budget() -> None:
    with pytest.raises(ValueError):
        serialize_pair("a", "b", "{}", "c", "{}", max_attribute_characters=0)
    with pytest.raises(ValueError):
        serialize_pairs(["a"], ["b"], ["{}"], ["c", "d"], ["{}"])


def test_pair_serialization_normalizes_units_without_spaces() -> None:
    pair = serialize_pair(
        "Красота",
        "Крем 500мл 10шт",
        '{"объем":"500мл","количество":"10шт"}',
        "Крем 0,5л 10 шт",
        '{"объём":"0,5л","количество":"10 шт"}',
    )

    assert "volume=500 ml" in pair.text
    assert "pack=10 pcs" in pair.text
    assert pair.identity_conflicts == 0
