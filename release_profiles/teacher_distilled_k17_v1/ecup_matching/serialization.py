"""Deterministic item and pair text serialization for neural product matching."""

from __future__ import annotations

import re
import unicodedata
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from decimal import Decimal, InvalidOperation
from itertools import zip_longest
from typing import Any

import orjson

EXACT_PRODUCT_INSTRUCTION = (
    "Оцени, являются ли две карточки одним и тем же вариантом товара: бренд, модель, "
    "артикул, размер, цвет, объём и количество в упаковке должны совпадать."
)
ITEM_SERIALIZER_VERSION = "item_v1"
PAIR_SERIALIZER_VERSION = "pair_v2.1"

_CONTROL = re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]+")
_WHITESPACE = re.compile(r"\s+")
_NON_ALNUM_SPACE = re.compile(r"[^0-9a-zа-я]+")
_DECIMAL_COMMA = re.compile(r"(?<=\d),(?=\d)")
_TITLE_IDENTIFIER = re.compile(r"(?<![0-9a-zа-я])[0-9a-zа-я._/-]*\d[0-9a-zа-я._/-]*(?![0-9a-zа-я])")
_HIGH_PRIORITY_ATTRIBUTE_FRAGMENTS = (
    "бренд",
    "brand",
    "производитель",
    "manufacturer",
    "артикул",
    "модель",
    "model",
    "mpn",
    "партномер",
    "код товара",
    "тип",
    "вид товара",
    "размер",
    "объем",
    "объём",
    "вес",
    "количество",
    "цвет",
)

_IDENTITY_ATTRIBUTE_GROUPS: tuple[tuple[str, tuple[str, ...]], ...] = (
    ("brand", ("бренд", "brand", "производитель", "manufacturer")),
    ("model", ("модель", "model", "mpn", "партномер", "part number")),
    ("article", ("артикул", "article", "sku", "код товара")),
    ("size", ("размер", "size", "диаметр", "длина", "ширина", "высота")),
    ("volume", ("объем", "объём", "volume", "литраж", "емкость", "ёмкость")),
    ("weight", ("вес", "weight", "масса")),
    ("pack", ("количество", "упаков", "комплект", "набор", "шт", "pack")),
    ("color", ("цвет", "color")),
)

_UNIT_ALIASES: tuple[tuple[re.Pattern[str], str], ...] = (
    (re.compile(r"\b(?:миллилитр(?:а|ов)?|мл|ml)\b"), "ml"),
    (re.compile(r"\b(?:литр(?:а|ов)?|л|l)\b"), "l"),
    (re.compile(r"\b(?:килограмм(?:а|ов)?|кг|kg)\b"), "kg"),
    (re.compile(r"\b(?:грамм(?:а|ов)?|гр|г|g)\b"), "g"),
    (re.compile(r"\b(?:миллиметр(?:а|ов)?|мм|mm)\b"), "mm"),
    (re.compile(r"\b(?:сантиметр(?:а|ов)?|см|cm)\b"), "cm"),
    (re.compile(r"\b(?:метр(?:а|ов)?|м|m)\b"), "m"),
    (re.compile(r"\b(?:штук(?:а|и)?|шт|pcs?)\b"), "pcs"),
)
_MEASUREMENT = re.compile(
    r"(?<![0-9a-zа-я])(?P<value>\d+(?:\.\d+)?)\s*"
    r"(?P<unit>ml|мл|kg|кг|mm|мм|cm|см|pcs|шт|гр|g|г|l|л|m|м)\b"
)
_UNIT_CANONICAL = {
    "мл": "ml",
    "л": "l",
    "кг": "kg",
    "гр": "g",
    "г": "g",
    "мм": "mm",
    "см": "cm",
    "м": "m",
    "шт": "pcs",
}
_BASE_UNIT: dict[str, tuple[str, Decimal]] = {
    "ml": ("ml", Decimal(1)),
    "l": ("ml", Decimal(1_000)),
    "g": ("g", Decimal(1)),
    "kg": ("g", Decimal(1_000)),
    "mm": ("mm", Decimal(1)),
    "cm": ("mm", Decimal(10)),
    "m": ("mm", Decimal(1_000)),
    "pcs": ("pcs", Decimal(1)),
}


@dataclass(frozen=True, slots=True)
class PairSerialization:
    """A canonical model input plus label-free comparison diagnostics."""

    text: str
    identity_matches: int
    identity_conflicts: int
    missing_identity_fields: int


def normalize_neural_text(value: object) -> str:
    """Normalize Unicode and whitespace while preserving punctuation and decimals."""

    if value is None:
        return ""
    text = unicodedata.normalize("NFKC", str(value)).casefold().replace("ё", "е")
    text = _CONTROL.sub(" ", text)
    return _WHITESPACE.sub(" ", text).strip()


def _stable_value(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, str | int | float | bool):
        return normalize_neural_text(value)
    try:
        serialized = orjson.dumps(value, option=orjson.OPT_SORT_KEYS).decode("utf-8")
    except TypeError:
        serialized = str(value)
    return normalize_neural_text(serialized)


def _attribute_priority(item: tuple[str, str]) -> tuple[int, str, str]:
    key, value = item
    priority = next(
        (
            index
            for index, fragment in enumerate(_HIGH_PRIORITY_ATTRIBUTE_FRAGMENTS)
            if fragment in key
        ),
        len(_HIGH_PRIORITY_ATTRIBUTE_FRAGMENTS),
    )
    return priority, key, value


def _normalized_attribute_payload(
    raw_attributes: str | None,
) -> tuple[tuple[tuple[str, str], ...], str]:
    if not raw_attributes:
        return (), "missing"
    try:
        parsed = orjson.loads(raw_attributes)
    except (orjson.JSONDecodeError, TypeError):
        return (), "malformed"
    if not isinstance(parsed, Mapping):
        return (), "malformed"

    normalized: list[tuple[str, str]] = []
    for raw_key, raw_value in parsed.items():
        key = normalize_neural_text(raw_key)
        value = _stable_value(raw_value)
        if key and value:
            normalized.append((key, value[:512]))
    state = "present" if normalized else "empty"
    return tuple(sorted(normalized, key=_attribute_priority)), state


def normalized_attributes(raw_attributes: str | None) -> tuple[tuple[str, str], ...]:
    """Parse an attribute JSON object into a stable, priority-ordered sequence."""

    return _normalized_attribute_payload(raw_attributes)[0]


def serialize_item(
    category: str | None,
    name: str | None,
    raw_attributes: str | None,
    *,
    max_attribute_characters: int = 4_096,
) -> str:
    """Serialize one product card with high-value identifiers before long-tail fields."""

    if max_attribute_characters <= 0:
        raise ValueError("max_attribute_characters must be positive")

    normalized_category = normalize_neural_text(category)
    normalized_name = normalize_neural_text(name)
    lines = [f"категория: {normalized_category}", f"название: {normalized_name}"]
    used_characters = 0
    for key, value in normalized_attributes(raw_attributes):
        attribute = f"{key}: {value}"
        remaining = max_attribute_characters - used_characters
        if remaining <= 0:
            break
        lines.append(attribute[:remaining])
        used_characters += min(len(attribute), remaining)
    return "\n".join(lines)


def build_model_query(item_text: str, prompt_mode: str) -> str:
    """Build a query-side text for native relevance or exact-product judgment."""

    if prompt_mode == "native":
        return item_text
    if prompt_mode == "exact":
        return f"{EXACT_PRODUCT_INSTRUCTION}\n\nпервая карточка:\n{item_text}"
    raise ValueError(f"unknown prompt mode: {prompt_mode}")


def serialize_items(
    categories: Sequence[str],
    names: Sequence[str],
    attributes: Sequence[str],
    *,
    max_attribute_characters: int = 4_096,
) -> list[str]:
    """Serialize aligned columns without row-wise dataframe iteration."""

    if not (len(categories) == len(names) == len(attributes)):
        raise ValueError("item columns must have equal lengths")
    return [
        serialize_item(
            category,
            name,
            raw_attributes,
            max_attribute_characters=max_attribute_characters,
        )
        for category, name, raw_attributes in zip(categories, names, attributes, strict=True)
    ]


def _comparison_value(value: str) -> str:
    normalized = _DECIMAL_COMMA.sub(".", normalize_neural_text(value))
    for pattern, replacement in _UNIT_ALIASES:
        normalized = pattern.sub(replacement, normalized)

    def normalize_measurement(match: re.Match[str]) -> str:
        try:
            numeric = Decimal(match.group("value"))
        except InvalidOperation:
            return match.group(0)
        unit = _UNIT_CANONICAL.get(match.group("unit"), match.group("unit"))
        base_unit, multiplier = _BASE_UNIT[unit]
        base_value = (numeric * multiplier).normalize()
        rendered = format(base_value, "f")
        if "." in rendered:
            rendered = rendered.rstrip("0").rstrip(".")
        return f"{rendered or '0'} {base_unit}"

    normalized = _MEASUREMENT.sub(normalize_measurement, normalized)
    normalized = _NON_ALNUM_SPACE.sub(" ", normalized)
    return _WHITESPACE.sub(" ", normalized).strip()


def _canonical_conflict(left_values: set[str], right_values: set[str]) -> str:
    first, second = sorted(("|".join(sorted(left_values)), "|".join(sorted(right_values))))
    return f"{first}<>{second}"


def _identity_values(
    attributes: tuple[tuple[str, str], ...],
) -> dict[str, tuple[str, ...]]:
    grouped: dict[str, set[str]] = {name: set() for name, _ in _IDENTITY_ATTRIBUTE_GROUPS}
    for key, raw_value in attributes:
        value = _comparison_value(raw_value)
        if not value:
            continue
        for group, fragments in _IDENTITY_ATTRIBUTE_GROUPS:
            if any(fragment in key for fragment in fragments):
                grouped[group].add(value[:96])
                break
    return {group: tuple(sorted(values)[:3]) for group, values in grouped.items() if values}


def _title_identifiers(name: str | None) -> tuple[str, ...]:
    normalized = normalize_neural_text(name)
    identifiers = {
        _NON_ALNUM_SPACE.sub("", match.group(0)) for match in _TITLE_IDENTIFIER.finditer(normalized)
    }
    return tuple(sorted(value for value in identifiers if len(value) >= 3))


def _marked_item_sections(
    name: str | None,
    attributes: tuple[tuple[str, str], ...],
    attribute_state: str,
    *,
    max_attribute_characters: int,
) -> tuple[str, str, tuple[str, ...], str]:
    title = (normalize_neural_text(name) or "[MISSING]")[:384]
    lines: list[str] = []
    used_characters = 0
    for key, value in attributes:
        attribute = f"{key} [VALUE] {value}"
        remaining = max_attribute_characters - used_characters
        if remaining <= 0:
            break
        lines.append(attribute[:remaining])
        used_characters += min(len(attribute), remaining)
    frozen_lines = tuple(lines)
    sort_key = "\n".join((title, attribute_state, *frozen_lines))
    return title, attribute_state, frozen_lines, sort_key


def serialize_pair(
    category: str | None,
    left_name: str | None,
    left_attributes: str | None,
    right_name: str | None,
    right_attributes: str | None,
    *,
    max_attribute_characters: int = 4_096,
) -> PairSerialization:
    """Build a canonical pair-aware input without using labels or learned scores."""

    if max_attribute_characters <= 0:
        raise ValueError("max_attribute_characters must be positive")

    left_parsed, left_state = _normalized_attribute_payload(left_attributes)
    right_parsed, right_state = _normalized_attribute_payload(right_attributes)
    left_identity = _identity_values(left_parsed)
    right_identity = _identity_values(right_parsed)

    matches: list[str] = []
    conflicts: list[str] = []
    missing: list[str] = []
    for group, _ in _IDENTITY_ATTRIBUTE_GROUPS:
        left_values = set(left_identity.get(group, ()))
        right_values = set(right_identity.get(group, ()))
        shared = sorted(left_values & right_values)
        if shared:
            matches.append(f"{group}={'|'.join(shared)}")
        elif left_values and right_values:
            conflicts.append(f"{group}={_canonical_conflict(left_values, right_values)}")
        elif left_values or right_values:
            missing.append(group)

    left_title_ids = set(_title_identifiers(left_name))
    right_title_ids = set(_title_identifiers(right_name))
    shared_title_ids = sorted(left_title_ids & right_title_ids)
    if shared_title_ids:
        matches.append(f"title_id={'|'.join(shared_title_ids[:8])}")
    elif left_title_ids and right_title_ids:
        conflicts.append(f"title_id={_canonical_conflict(left_title_ids, right_title_ids)}")

    left_sections = _marked_item_sections(
        left_name,
        left_parsed,
        left_state,
        max_attribute_characters=max_attribute_characters,
    )
    right_sections = _marked_item_sections(
        right_name,
        right_parsed,
        right_state,
        max_attribute_characters=max_attribute_characters,
    )
    if right_sections[3] < left_sections[3]:
        left_sections, right_sections = right_sections, left_sections

    left_title, left_attribute_state, left_lines, _ = left_sections
    right_title, right_attribute_state, right_lines, _ = right_sections

    header = [
        f"[CATEGORY] {normalize_neural_text(category) or '[MISSING]'}",
        f"[LEFT] [TITLE] {left_title}",
        f"[RIGHT] [TITLE] {right_title}",
        f"[MATCH] {'; '.join(matches) if matches else 'none'}",
        f"[CONFLICT] {'; '.join(conflicts) if conflicts else 'none'}",
        f"[MISSING_IDENTITY] {','.join(missing) if missing else 'none'}",
        f"[LEFT] [ATTRIBUTES] {left_attribute_state}",
        f"[RIGHT] [ATTRIBUTES] {right_attribute_state}",
    ]
    body: list[str] = []
    for left_line, right_line in zip_longest(left_lines, right_lines):
        if left_line is not None:
            body.append(f"[LEFT] [ATTR] {left_line}")
        if right_line is not None:
            body.append(f"[RIGHT] [ATTR] {right_line}")
    text = "\n".join((*header, *body))
    return PairSerialization(
        text=text,
        identity_matches=len(matches),
        identity_conflicts=len(conflicts),
        missing_identity_fields=len(missing),
    )


def serialize_pairs(
    categories: Sequence[str],
    left_names: Sequence[str],
    left_attributes: Sequence[str],
    right_names: Sequence[str],
    right_attributes: Sequence[str],
    *,
    max_attribute_characters: int = 4_096,
) -> list[PairSerialization]:
    """Serialize aligned product pairs with a canonical symmetric representation."""

    lengths = {
        len(categories),
        len(left_names),
        len(left_attributes),
        len(right_names),
        len(right_attributes),
    }
    if len(lengths) != 1:
        raise ValueError("pair columns must have equal lengths")
    return [
        serialize_pair(
            category,
            left_name,
            left_raw,
            right_name,
            right_raw,
            max_attribute_characters=max_attribute_characters,
        )
        for category, left_name, left_raw, right_name, right_raw in zip(
            categories,
            left_names,
            left_attributes,
            right_names,
            right_attributes,
            strict=True,
        )
    ]
