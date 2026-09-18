"""Prompting and target construction for local product-matching teachers."""

from __future__ import annotations

import math
from collections.abc import Mapping
from typing import Literal

from ecup_matching.serialization import normalize_neural_text, serialize_item

TeacherPromptMode = Literal["strict", "category_rules"]

TEACHER_PROMPT_VERSION = "qwen_exact_product_v1"

_CATEGORY_RULES: dict[str, str] = {
    "одежда": (
        "Для одежды особенно проверяй вид изделия, бренд/модель, пол, размер, цвет, "
        "материал и комплектность. Разный размер или цвет означает разные варианты товара."
    ),
    "обувь": (
        "Для обуви особенно проверяй бренд/модель, пол, размер, цвет, материал и сезон. "
        "Разный размер или цвет означает разные варианты товара."
    ),
    "ювелирные изделия": (
        "Для ювелирных изделий особенно проверяй тип, металл, пробу, вставку, размер, "
        "вес, артикул и комплектность. Различие любого вариантного параметра означает "
        "разные товары."
    ),
}


def category_rule(category: object) -> str:
    """Return a stable category-specific exact-identity rule when one is registered."""

    return _CATEGORY_RULES.get(normalize_neural_text(category), "")


def build_teacher_user_prompt(
    *,
    category: object,
    left_name: object,
    left_attributes: object,
    right_name: object,
    right_attributes: object,
    mode: TeacherPromptMode,
    reverse: bool = False,
    max_attribute_characters: int = 1_600,
) -> str:
    """Build one deterministic prompt without labels, IDs, or learned scores."""

    if mode not in {"strict", "category_rules"}:
        raise ValueError(f"unsupported teacher prompt mode: {mode}")
    if max_attribute_characters < 1:
        raise ValueError("max_attribute_characters must be positive")

    left = serialize_item(
        str(category),
        "" if left_name is None else str(left_name),
        None if left_attributes is None else str(left_attributes),
        max_attribute_characters=max_attribute_characters,
    )
    right = serialize_item(
        str(category),
        "" if right_name is None else str(right_name),
        None if right_attributes is None else str(right_attributes),
        max_attribute_characters=max_attribute_characters,
    )
    if reverse:
        left, right = right, left

    rule = category_rule(category) if mode == "category_rules" else ""
    rule_block = f"\nПравило категории: {rule}" if rule else ""
    return (
        "Определи, описывают ли две карточки один и тот же конкретный вариант товара. "
        "Совпадение общего типа товара недостаточно. Бренд, модель, артикул, размер, "
        "цвет, объем, вес и количество в упаковке должны совпадать, если они указаны. "
        "Разные вариантные параметры означают разные товары. Не используй порядок карточек."
        f"{rule_block}\n\nКарточка A:\n{left}\n\nКарточка B:\n{right}\n\n"
        "Ответь только одним символом: 1 для одного товара, 0 для разных товаров."
    )


def corrected_teacher_target(
    organizer_target: float,
    teacher_probability: float,
    *,
    teacher_weight: float = 0.8,
) -> float:
    """Blend an independent teacher with the organizer soft label."""

    if not 0.0 <= organizer_target <= 1.0 or not math.isfinite(organizer_target):
        raise ValueError("organizer_target must be a finite probability")
    if not 0.0 <= teacher_probability <= 1.0 or not math.isfinite(teacher_probability):
        raise ValueError("teacher_probability must be a finite probability")
    if not 0.0 <= teacher_weight <= 1.0 or not math.isfinite(teacher_weight):
        raise ValueError("teacher_weight must be a finite value in [0, 1]")
    return teacher_weight * teacher_probability + (1.0 - teacher_weight) * organizer_target


def choose_binary_token_ids(tokenizer: object) -> Mapping[str, int]:
    """Resolve single-token binary labels for next-token logit scoring."""

    encode = getattr(tokenizer, "encode", None)
    if encode is None:
        raise TypeError("tokenizer must expose encode")
    for negative, positive in (("0", "1"), (" 0", " 1"), ("no", "yes")):
        negative_ids = encode(negative, add_special_tokens=False)
        positive_ids = encode(positive, add_special_tokens=False)
        if len(negative_ids) == len(positive_ids) == 1 and negative_ids[0] != positive_ids[0]:
            return {
                "negative": int(negative_ids[0]),
                "positive": int(positive_ids[0]),
                "negative_text": negative,
                "positive_text": positive,
            }
    raise ValueError("tokenizer has no supported single-token binary label pair")

