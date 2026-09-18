from __future__ import annotations

import pytest

from ecup_matching.teacher_distillation import (
    build_teacher_user_prompt,
    choose_binary_token_ids,
    corrected_teacher_target,
)


class _Tokenizer:
    def encode(self, text: str, *, add_special_tokens: bool) -> list[int]:
        assert not add_special_tokens
        return {"0": [10], "1": [11]}.get(text, [1, 2])


def test_teacher_prompt_is_symmetric_by_explicit_reversal() -> None:
    arguments = {
        "category": "Обувь",
        "left_name": "Модель A, размер 42",
        "left_attributes": '{"Размер":"42"}',
        "right_name": "Модель A, размер 43",
        "right_attributes": '{"Размер":"43"}',
        "mode": "category_rules",
    }
    forward = build_teacher_user_prompt(**arguments)
    backward = build_teacher_user_prompt(**arguments, reverse=True)
    assert "Разный размер" in forward
    assert "модель a, размер 42" in forward
    assert "модель a, размер 43" in backward
    assert forward != backward


def test_corrected_teacher_target_is_bounded_blend() -> None:
    assert corrected_teacher_target(0.0, 1.0, teacher_weight=0.75) == pytest.approx(0.75)
    assert corrected_teacher_target(1.0, 0.0, teacher_weight=0.75) == pytest.approx(0.25)
    with pytest.raises(ValueError):
        corrected_teacher_target(0.0, 1.1)


def test_binary_token_ids_require_distinct_single_tokens() -> None:
    assert choose_binary_token_ids(_Tokenizer()) == {
        "negative": 10,
        "positive": 11,
        "negative_text": "0",
        "positive_text": "1",
    }
