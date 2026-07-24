from __future__ import annotations

from stage2_asr.types import Edit
from stage2_asr.validators import (
    validate_edits_evidence_ladder,
    validate_edits_span_local,
    validate_judgment_schema,
)


def test_span_local_rejects_expansion():
    ok, _ = validate_edits_span_local([Edit("模型", "大语言模型", "C", anchor="hotword")])
    assert ok is False


def test_tier_b_requires_exact_pinyin():
    ok, _ = validate_edits_evidence_ladder([Edit("产用", "采用", "B")])
    assert ok is False
    ok2, _ = validate_edits_evidence_ladder([Edit("帐号", "账号", "B")])
    assert ok2 is True


def test_tier_c_requires_anchor_and_fuzzy_pinyin():
    ok, _ = validate_edits_evidence_ladder([Edit("单方接", "单框架", "C")])
    assert ok is False
    ok2, _ = validate_edits_evidence_ladder(
        [Edit("单方接", "单框架", "C", anchor="neighbor_draft")]
    )
    assert ok2 is True


def test_schema():
    assert validate_judgment_schema({"text": "a", "base_model": "moss", "edits": []})[0]
    assert not validate_judgment_schema({"text": "a"})[0]
