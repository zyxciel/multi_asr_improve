from __future__ import annotations

from stage2_asr.agreement import all_hyps_agree, char_error_rate
from stage2_asr.pinyin_util import pinyin_edit_distance, pinyin_equal, to_pinyin
from stage2_asr.types import Edit, Hypothesis
from stage2_asr.validators import validate_edits_span_local, validate_judgment_schema


def test_cer_zero_agreement():
    hyps = [
        Hypothesis("moss", "你好世界"),
        Hypothesis("qwen", "你好，世界"),
        Hypothesis("firered", "你好世界。"),
    ]
    assert all_hyps_agree(hyps)
    assert char_error_rate("你好", "你好") == 0.0


def test_cer_disagree():
    hyps = [Hypothesis("moss", "产用"), Hypothesis("qwen", "采用")]
    assert not all_hyps_agree(hyps)


def test_pinyin_helpers():
    assert pinyin_equal("产用", "采用") is False
    assert to_pinyin("采用")
    assert pinyin_edit_distance("单方接", "单框架") <= 3


def test_span_local_validator():
    ok, _ = validate_edits_span_local([Edit("产用", "采用", "B")])
    assert ok
    bad, err = validate_edits_span_local([Edit("模型", "大语言模型", "C", anchor="hotword")])
    assert not bad
    assert err


def test_schema_requires_tier_c_anchor():
    ok, _ = validate_judgment_schema(
        {"text": "x", "base_model": "moss", "edits": [{"span_asr": "a", "span_out": "b", "tier": "C"}]}
    )
    assert not ok
    ok2, _ = validate_judgment_schema(
        {
            "text": "x",
            "base_model": "moss",
            "edits": [{"span_asr": "产用", "span_out": "采用", "tier": "C", "anchor": "hotword"}],
        }
    )
    assert ok2
