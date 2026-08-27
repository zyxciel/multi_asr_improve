from __future__ import annotations

from stage2_asr.pass_b import run_pass_b
from stage2_asr.runners.mock_llm import MockLlmJudge
from stage2_asr.types import PipelineConfig, Turn


def test_pass_b_hotword_alias():
    turns = [Turn(0, 1, "s0"), Turn(1, 2, "s1")]
    draft = {0: "需要单方接兼容", 1: "单框架已经上线"}
    out, audits = run_pass_b(turns, draft, hotwords=["单框架|单方接"])
    assert "单框架" in out[0]
    assert "单方接" not in out[0]
    assert audits
    assert all(a.get("pass") == "B" for a in audits)


def test_pass_b_llm_tier_bc_with_meeting_draft():
    turns = [
        Turn(0, 2, "s0", "系统奔至了"),
        Turn(2, 4, "s1", "刚才说蹦字问题"),
    ]
    draft = {0: "系统奔至了", 1: "刚才说蹦字问题"}

    class FixBengZhi:
        def judge(self, **kwargs):
            text = kwargs["hypotheses"][0].text
            if "奔至" in text:
                return {
                    "text": text.replace("奔至", "蹦字"),
                    "base_model": "draft",
                    "edits": [
                        {
                            "span_asr": "奔至",
                            "span_out": "蹦字",
                            "tier": "C",
                            "pinyin_asr": "benzhi",
                            "pinyin_out": "bengzi",
                            "anchor": "meeting_draft",
                        }
                    ],
                    "overlap": False,
                }
            return {
                "text": text,
                "base_model": "draft",
                "edits": [],
                "overlap": False,
            }

    out, audits = run_pass_b(
        turns,
        draft,
        hotwords=[],
        llm_judge=FixBengZhi(),
        config=PipelineConfig(),
    )
    assert "蹦字" in out[0]
    assert any(a.get("path") == "llm" and a.get("pass") == "B" for a in audits)


def test_pass_b_empty_edits_keeps_draft():
    """Pass B must not apply a whole-turn rewrite when edits is empty."""
    turns = [
        Turn(0, 2, "s0", "这一夫多妻制。"),
        Turn(2, 4, "s1", "求婚啊。"),
    ]
    draft = {0: "这一夫多妻制。", 1: "求婚啊。"}

    class EmptyEditsRewrite:
        def judge(self, **kwargs):
            text = kwargs["hypotheses"][0].text
            if "一夫多妻" in text:
                return {
                    "text": "这一夫妻子。",
                    "base_model": "draft",
                    "edits": [],
                    "overlap": False,
                }
            return {
                "text": "丑化。",
                "base_model": "draft",
                "edits": [],
                "overlap": False,
            }

    out, audits = run_pass_b(
        turns,
        draft,
        llm_judge=EmptyEditsRewrite(),
        config=PipelineConfig(),
    )
    assert out[0] == "这一夫多妻制。"
    assert out[1] == "求婚啊。"
    assert any(a.get("path") == "empty_edits_reject" for a in audits)


def test_pass_b_applies_validated_edits_not_judgment_text():
    """judgment.text is untrusted; the turn must equal draft with validated spans applied."""
    turns = [Turn(0, 2, "s0", "系统奔至了")]
    draft = {0: "系统奔至了"}

    class MismatchedText:
        def judge(self, **kwargs):
            return {
                "text": "完全无关的会议总结",
                "base_model": "draft",
                "edits": [
                    {
                        "span_asr": "奔至",
                        "span_out": "蹦字",
                        "tier": "C",
                        "pinyin_asr": "benzhi",
                        "pinyin_out": "bengzi",
                        "anchor": "meeting_draft",
                    }
                ],
                "overlap": False,
            }

    out, audits = run_pass_b(
        turns,
        draft,
        llm_judge=MismatchedText(),
        config=PipelineConfig(),
    )
    assert out[0] == "系统蹦字了"
    assert any(a.get("path") == "llm" and a.get("span_asr") == "奔至" for a in audits)


def test_pass_b_rejects_tier_a_and_keeps_draft():
    turns = [Turn(0, 2, "s0", "原文")]
    draft = {0: "原文"}

    class TierAOnly:
        def judge(self, **kwargs):
            return {
                "text": "被改写了",
                "base_model": "qwen",
                "edits": [
                    {
                        "span_asr": "原文",
                        "span_out": "被改写了",
                        "tier": "A",
                    }
                ],
                "overlap": False,
            }

    out, audits = run_pass_b(
        turns,
        draft,
        llm_judge=TierAOnly(),
        config=PipelineConfig(llm_max_retries=1),
    )
    assert out[0] == "原文"
    assert any(a.get("fallback") for a in audits)


def test_pass_b_moss_aware_rejects_non_moss_takeover_on_heavy_overlap():
    turns = [Turn(0, 2, "s0", "moss文本")]
    draft = {0: "moss文本"}

    class NonMossBase:
        def judge(self, **kwargs):
            return {
                "text": "qwen接管",
                "base_model": "qwen",
                "edits": [],
                "overlap": True,
            }

    out, audits = run_pass_b(
        turns,
        draft,
        llm_judge=NonMossBase(),
        overlap_turn_indices={0},
        heavy_overlap_turn_indices={0},
        moss_texts={0: "moss文本"},
        config=PipelineConfig(),
    )
    assert out[0] == "moss文本"
    assert any(a.get("path") == "moss_aware_reject" for a in audits)


def test_pass_b_light_overlap_does_not_moss_aware_reject():
    turns = [Turn(0, 2, "s0", "moss文本")]
    draft = {0: "moss文本"}

    class NonMossBase:
        def judge(self, **kwargs):
            return {
                "text": "qwen接管",
                "base_model": "qwen",
                "edits": [],
                "overlap": True,
            }

    out, audits = run_pass_b(
        turns,
        draft,
        llm_judge=NonMossBase(),
        overlap_turn_indices={0},
        moss_texts={0: "moss文本"},
        config=PipelineConfig(),
    )
    assert out[0] == "moss文本"
    assert not any(a.get("path") == "moss_aware_reject" for a in audits)


def test_pass_b_with_mock_llm_smoke():
    turns = [Turn(0, 2, "s0"), Turn(3, 5, "s1")]
    draft = {0: "帐号异常", 1: "单框架已经上线"}
    out, _ = run_pass_b(
        turns,
        draft,
        hotwords=["单框架|单方接"],
        llm_judge=MockLlmJudge(),
        config=PipelineConfig(),
    )
    assert "账号" in out[0] or out[0] == "帐号异常"


def test_pass_b_fallback_judge_after_primary_fails():
    turns = [Turn(0, 2, "s0", "系统奔至了")]
    draft = {0: "系统奔至了"}

    class AlwaysBad:
        name = "qwen"

        def judge(self, **kwargs):
            return {"text": "broken"}  # missing required fields

    class DeepSeekOk:
        name = "deepseek"

        def judge(self, **kwargs):
            text = kwargs["hypotheses"][0].text
            return {
                "text": text.replace("奔至", "蹦字"),
                "base_model": "draft",
                "edits": [
                    {
                        "span_asr": "奔至",
                        "span_out": "蹦字",
                        "tier": "C",
                        "pinyin_asr": "benzhi",
                        "pinyin_out": "bengzi",
                        "anchor": "meeting_draft",
                    }
                ],
                "overlap": False,
            }

    out, audits = run_pass_b(
        turns,
        draft,
        llm_judge=AlwaysBad(),
        fallback_judge=DeepSeekOk(),
        config=PipelineConfig(llm_max_retries=1),
    )
    assert "蹦字" in out[0]
    assert any(a.get("fallback_judge_ok") for a in audits)
