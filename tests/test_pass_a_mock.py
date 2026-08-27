from __future__ import annotations

from stage2_asr.pass_a import run_pass_a_for_unit
from stage2_asr.runners.mock_llm import MockLlmJudge
from stage2_asr.types import AsrUnit, Hypothesis, PipelineConfig, Turn


def test_pass_a_skips_when_agree_non_overlap():
    unit = AsrUnit("u", 0, 1, "s0", [0])
    turns = [Turn(0, 1, "s0", "你好")]
    hyps = [Hypothesis("moss", "你好"), Hypothesis("qwen", "你好")]
    text, audit = run_pass_a_for_unit(
        unit=unit,
        turns=turns,
        hyps=hyps,
        draft_texts={0: "你好"},
        llm_judge=MockLlmJudge(),
        hotwords=[],
        config=PipelineConfig(),
    )
    assert text == "你好"
    assert audit["skipped_llm"] is True


def test_pass_a_repairs_with_tier_c():
    unit = AsrUnit("u", 0, 2, "s0", [0])
    turns = [Turn(0, 2, "s0", "单框架产用")]
    hyps = [
        Hypothesis("moss", "单框架产用"),
        Hypothesis("firered", "单框架采用"),
    ]
    text, audit = run_pass_a_for_unit(
        unit=unit,
        turns=turns,
        hyps=hyps,
        draft_texts={0: "单框架产用"},
        llm_judge=MockLlmJudge(),
        hotwords=[],
        config=PipelineConfig(),
    )
    assert "采用" in text
    assert audit.get("skipped_llm") is False


def test_pass_a_deepseek_fallback_after_qwen_failures():
    unit = AsrUnit("u", 0, 1, "s0", [0])
    turns = [Turn(0, 1, "s0", "你好")]
    hyps = [Hypothesis("moss", "你好"), Hypothesis("qwen", "您好")]

    class AlwaysBad:
        name = "qwen36"

        def judge(self, **kwargs):
            return {"text": "only"}

    class DeepSeekOk:
        name = "deepseek"

        def judge(self, **kwargs):
            return {
                "text": "你好",
                "base_model": "moss",
                "edits": [],
                "overlap": False,
            }

    text, audit = run_pass_a_for_unit(
        unit=unit,
        turns=turns,
        hyps=hyps,
        draft_texts={0: "你好"},
        llm_judge=AlwaysBad(),
        fallback_judge=DeepSeekOk(),
        hotwords=[],
        config=PipelineConfig(llm_max_retries=1),
    )
    assert text == "你好"
    assert audit.get("fallback_judge") == "deepseek"
    assert audit.get("fallback_judge_ok") is True
    assert audit.get("fallback") is not True


def test_pass_a_light_overlap_keeps_qwen_judgment_without_forcing_moss():
    """0.5s / low-ratio overlap must not override a 2–1 中建 vote."""
    unit = AsrUnit(
        "u",
        0.0,
        5.0,
        "s0",
        [0],
        overlap_ratio=0.1,
        contains_overlap=True,
        heavy_overlap=False,
    )
    turns = [Turn(0.0, 5.0, "s0", "就是中介那边有吧")]
    hyps = [
        Hypothesis("moss", "就是中介那边有吧"),
        Hypothesis("qwen", "就是中建那边有啊"),
        Hypothesis("firered", "就是中建那边有吗"),
    ]

    class JudgeQwen:
        def judge(self, **kwargs):
            return {
                "text": "就是中建那边有啊",
                "base_model": "qwen",
                "edits": [],
                "overlap": True,
            }

    text, audit = run_pass_a_for_unit(
        unit=unit,
        turns=turns,
        hyps=hyps,
        draft_texts={0: "就是中介那边有吧"},
        llm_judge=JudgeQwen(),
        hotwords=[],
        config=PipelineConfig(),
    )
    assert text == "就是中建那边有啊"
    assert audit.get("forced_moss_base") is not True


def test_pass_a_heavy_overlap_still_forces_moss_without_edits():
    unit = AsrUnit(
        "u",
        0.0,
        5.0,
        "s0",
        [0],
        overlap_ratio=0.5,
        contains_overlap=True,
        heavy_overlap=True,
    )
    turns = [Turn(0.0, 5.0, "s0", "就是中介那边有吧")]
    hyps = [
        Hypothesis("moss", "就是中介那边有吧"),
        Hypothesis("qwen", "就是中建那边有啊"),
    ]

    class JudgeQwen:
        def judge(self, **kwargs):
            return {
                "text": "就是中建那边有啊",
                "base_model": "qwen",
                "edits": [],
                "overlap": True,
            }

    text, audit = run_pass_a_for_unit(
        unit=unit,
        turns=turns,
        hyps=hyps,
        draft_texts={0: "就是中介那边有吧"},
        llm_judge=JudgeQwen(),
        hotwords=[],
        config=PipelineConfig(),
    )
    assert text == "就是中介那边有吧"
    assert audit.get("forced_moss_base") is True
