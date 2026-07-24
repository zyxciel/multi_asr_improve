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
