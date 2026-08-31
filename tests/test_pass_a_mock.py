from __future__ import annotations

from stage2_asr.pass_a import run_pass_a_batch, run_pass_a_for_unit
from stage2_asr.pinyin_util import to_pinyin
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


def test_pass_a_empty_edits_hallucination_falls_back_to_base():
    unit = AsrUnit("u", 0, 1, "s0", [0])
    turns = [Turn(0, 1, "s0", "你好世界")]
    hyps = [Hypothesis("moss", "你好世界"), Hypothesis("qwen", "您好世界")]

    class Hallucinate:
        def judge(self, **kwargs):
            return {
                "text": "完全不相干的句子这是幻觉",
                "base_model": "moss",
                "edits": [],
                "overlap": False,
            }

    text, audit = run_pass_a_for_unit(
        unit=unit,
        turns=turns,
        hyps=hyps,
        draft_texts={0: "你好世界"},
        llm_judge=Hallucinate(),
        hotwords=[],
        config=PipelineConfig(),
    )
    assert text == "你好世界"
    assert audit.get("empty_edits_reject") is True


def test_pass_a_empty_edits_may_select_another_hyp():
    unit = AsrUnit("u", 0, 1, "s0", [0])
    turns = [Turn(0, 1, "s0", "你好世界")]
    hyps = [Hypothesis("moss", "你好世界"), Hypothesis("qwen", "您好世界")]

    class SelectQwen:
        def judge(self, **kwargs):
            return {
                "text": "您好世界",
                "base_model": "qwen",
                "edits": [],
                "overlap": False,
            }

    text, audit = run_pass_a_for_unit(
        unit=unit,
        turns=turns,
        hyps=hyps,
        draft_texts={0: "你好世界"},
        llm_judge=SelectQwen(),
        hotwords=[],
        config=PipelineConfig(),
    )
    assert text == "您好世界"
    assert audit.get("empty_edits_reject") is not True


def test_pass_a_replays_edits_onto_base_not_judgment_text():
    unit = AsrUnit("u", 0, 2, "s0", [0])
    turns = [Turn(0, 2, "s0", "单框架产用")]
    hyps = [
        Hypothesis("moss", "单框架产用"),
        Hypothesis("qwen", "单框架采用"),
    ]

    class MismatchedText:
        def judge(self, **kwargs):
            return {
                "text": "完全幻觉全文",
                "base_model": "moss",
                "edits": [
                    {
                        "span_asr": "产用",
                        "span_out": "采用",
                        "tier": "C",
                        "pinyin_asr": to_pinyin("产用"),
                        "pinyin_out": to_pinyin("采用"),
                        "anchor": "hyp",
                    }
                ],
                "overlap": False,
            }

    text, audit = run_pass_a_for_unit(
        unit=unit,
        turns=turns,
        hyps=hyps,
        draft_texts={0: "单框架产用"},
        llm_judge=MismatchedText(),
        hotwords=[],
        config=PipelineConfig(),
    )
    assert text == "单框架采用"
    assert "幻觉" not in text


def test_pass_a_concatenates_split_slices_of_same_turn():
    turns = [Turn(0.0, 40.0, "s0", "原始整句。")]
    items = [
        {
            "unit": AsrUnit("unit_0000", 0.0, 20.0, "s0", [0]),
            "hyps": [
                Hypothesis("moss", "前一半内容。"),
                Hypothesis("qwen", "前一半内容。"),
            ],
        },
        {
            "unit": AsrUnit("unit_0001", 20.0, 40.0, "s0", [0]),
            "hyps": [
                Hypothesis("moss", "后一半内容。"),
                Hypothesis("qwen", "后一半内容。"),
            ],
        },
    ]
    draft = {0: "原始整句。"}
    run_pass_a_batch(
        items=items,
        turns=turns,
        draft_texts=draft,
        llm_judge=MockLlmJudge(),
        hotwords=[],
        config=PipelineConfig(),
    )
    assert "前一半内容" in draft[0]
    assert "后一半内容" in draft[0]


def test_pass_a_does_not_double_full_moss_text_on_split_slices():
    # In production the moss hyp is the FULL Mode-C turn text on every slice of a
    # split long turn (Mode-C text is not split). Concat must not double it.
    full = "第一句内容。第二句内容。第三句内容。第四句内容。"
    turns = [Turn(0.0, 40.0, "s0", full)]

    class EchoMoss(MockLlmJudge):
        def judge(self, *, hypotheses, neighbor_draft, hotwords, overlap, heavy_overlap, unit_id):
            h = hypotheses[0]  # moss is first
            return {"text": h.text, "base_model": h.model, "edits": [], "overlap": False}

    items = [
        {
            "unit": AsrUnit("unit_0000", 0.0, 20.0, "s0", [0]),
            "hyps": [Hypothesis("moss", full), Hypothesis("qwen", "前一半内容。")],
        },
        {
            "unit": AsrUnit("unit_0001", 20.0, 40.0, "s0", [0]),
            "hyps": [Hypothesis("moss", full), Hypothesis("qwen", "后一半内容。")],
        },
    ]
    draft = {0: full}
    run_pass_a_batch(
        items=items,
        turns=turns,
        draft_texts=draft,
        llm_judge=EchoMoss(),
        hotwords=[],
        config=PipelineConfig(),
    )
    assert draft[0] == full
