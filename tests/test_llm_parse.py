from __future__ import annotations

import json

import pytest

from stage2_asr.llm_parse import extract_json_and_reasoning, parse_judgment_json
from stage2_asr.runners.base import UnsupportedRunnerError
from stage2_asr.runners.llm_qwen36 import Qwen36LlmJudge
from stage2_asr.types import Hypothesis


def test_extract_strips_think_block():
    raw = (
        "<think>\n先比较几个候选，产用应为采用。\n</think>\n"
        '{"text":"采用","base_model":"qwen","edits":[],"overlap":false}'
    )
    parsed = extract_json_and_reasoning(raw)
    assert "采用" in parsed.reasoning
    payload = json.loads(parsed.json_text)
    assert payload["text"] == "采用"


def test_extract_handles_preamble_and_fence():
    raw = (
        "好的，下面是结果：\n```json\n"
        '{"text":"单框架","base_model":"draft","edits":[],"overlap":false}\n```'
    )
    payload, reasoning = parse_judgment_json(raw)
    assert payload["text"] == "单框架"
    assert reasoning is not None


def test_extract_raises_without_json():
    with pytest.raises(UnsupportedRunnerError):
        extract_json_and_reasoning("只有一堆分析没有大括号")


def test_judge_parses_thinking_wrapped_generate_fn():
    def gen(system, user):
        return (
            "<think>冗长推理……</think>\n"
            '{"text":"采用","base_model":"qwen","edits":[],"overlap":false}'
        )

    logs: list[dict] = []
    judge = Qwen36LlmJudge(
        enabled=True,
        generate_fn=gen,
        enable_thinking=False,
        log_fn=logs.append,
    )
    out = judge.judge(
        hypotheses=[Hypothesis("qwen", "产用")],
        neighbor_draft=[],
        hotwords=[],
        overlap=False,
        heavy_overlap=False,
        unit_id="u0",
    )
    assert out["text"] == "采用"
    assert any(e.get("pass") == "parse_reasoning" for e in logs)
