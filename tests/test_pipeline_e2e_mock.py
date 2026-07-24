from __future__ import annotations

import json
from pathlib import Path

from stage2_asr.pipeline import run_pipeline
from stage2_asr.prompt import SYSTEM_PROMPT, render_user_prompt
from stage2_asr.runners.base import UnsupportedRunnerError
from stage2_asr.runners.firered_asr2s import FireRedAsr2sConfig, FireRedAsr2sRunner
from stage2_asr.runners.llm_qwen36 import Qwen36LlmJudge
from stage2_asr.runners.mock_asr import MockAsrRunner
from stage2_asr.runners.mock_llm import MockLlmJudge
from stage2_asr.runners.qwen3_asr import Qwen3AsrRunner
from stage2_asr.types import AsrUnit, PipelineConfig


FIXTURES = Path(__file__).parent / "fixtures"


def test_pipeline_e2e_mock(tmp_path: Path):
    out = tmp_path / "work"
    result = run_pipeline(
        input_json=FIXTURES / "mode_c.json",
        audio_path=tmp_path / "missing.wav",
        work_dir=out,
        asr_runner=MockAsrRunner(),
        llm_judge=MockLlmJudge(),
        config=PipelineConfig(),
        hotwords=["单框架|单方接"],
    )
    assert result["n_turns"] >= 1
    assert (out / "asr_units.json").exists()
    assert (out / "asr_hypotheses.json").exists()
    assert (out / "mode_c_draft.json").exists()
    assert (out / "mode_c_asr_final.json").exists()
    assert (out / "llm_edits.jsonl").exists()

    final = json.loads((out / "mode_c_asr_final.json").read_text(encoding="utf-8"))
    texts = " ".join(t["text"] for t in final["turns"])
    assert "采用" in texts or "单框架" in texts or "大家好" in texts

    cache_files = list((out / "asr_cache").glob("*.json"))
    assert cache_files

    units = json.loads((out / "asr_units.json").read_text(encoding="utf-8"))["units"]
    assert any(u.get("heavy_overlap") for u in units)


def test_prompt_template_renders():
    assert "FIDELITY" in SYSTEM_PROMPT
    prompt = render_user_prompt(
        hypotheses_with_pinyin="moss: 你好",
        hotwords="[]",
        neighbor_draft="[]",
        overlap_flag=False,
        heavy_overlap_flag=True,
    )
    assert "heavy_overlap" in prompt.lower() or "HEAVY_OVERLAP=true" in prompt


def test_stubs_raise_without_weights():
    import pytest

    unit = AsrUnit("u", 0, 1, "s0", [0])
    with pytest.raises(UnsupportedRunnerError):
        Qwen3AsrRunner().transcribe_unit(unit, [], "x.wav")
    runner = FireRedAsr2sRunner()
    assert runner.config == FireRedAsr2sConfig(vad=False, lid=True, punc=True, asr=True)
    with pytest.raises(UnsupportedRunnerError):
        runner.transcribe_unit(unit, [], "x.wav")
    with pytest.raises(UnsupportedRunnerError):
        Qwen36LlmJudge().judge(
            hypotheses=[],
            neighbor_draft=[],
            hotwords=[],
            overlap=False,
            heavy_overlap=False,
            unit_id="u",
        )


def test_pass_a_retry_then_fallback():
    from stage2_asr.pass_a import run_pass_a_for_unit
    from stage2_asr.types import Hypothesis, Turn

    class BadThenGood:
        def __init__(self):
            self.n = 0

        def judge(self, **kwargs):
            self.n += 1
            if self.n == 1:
                return {"text": "only"}
            return {
                "text": "你好",
                "base_model": "moss",
                "edits": [],
                "overlap": False,
            }

    unit = AsrUnit("u0", 0, 1, "s0", [0], overlap_ratio=0.0)
    turns = [Turn(0, 1, "s0", "你好")]
    hyps = [Hypothesis("moss", "你好"), Hypothesis("qwen", "您好")]
    text, audit = run_pass_a_for_unit(
        unit=unit,
        turns=turns,
        hyps=hyps,
        draft_texts={0: "你好"},
        llm_judge=BadThenGood(),
        hotwords=[],
        config=PipelineConfig(llm_max_retries=2),
    )
    assert text == "你好"
    assert audit["retries"] >= 1
