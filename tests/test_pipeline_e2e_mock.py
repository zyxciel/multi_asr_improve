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
    assert (out / "pass_stats.json").exists()

    final = json.loads((out / "mode_c_asr_final.json").read_text(encoding="utf-8"))
    texts = " ".join(t["text"] for t in final["turns"])
    assert "采用" in texts or "单框架" in texts or "大家好" in texts

    stats = json.loads((out / "pass_stats.json").read_text(encoding="utf-8"))
    assert "pass_a" in stats and "pass_b" in stats

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
    assert "产用" in prompt
    assert "单方接" in prompt
    assert "奔至" in prompt
    assert "Few-shots" in prompt


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


def _read_hyp_records(path: Path) -> list[dict]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(payload, list):
        return payload
    return payload["records"]


def test_stage_asr_then_llm_uses_cached_hypotheses(tmp_path: Path):
    out = tmp_path / "work"
    # Stage 1: ASR only
    asr_result = run_pipeline(
        input_json=FIXTURES / "mode_c.json",
        audio_path=tmp_path / "missing.wav",
        work_dir=out,
        asr_runner=MockAsrRunner(),
        llm_judge=MockLlmJudge(),
        config=PipelineConfig(),
        hotwords=["单框架|单方接"],
        stage="asr",
        asr_models=["moss", "qwen"],
    )
    assert asr_result["stage"] == "asr"
    assert (out / "asr_hypotheses.json").exists()
    assert not (out / "mode_c_draft.json").exists()

    # Stage 2: LLM only from cached ASR results
    llm_result = run_pipeline(
        input_json=FIXTURES / "mode_c.json",
        audio_path=tmp_path / "missing.wav",
        work_dir=out,
        asr_runner=MockAsrRunner(),  # should not be used in llm stage
        llm_judge=MockLlmJudge(),
        config=PipelineConfig(),
        hotwords=["单框架|单方接"],
        stage="llm",
    )
    assert llm_result["stage"] == "llm"
    assert (out / "mode_c_draft.json").exists()
    assert (out / "mode_c_asr_final.json").exists()


def test_stage_asr_can_accumulate_different_models(tmp_path: Path):
    out = tmp_path / "work"
    run_pipeline(
        input_json=FIXTURES / "mode_c.json",
        audio_path=tmp_path / "missing.wav",
        work_dir=out,
        asr_runner=MockAsrRunner(),
        llm_judge=MockLlmJudge(),
        config=PipelineConfig(),
        stage="asr",
        asr_models=["qwen"],
    )
    run_pipeline(
        input_json=FIXTURES / "mode_c.json",
        audio_path=tmp_path / "missing.wav",
        work_dir=out,
        asr_runner=MockAsrRunner(),
        llm_judge=MockLlmJudge(),
        config=PipelineConfig(),
        stage="asr",
        asr_models=["firered"],
    )
    records = _read_hyp_records(out / "asr_hypotheses.json")
    non_skipped = [r for r in records if not r.get("skipped")]
    assert non_skipped
    models = {h["model"] for r in non_skipped for h in r.get("hyps", [])}
    assert "qwen" in models
    assert "firered" in models


def test_stage_asr_moss_merges_into_existing_qwen_firered(tmp_path: Path):
    """Moss must be addable after qwen/firered, even if a prior empty moss cache exists."""
    out = tmp_path / "work"
    run_pipeline(
        input_json=FIXTURES / "mode_c.json",
        audio_path=tmp_path / "missing.wav",
        work_dir=out,
        asr_runner=MockAsrRunner(),
        llm_judge=MockLlmJudge(),
        config=PipelineConfig(),
        stage="asr",
        asr_models=["qwen", "firered"],
    )
    before = {
        r["unit_id"]: {h["model"] for h in (r.get("hyps") or [])}
        for r in _read_hyp_records(out / "asr_hypotheses.json")
        if not r.get("skipped")
    }
    cache_dir = out / "asr_cache"
    cache_dir.mkdir(parents=True, exist_ok=True)
    units = json.loads((out / "asr_units.json").read_text(encoding="utf-8"))["units"]
    for u in units:
        (cache_dir / f"{u['unit_id']}__mock_asr__moss.json").write_text("[]", encoding="utf-8")

    run_pipeline(
        input_json=FIXTURES / "mode_c.json",
        audio_path=tmp_path / "missing.wav",
        work_dir=out,
        asr_runner=MockAsrRunner(),
        llm_judge=MockLlmJudge(),
        config=PipelineConfig(),
        stage="asr",
        asr_models=["moss"],
    )
    records = _read_hyp_records(out / "asr_hypotheses.json")
    models = {
        r["unit_id"]: {h["model"] for h in (r.get("hyps") or [])}
        for r in records
        if not r.get("skipped")
    }
    assert models
    for uid, ms in models.items():
        assert "moss" in ms, uid
        if "qwen" in before.get(uid, set()) or "firered" in before.get(uid, set()):
            assert "qwen" in ms and "firered" in ms, uid


def test_stage_all_merges_prior_hyps(tmp_path: Path):
    out = tmp_path / "work"
    run_pipeline(
        input_json=FIXTURES / "mode_c.json",
        audio_path=tmp_path / "missing.wav",
        work_dir=out,
        asr_runner=MockAsrRunner(),
        llm_judge=MockLlmJudge(),
        config=PipelineConfig(),
        stage="asr",
        asr_models=["qwen"],
    )
    run_pipeline(
        input_json=FIXTURES / "mode_c.json",
        audio_path=tmp_path / "missing.wav",
        work_dir=out,
        asr_runner=MockAsrRunner(),
        llm_judge=MockLlmJudge(),
        config=PipelineConfig(),
        stage="all",
        asr_models=["moss"],
        hotwords=["单框架|单方接"],
    )
    records = _read_hyp_records(out / "asr_hypotheses.json")
    models = {h["model"] for r in records if not r.get("skipped") for h in r.get("hyps", [])}
    assert "qwen" in models
    assert "moss" in models


def test_llm_stage_reloads_persisted_units(tmp_path: Path):
    out = tmp_path / "work"
    run_pipeline(
        input_json=FIXTURES / "mode_c.json",
        audio_path=tmp_path / "missing.wav",
        work_dir=out,
        asr_runner=MockAsrRunner(),
        llm_judge=MockLlmJudge(),
        config=PipelineConfig(),
        stage="asr",
        asr_models=["moss", "qwen"],
    )
    units_before = json.loads((out / "asr_units.json").read_text(encoding="utf-8"))["units"]
    # Corrupt unit_ids would break Pass A if rebuilt differently; pin by reloading.
    run_pipeline(
        input_json=FIXTURES / "mode_c.json",
        audio_path=tmp_path / "missing.wav",
        work_dir=out,
        asr_runner=MockAsrRunner(),
        llm_judge=MockLlmJudge(),
        config=PipelineConfig(),
        stage="llm",
        hotwords=["单框架|单方接"],
    )
    units_after = json.loads((out / "asr_units.json").read_text(encoding="utf-8"))["units"]
    assert [u["unit_id"] for u in units_before] == [u["unit_id"] for u in units_after]
    assert (out / "mode_c_asr_final.json").exists()


def test_pass_b_stage_preserves_pass_a_artifacts(tmp_path: Path):
    out = tmp_path / "work"
    run_pipeline(
        input_json=FIXTURES / "mode_c.json",
        audio_path=tmp_path / "missing.wav",
        work_dir=out,
        asr_runner=MockAsrRunner(),
        llm_judge=MockLlmJudge(),
        config=PipelineConfig(),
        stage="asr",
        asr_models=["moss", "qwen"],
    )
    run_pipeline(
        input_json=FIXTURES / "mode_c.json",
        audio_path=tmp_path / "missing.wav",
        work_dir=out,
        asr_runner=MockAsrRunner(),
        llm_judge=MockLlmJudge(),
        config=PipelineConfig(),
        stage="pass_a",
        hotwords=["单框架|单方接"],
    )
    edits_a = (out / "llm_edits.jsonl").read_text(encoding="utf-8")
    assert '"pass": "A"' in edits_a or '"pass":"A"' in edits_a
    stats_a = json.loads((out / "pass_stats.json").read_text(encoding="utf-8"))
    assert "pass_a" in stats_a

    run_pipeline(
        input_json=FIXTURES / "mode_c.json",
        audio_path=tmp_path / "missing.wav",
        work_dir=out,
        asr_runner=MockAsrRunner(),
        llm_judge=MockLlmJudge(),
        config=PipelineConfig(),
        stage="pass_b",
        hotwords=["单框架|单方接"],
    )
    edits = [
        json.loads(line)
        for line in (out / "llm_edits.jsonl").read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    assert any(e.get("pass") == "A" for e in edits)
    stats = json.loads((out / "pass_stats.json").read_text(encoding="utf-8"))
    assert "pass_a" in stats and "pass_b" in stats
