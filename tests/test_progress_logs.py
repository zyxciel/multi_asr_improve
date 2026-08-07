from __future__ import annotations

from pathlib import Path

from stage2_asr.pipeline import run_pipeline
from stage2_asr.runners.mock_asr import MockAsrRunner
from stage2_asr.runners.mock_llm import MockLlmJudge
from stage2_asr.types import PipelineConfig


FIXTURES = Path(__file__).parent / "fixtures"


def test_pipeline_emits_pass_progress_on_stderr(tmp_path: Path, capsys):
    out = tmp_path / "work"
    run_pipeline(
        input_json=FIXTURES / "mode_c.json",
        audio_path=tmp_path / "missing.wav",
        work_dir=out,
        asr_runner=MockAsrRunner(),
        llm_judge=MockLlmJudge(),
        config=PipelineConfig(),
        hotwords=["单框架|单方接"],
        stage="all",
    )
    err = capsys.readouterr().err
    assert "[asr]" in err or "ASR" in err
    assert "[pass_a]" in err
    assert "[pass_b]" in err
