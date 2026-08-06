from __future__ import annotations

import json
from pathlib import Path

from stage2_asr.pipeline import load_mode_c, run_pipeline
from stage2_asr.runners.mock_asr import MockAsrRunner
from stage2_asr.runners.mock_llm import MockLlmJudge
from stage2_asr.types import PipelineConfig


def test_load_mode_c_plain_turn_array(tmp_path: Path):
    path = tmp_path / "mode_c.json"
    path.write_text(
        json.dumps(
            [
                {"start": 0.37, "end": 0.85, "text": "yes", "speaker_id": "speaker_0"},
                {"start": 1.0, "end": 2.0, "text": "hello", "speaker_id": "speaker_1"},
            ],
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    turns, doc = load_mode_c(path)
    assert len(turns) == 2
    assert turns[0].text == "yes"
    assert turns[0].speaker_id == "speaker_0"
    assert doc["turns"] == json.loads(path.read_text(encoding="utf-8"))


def test_pipeline_accepts_plain_turn_array(tmp_path: Path):
    mode_c = tmp_path / "mode_c.json"
    mode_c.write_text(
        json.dumps(
            [
                {
                    "start": 0.0,
                    "end": 2.0,
                    "speaker_id": "speaker_0",
                    "text": "另外一个因为这些那个模型呢它是需要兼容车机手机还有单框架产用他们用的。",
                },
                {"start": 3.0, "end": 5.0, "speaker_id": "speaker_0", "text": "大家好"},
            ],
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    out = tmp_path / "work"
    result = run_pipeline(
        input_json=mode_c,
        audio_path=tmp_path / "missing.wav",
        work_dir=out,
        asr_runner=MockAsrRunner(),
        llm_judge=MockLlmJudge(),
        config=PipelineConfig(),
        hotwords=["单框架|单方接"],
    )
    assert result["n_turns"] == 2
    assert (out / "mode_c_asr_final.json").exists()
