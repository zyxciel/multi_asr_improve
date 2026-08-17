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
    assert turns[0].asr_status.value == "provisional"
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


def test_moss_hyp_from_mode_c_text_without_asr_status(tmp_path: Path):
    """Diarizen Mode-C often has text but no asr_status (defaults were empty → moss dropped)."""
    mode_c = tmp_path / "mode_c.json"
    mode_c.write_text(
        json.dumps(
            [
                {"start": 0.0, "end": 2.0, "speaker_id": "speaker_0", "text": "你们有过娃娃亲吗？"},
                {"start": 3.0, "end": 5.0, "speaker_id": "speaker_1", "text": "没有"},
            ],
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    out = tmp_path / "work"
    run_pipeline(
        input_json=mode_c,
        audio_path=tmp_path / "missing.wav",
        work_dir=out,
        asr_runner=MockAsrRunner(),
        llm_judge=MockLlmJudge(),
        config=PipelineConfig(),
        stage="asr",
        asr_models=["moss"],
    )
    payload = json.loads((out / "asr_hypotheses.json").read_text(encoding="utf-8"))
    records = payload["records"] if isinstance(payload, dict) else payload
    moss_texts = [
        h["text"]
        for r in records
        if not r.get("skipped")
        for h in (r.get("hyps") or [])
        if h.get("model") == "moss"
    ]
    assert any("娃娃亲" in t for t in moss_texts)
