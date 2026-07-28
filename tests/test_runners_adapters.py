from __future__ import annotations

from pathlib import Path

import numpy as np

from stage2_asr.audio_io import crop_unit_wav, make_silent_wav, write_wav_mono16k
from stage2_asr.eval_metrics import cer, corpus_cer, cp_cer
from stage2_asr.runners.base import UnsupportedRunnerError
from stage2_asr.runners.ensemble import EnsembleAsrRunner
from stage2_asr.runners.firered_asr2s import FireRedAsr2sConfig, FireRedAsr2sRunner
from stage2_asr.runners.llm_deepseek import DeepSeekLlmJudge
from stage2_asr.runners.llm_qwen36 import Qwen36LlmJudge
from stage2_asr.runners.qwen3_asr import Qwen3AsrRunner
from stage2_asr.types import AsrStatus, AsrUnit, Turn


class _FakeQwenModel:
    def transcribe(self, audio=None, language=None):
        return [type("R", (), {"text": "你好世界"})()]


class _FakeFireRedSystem:
    def process(self, wav_path, uttid="tmp"):
        return {
            "sentences": [
                {"text": "你好，世界。", "lang": "zh", "lang_confidence": 0.99}
            ]
        }


def test_audio_crop_roundtrip(tmp_path: Path):
    sr = 16000
    wav = tmp_path / "full.wav"
    # 2 seconds of noise
    audio = (np.sin(2 * np.pi * 440 * np.linspace(0, 2, sr * 2, endpoint=False))).astype(np.float32) * 0.2
    write_wav_mono16k(wav, audio, sr=sr)
    crop = crop_unit_wav(wav, 0.5, 1.5, work_dir=tmp_path, unit_id="u0", sr=sr)
    assert crop.exists()
    assert crop.stat().st_size > 44


def test_eval_metrics():
    assert cer("你好", "你好") == 0.0
    assert cer("你好", "您好") > 0
    assert 0.0 <= cp_cer("账号", "帐号") < 1.0
    assert corpus_cer([("你好", "你好"), ("世界", "世界")]) == 0.0


def test_qwen_runner_with_injected_model():
    runner = Qwen3AsrRunner(model=_FakeQwenModel())
    unit = AsrUnit("u", 0, 1, "s0", [0])
    hyps = runner.transcribe_unit(unit, [], "x.wav")
    assert hyps[0].model == "qwen"
    assert hyps[0].text == "你好世界"


def test_qwen_runner_disabled_raises():
    import pytest

    with pytest.raises(UnsupportedRunnerError):
        Qwen3AsrRunner().transcribe_unit(AsrUnit("u", 0, 1, "s0", [0]), [], "x.wav")


def test_firered_runner_vad_off_lid_punc_with_fake_system():
    cfg = FireRedAsr2sConfig(vad=False, lid=True, punc=True)
    runner = FireRedAsr2sRunner(system=_FakeFireRedSystem(), config=cfg)
    assert runner.config.vad is False
    hyps = runner.transcribe_unit(AsrUnit("u", 0, 1, "s0", [0]), [], "x.wav")
    assert hyps[0].model == "firered"
    assert hyps[0].lid == "zh"
    assert "世界" in hyps[0].text
    assert hyps[0].meta["vad"] is False
    assert hyps[0].meta["punc"] is True


def test_ensemble_moss_exclusive():
    ens = EnsembleAsrRunner(Qwen3AsrRunner(model=_FakeQwenModel()), FireRedAsr2sRunner(system=_FakeFireRedSystem()))
    turns = [Turn(0, 1, "s0", "来自MOSS", asr_status=AsrStatus.PROVISIONAL)]
    unit = AsrUnit("u", 0, 1, "s0", [0], heavy_overlap=True)
    hyps = ens.transcribe_unit(unit, turns, "x.wav", moss_exclusive=True)
    assert [h.model for h in hyps] == ["moss"]


def test_ensemble_all_models():
    ens = EnsembleAsrRunner(Qwen3AsrRunner(model=_FakeQwenModel()), FireRedAsr2sRunner(system=_FakeFireRedSystem()))
    turns = [Turn(0, 1, "s0", "来自MOSS", asr_status=AsrStatus.PROVISIONAL)]
    unit = AsrUnit("u", 0, 1, "s0", [0])
    hyps = ens.transcribe_unit(unit, turns, "x.wav", moss_exclusive=False)
    assert {h.model for h in hyps} == {"moss", "qwen", "firered"}


def test_ensemble_selected_models():
    ens = EnsembleAsrRunner(Qwen3AsrRunner(model=_FakeQwenModel()), FireRedAsr2sRunner(system=_FakeFireRedSystem()))
    turns = [Turn(0, 1, "s0", "来自MOSS", asr_status=AsrStatus.PROVISIONAL)]
    unit = AsrUnit("u", 0, 1, "s0", [0])
    hyps = ens.transcribe_unit(
        unit, turns, "x.wav", moss_exclusive=False, selected_models={"moss", "firered"}
    )
    assert {h.model for h in hyps} == {"moss", "firered"}


def test_llm_judge_with_generate_fn():
    payload = {
        "text": "你好",
        "base_model": "moss",
        "edits": [],
        "overlap": False,
    }

    def gen(system, user):
        assert "FIDELITY" in system or "phonetic" in system.lower() or "JSON" in system
        assert "产用" in user
        return __import__("json").dumps(payload, ensure_ascii=False)

    judge = Qwen36LlmJudge(generate_fn=gen)
    out = judge.judge(
        hypotheses=[],
        neighbor_draft=[],
        hotwords=[],
        overlap=False,
        heavy_overlap=False,
        unit_id="u",
    )
    assert out["text"] == "你好"


def test_deepseek_judge_disabled_raises():
    import pytest

    with pytest.raises(UnsupportedRunnerError):
        DeepSeekLlmJudge().judge(
            hypotheses=[],
            neighbor_draft=[],
            hotwords=[],
            overlap=False,
            heavy_overlap=False,
            unit_id="u",
        )


def test_deepseek_judge_with_generate_fn():
    payload = {
        "text": "你好",
        "base_model": "moss",
        "edits": [],
        "overlap": False,
    }

    def gen(system, user):
        assert "Few-shots" in user or "产用" in user
        return __import__("json").dumps(payload, ensure_ascii=False)

    judge = DeepSeekLlmJudge(generate_fn=gen)
    out = judge.judge(
        hypotheses=[],
        neighbor_draft=[],
        hotwords=[],
        overlap=False,
        heavy_overlap=False,
        unit_id="u",
    )
    assert out["text"] == "你好"
    assert judge.name == "deepseek"
