from __future__ import annotations

import numpy as np

from stage2_asr.types import PipelineConfig, Turn
from stage2_asr.units import build_asr_units, find_min_energy_split_time, overlap_ratio_for_span


def test_concat_same_speaker_within_gap():
    turns = [
        Turn(0.0, 1.0, "speaker_0", text="a"),
        Turn(2.0, 3.0, "speaker_0", text="b"),  # gap 1s <= 5
        Turn(10.0, 11.0, "speaker_0", text="c"),  # gap 7s > 5
    ]
    units = build_asr_units(turns)
    assert len(units) == 2
    assert units[0].turn_indices == [0, 1]
    assert units[0].moss_merged is True
    assert units[1].turn_indices == [2]


def test_no_concat_different_speakers():
    turns = [
        Turn(0.0, 1.0, "speaker_0"),
        Turn(1.1, 2.0, "speaker_1"),
    ]
    units = build_asr_units(turns)
    assert len(units) == 2


def test_short_skip_after_failed_merge():
    turns = [Turn(0.0, 0.2, "speaker_0", text="x")]
    units = build_asr_units(turns)
    assert len(units) == 1
    assert units[0].skip_asr is True
    assert units[0].skip_reason == "too_short"


def test_overlap_ratio_and_heavy_gate():
    turns = [
        Turn(0.0, 10.0, "speaker_0"),
        Turn(0.0, 4.0, "speaker_1"),  # 4/10 = 0.4 > 0.3
    ]
    ratio = overlap_ratio_for_span(0.0, 10.0, "speaker_0", turns)
    assert abs(ratio - 0.4) < 1e-6
    units = build_asr_units(turns)
    u0 = next(u for u in units if u.speaker_id == "speaker_0")
    assert u0.heavy_overlap is True
    assert u0.contains_overlap is True


def test_light_overlap_not_heavy():
    turns = [
        Turn(0.0, 10.0, "speaker_0"),
        Turn(0.0, 2.0, "speaker_1"),  # 0.2
    ]
    units = build_asr_units(turns)
    u0 = next(u for u in units if u.speaker_id == "speaker_0")
    assert u0.heavy_overlap is False
    assert u0.contains_overlap is True


def test_energy_split_prefers_quiet_middle():
    sr = 16000
    cfg = PipelineConfig(max_asr_seconds=1.0)
    # 2s signal: loud, quiet, loud
    t = np.linspace(0, 2.0, sr * 2, endpoint=False)
    audio = np.sin(2 * np.pi * 200 * t)
    audio[int(0.9 * sr) : int(1.1 * sr)] *= 0.01
    split = find_min_energy_split_time(audio, sr, 2.0, cfg)
    assert 0.7 < split < 1.3

    turns = [Turn(0.0, 2.0, "speaker_0", text="long")]
    units = build_asr_units(turns, cfg, audio=audio, sample_rate=sr)
    assert all(u.duration <= cfg.max_asr_seconds + 1e-6 for u in units)
    assert len(units) >= 2
