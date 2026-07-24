from __future__ import annotations

from stage2_asr.types import PipelineConfig, Turn
from stage2_asr.units import build_asr_units, overlap_ratio_for_span


def test_overlap_ratio_formula():
    turns = [Turn(0, 10, "s0"), Turn(2, 5, "s1")]
    assert abs(overlap_ratio_for_span(0, 10, "s0", turns) - 0.3) < 1e-9


def test_dynamic_gate_threshold():
    heavy = [Turn(0, 10, "s0"), Turn(0, 4, "s1")]
    light = [Turn(0, 10, "s0"), Turn(0, 2, "s1")]
    uh = next(u for u in build_asr_units(heavy) if u.speaker_id == "s0")
    ul = next(u for u in build_asr_units(light) if u.speaker_id == "s0")
    assert uh.heavy_overlap is True
    assert ul.heavy_overlap is False
    assert ul.contains_overlap is True
