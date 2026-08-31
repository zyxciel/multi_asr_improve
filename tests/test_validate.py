from __future__ import annotations

import json
from pathlib import Path

from stage2_asr.pipeline import load_mode_c
from stage2_asr.types import Turn
from stage2_asr.validate import validate_turns


def test_skips_nan_and_too_short():
    turns = [
        Turn(0.0, 1.0, "speaker_0", text="ok"),
        Turn(float("nan"), 1.0, "speaker_0", text="bad"),
        Turn(1.0, 1.005, "speaker_1", text="tiny"),  # < 0.01
        Turn(2.0, 1.5, "speaker_1", text="inverted"),
        Turn(3.0, float("inf"), "speaker_2", text="inf"),
    ]
    kept, skipped = validate_turns(turns)
    assert len(kept) == 1
    assert kept[0].text == "ok"
    assert len(skipped) == 4
    assert all(s["tag"] == "skipped_invalid_ts" for s in skipped)


def test_keeps_normal_turns():
    turns = [Turn(0.0, 0.5, "s0"), Turn(0.5, 1.2, "s1")]
    kept, skipped = validate_turns(turns)
    assert len(kept) == 2
    assert skipped == []


def test_from_dict_null_and_non_numeric_timestamps_become_nan():
    import math

    null_start = Turn.from_dict({"start": None, "end": 1.0, "speaker_id": "s0", "text": "x"})
    bad_start = Turn.from_dict({"start": "abc", "end": 1.0, "speaker_id": "s0", "text": "y"})
    assert math.isnan(null_start.start)
    assert math.isnan(bad_start.start)


def test_from_dict_missing_speaker_id_does_not_crash():
    t = Turn.from_dict({"start": 0.0, "end": 1.0, "text": "ok"})
    assert t.text == "ok"
    assert t.speaker_id == "?"


def test_load_mode_c_null_start_is_skipped_not_crash(tmp_path: Path):
    path = tmp_path / "mode_c.json"
    path.write_text(
        json.dumps(
            {
                "turns": [
                    {"start": None, "end": 1.0, "speaker_id": "s0", "text": "bad"},
                    {"start": 1.0, "end": 2.0, "speaker_id": "s1", "text": "ok"},
                    {"start": "abc", "end": 3.0, "speaker_id": "s2", "text": "also-bad"},
                ]
            },
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    raw, _ = load_mode_c(path)
    kept, skipped = validate_turns(raw)
    assert [t.text for t in kept] == ["ok"]
    assert len(skipped) == 2
    assert all(s["tag"] == "skipped_invalid_ts" for s in skipped)
