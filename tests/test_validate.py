from __future__ import annotations

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
