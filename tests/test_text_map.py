from __future__ import annotations

from stage2_asr.text_map import distribute_unit_text, join_turn_texts
from stage2_asr.types import Turn


def test_join_turn_texts_uses_joiner_when_no_ending_punct():
    assert join_turn_texts(["嗯，你们", "欢", "嗯，主"]) == "嗯，你们。欢。嗯，主"


def test_join_turn_texts_does_not_double_period():
    assert join_turn_texts(["嗯，你们。", "欢。", "嗯，主"]) == "嗯，你们。欢。嗯，主"


def test_distribute_splits_on_joiner_not_duration():
    """Duration slicing would leak '。' into the middle of a turn (欢。。嗯，主)."""
    turns = [
        Turn(0.0, 1.17, "s3", "嗯，你们"),
        Turn(2.0, 3.17, "s3", "欢"),
        Turn(4.0, 5.25, "s3", "嗯，主"),
        Turn(6.0, 7.25, "s3", "场"),
        Turn(8.0, 9.0, "s3", "诶"),
    ]
    text = join_turn_texts([t.text for t in turns])
    # Unequal durations would mis-cut if we sliced by character count.
    out = distribute_unit_text(list(range(5)), text, turns)
    assert out[0] == "嗯，你们"
    assert out[1] == "欢"
    assert out[2] == "嗯，主"
    assert out[3] == "场"
    assert out[4] == "诶"
    assert "。。" not in "".join(out.values())


def test_distribute_repairs_leaked_double_joiner():
    turns = [
        Turn(0.0, 1.0, "s0", "欢"),
        Turn(1.0, 2.0, "s0", "嗯，主"),
    ]
    out = distribute_unit_text([0, 1], "欢。。嗯，主", turns)
    assert out[0] == "欢"
    assert out[1] == "嗯，主"


def test_distribute_duration_fallback_strips_boundary_joiners():
    turns = [
        Turn(0.0, 1.0, "s0"),
        Turn(1.0, 1.1, "s0"),  # much shorter → duration split cannot match 2 sentences
    ]
    text = "abcdefghij"
    out = distribute_unit_text([0, 1], text, turns)
    assert out[0] + out[1] == text
    assert not out[0].startswith("。")
    assert not out[1].endswith("。") or text.endswith("。")
