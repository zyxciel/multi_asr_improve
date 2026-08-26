from __future__ import annotations

from stage2_asr.text_map import (
    distribute_unit_text,
    join_turn_texts,
    merge_consecutive_turns,
)
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


def test_merge_consecutive_same_speaker_keeps_original_endpoints():
    turns = [
        Turn(0.0, 1.0, "s0", "以前那个温"),
        Turn(1.1, 2.0, "s0", "度的问题"),
        Turn(8.0, 9.0, "s0", "好的"),  # gap 6 > 5
    ]
    merged, members = merge_consecutive_turns(turns, max_duration=30.0, max_merge_gap=5.0)
    assert [(t.start, t.end, t.speaker_id, t.text) for t in merged] == [
        (0.0, 2.0, "s0", "以前那个温度的问题"),
        (8.0, 9.0, "s0", "好的"),
    ]
    assert members == [[0, 1], [2]]


def test_merge_does_not_join_different_speakers_or_non_adjacent_rows():
    turns = [
        Turn(0.0, 1.0, "s0", "甲"),
        Turn(1.1, 2.0, "s1", "乙"),
        Turn(2.1, 3.0, "s0", "丙"),
    ]
    merged, members = merge_consecutive_turns(turns, max_duration=30.0, max_merge_gap=5.0)
    assert len(merged) == 3
    assert [t.text for t in merged] == ["甲", "乙", "丙"]
    assert [t.start for t in merged] == [0.0, 1.1, 2.1]
    assert [t.end for t in merged] == [1.0, 2.0, 3.0]
    assert members == [[0], [1], [2]]


def test_merge_keeps_overlapping_speakers_and_their_timestamps():
    turns = [
        Turn(0.0, 5.0, "s0", "以前那个Windows产品"),
        Turn(2.0, 6.0, "s1", "以前那个Windows产品"),
    ]
    merged, members = merge_consecutive_turns(turns, max_duration=30.0, max_merge_gap=5.0)
    assert len(merged) == 2
    assert merged[0].start == 0.0 and merged[0].end == 5.0
    assert merged[1].start == 2.0 and merged[1].end == 6.0
    assert members == [[0], [1]]


def test_merge_splits_overlong_turn_into_equal_chunks():
    turns = [Turn(0.0, 70.0, "s0", "很长的一段话")]
    merged, members = merge_consecutive_turns(turns, max_duration=30.0, max_merge_gap=5.0)
    assert [(round(t.start, 6), round(t.end, 6), t.text) for t in merged] == [
        (0.0, 30.0, "很长的一段话"),
        (30.0, 60.0, ""),
        (60.0, 70.0, ""),
    ]
    assert members == [[0], [0], [0]]

