from __future__ import annotations

from stage2_asr.neighbors import cap_neighbors, meeting_draft
from stage2_asr.types import PipelineConfig, Turn


def test_cap_neighbors_prefers_nearby_turns_not_meeting_prefix():
    turns = [
        Turn(0.0, 10.0, "s0", "开会开场白很长" * 20),
        Turn(20.0, 30.0, "s1", "一夫多妻制"),
        Turn(31.0, 40.0, "s0", "求婚啊"),
        Turn(200.0, 210.0, "s1", "很远的收尾"),
    ]
    texts = {i: t.text for i, t in enumerate(turns)}
    meeting = meeting_draft(turns, texts)
    cfg = PipelineConfig(neighbor_max_turns=1, neighbor_window_seconds=600.0)
    neighbors = cap_neighbors(meeting, 2, cfg)
    assert [row["text"] for row in neighbors] == ["一夫多妻制"]


def test_cap_neighbors_char_budget_keeps_nearby_keyword_over_long_prefix():
    opening = "开场" * 5000
    turns = [
        Turn(0.0, 10.0, "s0", opening),
        Turn(90.0, 99.0, "s1", "一夫多妻制"),
        Turn(100.0, 110.0, "s0", "求婚啊"),
    ]
    meeting = meeting_draft(turns, {i: t.text for i, t in enumerate(turns)})
    cfg = PipelineConfig(neighbor_max_turns=20, neighbor_window_seconds=600.0)
    neighbors = cap_neighbors(meeting, 2, cfg)
    texts = [row["text"] for row in neighbors]
    assert "一夫多妻制" in texts
    assert opening not in texts


def test_cap_neighbors_respects_time_window():
    turns = [
        Turn(0.0, 1.0, "s0", "远"),
        Turn(500.0, 501.0, "s0", "近"),
        Turn(502.0, 503.0, "s0", "当前"),
    ]
    meeting = meeting_draft(turns, {i: t.text for i, t in enumerate(turns)})
    cfg = PipelineConfig(neighbor_max_turns=20, neighbor_window_seconds=60.0)
    neighbors = cap_neighbors(meeting, 2, cfg)
    assert [row["text"] for row in neighbors] == ["近"]
