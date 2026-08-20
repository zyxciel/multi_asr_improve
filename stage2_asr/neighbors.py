from __future__ import annotations

from stage2_asr.types import PipelineConfig, Turn


def meeting_draft(turns: list[Turn], texts: dict[int, str]) -> list[dict]:
    return [
        {
            "turn_index": i,
            "start": t.start,
            "end": t.end,
            "speaker_id": t.speaker_id,
            "text": texts.get(i, t.text),
        }
        for i, t in enumerate(turns)
    ]


def cap_neighbors(
    meeting: list[dict],
    turn_index: int,
    cfg: PipelineConfig,
) -> list[dict]:
    """Cap meeting neighbors (~4096 tokens ≈ 8192 chars, neighbor_max_turns)."""
    neighbors = [row for row in meeting if row["turn_index"] != turn_index]
    char_budget = 4096 * 2
    used = 0
    capped: list[dict] = []
    for row in neighbors:
        cost = len(str(row.get("text", ""))) + 32
        if capped and used + cost > char_budget:
            break
        capped.append(row)
        used += cost
        if len(capped) >= cfg.neighbor_max_turns:
            break
    return capped
