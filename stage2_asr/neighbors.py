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
    """Nearest other turns within ±window, capped by neighbor_max_turns and ~8k chars."""
    current = next((row for row in meeting if int(row["turn_index"]) == int(turn_index)), None)
    if current is None:
        return []
    center = 0.5 * (float(current["start"]) + float(current["end"]))
    window = float(cfg.neighbor_window_seconds)
    cands: list[tuple[float, int, dict]] = []
    for row in meeting:
        idx = int(row["turn_index"])
        if idx == int(turn_index):
            continue
        mid = 0.5 * (float(row["start"]) + float(row["end"]))
        dist = abs(mid - center)
        if dist > window:
            continue
        cands.append((dist, idx, row))
    cands.sort(key=lambda item: (item[0], item[1]))

    char_budget = 4096 * 2
    used = 0
    capped: list[dict] = []
    for _, _, row in cands[: cfg.neighbor_max_turns]:
        cost = len(str(row.get("text", ""))) + 32
        if capped and used + cost > char_budget:
            break
        capped.append(row)
        used += cost
    return capped
