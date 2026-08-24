from __future__ import annotations

import numpy as np

from stage2_asr.types import AsrUnit, PipelineConfig, Turn


def union_overlap_duration(unit_start: float, unit_end: float, others: list[Turn]) -> float:
    intervals: list[tuple[float, float]] = []
    for t in others:
        a = max(unit_start, float(t.start))
        b = min(unit_end, float(t.end))
        if b > a:
            intervals.append((a, b))
    if not intervals:
        return 0.0
    intervals.sort()
    merged: list[list[float]] = [[intervals[0][0], intervals[0][1]]]
    for a, b in intervals[1:]:
        if a <= merged[-1][1]:
            merged[-1][1] = max(merged[-1][1], b)
        else:
            merged.append([a, b])
    return float(sum(b - a for a, b in merged))


def overlap_ratio_for_span(
    unit_start: float,
    unit_end: float,
    speaker_id: str,
    all_turns: list[Turn],
) -> float:
    dur = float(unit_end) - float(unit_start)
    if dur <= 0:
        return 0.0
    others = [t for t in all_turns if t.speaker_id != speaker_id]
    return union_overlap_duration(unit_start, unit_end, others) / dur


def _rms_envelope(audio: np.ndarray, sr: int, window_ms: float, hop_ms: float) -> tuple[np.ndarray, np.ndarray]:
    win = max(1, int(sr * window_ms / 1000.0))
    hop = max(1, int(sr * hop_ms / 1000.0))
    if audio.size == 0:
        return np.zeros(0), np.zeros(0)
    n = 1 + max(0, (len(audio) - win) // hop)
    energies = np.zeros(n, dtype=np.float64)
    centers = np.zeros(n, dtype=np.float64)
    for i in range(n):
        s = i * hop
        e = min(len(audio), s + win)
        chunk = audio[s:e]
        energies[i] = float(np.sqrt(np.mean(chunk.astype(np.float64) ** 2) + 1e-12))
        centers[i] = (s + e) * 0.5 / sr
    return energies, centers


def find_min_energy_split_time(
    audio: np.ndarray,
    sr: int,
    duration: float,
    config: PipelineConfig,
) -> float:
    edge = config.edge_zone_ratio * duration
    lo, hi = edge, duration - edge
    if hi <= lo:
        return duration * 0.5
    energies, centers = _rms_envelope(audio, sr, config.energy_window_ms, config.energy_hop_ms)
    if energies.size == 0:
        return duration * 0.5
    mask = (centers >= lo) & (centers <= hi)
    if not np.any(mask):
        return duration * 0.5
    idx = int(np.argmin(np.where(mask, energies, np.inf)))
    return float(centers[idx])


def split_long_turn(
    turn: Turn,
    turn_index: int,
    audio: np.ndarray | None,
    sr: int,
    config: PipelineConfig,
    unit_id_prefix: str,
) -> list[tuple[int, Turn]]:
    dur = turn.duration
    if dur <= config.max_asr_seconds:
        return [(turn_index, turn)]
    if audio is None or audio.size == 0:
        mid = turn.start + dur * 0.5
    else:
        s = int(turn.start * sr)
        e = int(turn.end * sr)
        crop = audio[max(0, s) : max(s + 1, e)]
        split_rel = find_min_energy_split_time(crop, sr, dur, config)
        mid = turn.start + split_rel
        if mid <= turn.start + 0.01 or mid >= turn.end - 0.01:
            mid = turn.start + dur * 0.5
    left = Turn(turn.start, mid, turn.speaker_id, turn.text, turn.asr_status, turn.source, turn.confidence)
    right = Turn(mid, turn.end, turn.speaker_id, "", turn.asr_status, turn.source, turn.confidence)
    return split_long_turn(left, turn_index, audio, sr, config, f"{unit_id_prefix}a") + split_long_turn(
        right, turn_index, audio, sr, config, f"{unit_id_prefix}b"
    )


def build_asr_units(
    turns: list[Turn],
    config: PipelineConfig | None = None,
    audio: np.ndarray | None = None,
    sample_rate: int | None = None,
) -> list[AsrUnit]:
    cfg = config or PipelineConfig()
    sr = sample_rate or cfg.sample_rate
    if not turns:
        return []

    ordered = sorted(enumerate(turns), key=lambda it: (it[1].start, it[1].end, it[0]))
    atomic: list[tuple[int, Turn]] = []
    for i, t in ordered:
        atomic.extend(split_long_turn(t, i, audio, sr, cfg, f"u{i:04d}"))
    atomic.sort(key=lambda it: (it[1].start, it[1].end, it[0]))

    units: list[AsrUnit] = []
    unit_seq = 0
    cur_indices = [atomic[0][0]]
    cur_start = float(atomic[0][1].start)
    cur_end = float(atomic[0][1].end)
    cur_spk = atomic[0][1].speaker_id

    def flush() -> None:
        nonlocal unit_seq, cur_indices, cur_start, cur_end, cur_spk
        ratio = overlap_ratio_for_span(cur_start, cur_end, cur_spk, turns)
        dur = cur_end - cur_start
        skip = dur < cfg.min_asr_seconds
        units.append(
            AsrUnit(
                unit_id=f"unit_{unit_seq:04d}",
                start=cur_start,
                end=cur_end,
                speaker_id=cur_spk,
                turn_indices=list(dict.fromkeys(cur_indices)),
                overlap_ratio=ratio,
                contains_overlap=ratio > 0,
                heavy_overlap=ratio > cfg.heavy_overlap_threshold,
                skip_asr=skip,
                skip_reason="too_short" if skip else None,
                moss_merged=len(list(dict.fromkeys(cur_indices))) > 1,
            )
        )
        unit_seq += 1

    for idx, t in atomic[1:]:
        gap = float(t.start) - cur_end
        same = t.speaker_id == cur_spk
        new_span = float(t.end) - cur_start
        # gap < 0 means the next turn overlaps or is nested in the current unit.
        if same and gap <= cfg.max_gap_seconds and new_span <= cfg.max_asr_seconds:
            cur_indices.append(idx)
            cur_end = max(cur_end, float(t.end))
            continue
        flush()
        cur_indices = [idx]
        cur_start = float(t.start)
        cur_end = float(t.end)
        cur_spk = t.speaker_id
    flush()
    return units
