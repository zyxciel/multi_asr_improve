from __future__ import annotations

import math

from stage2_asr.agreement import normalize_for_cer
from stage2_asr.types import AsrStatus, Turn

JOINER = "。"
_END_PUNCT = set("。，、？！；,.!?")


def join_turn_texts(texts: list[str]) -> str:
    """Join per-turn moss/unit strings without doubling terminal punctuation."""
    pieces = [t.strip() for t in texts if (t or "").strip()]
    if not pieces:
        return ""
    out = pieces[0]
    for piece in pieces[1:]:
        if out[-1] not in _END_PUNCT:
            out += JOINER
        out += piece
    return out


def _join_merge_texts(parts: list[str]) -> str:
    pieces = [p.strip() for p in parts if (p or "").strip()]
    if not pieces:
        return ""
    out = pieces[0]
    for piece in pieces[1:]:
        if out[-1].isascii() and piece[0].isascii() and (out[-1].isalnum() or piece[0].isalnum()):
            out += " " + piece
        else:
            out += piece
    return out


def merge_consecutive_turns(
    turns: list[Turn],
    texts: dict[int, str] | None = None,
    *,
    max_duration: float,
    max_merge_gap: float,
) -> tuple[list[Turn], list[list[int]]]:
    """Merge adjacent same-speaker rows. Timestamps stay original endpoints.

    Same rule as a TSV segment merge: split overlong rows into equal chunks,
    then join only originally consecutive rows of one speaker when the gap and
    combined duration allow it. Overlapping other speakers are left as-is.
    """
    if not turns:
        return [], []

    cleaned: list[tuple[int, float, float, str, str]] = []
    for i, t in enumerate(turns):
        start = float(t.start)
        end = float(t.end)
        spk = str(t.speaker_id)
        if texts is None:
            text = str(t.text or "")
        else:
            text = str(texts.get(i, t.text) or "")
        duration = end - start
        if duration > max_duration and max_duration > 0:
            n_chunks = int(math.ceil(duration / max_duration))
            for k in range(n_chunks):
                chunk_start = start + k * max_duration
                chunk_end = min(start + (k + 1) * max_duration, end)
                cleaned.append((i, chunk_start, chunk_end, spk, text if k == 0 else ""))
        else:
            cleaned.append((i, start, end, spk, text))

    groups: dict[str, list[tuple[int, tuple[int, float, float, str, str]]]] = {}
    for new_id, row in enumerate(cleaned):
        groups.setdefault(row[3], []).append((new_id, row))

    merged_rows: list[tuple[float, float, str, str, list[int]]] = []
    for spk, group in groups.items():
        group.sort(key=lambda item: item[0])
        _new_id0, row0 = group[0]
        seg_start = row0[1]
        seg_end = row0[2]
        seg_ids = [_new_id0]
        seg_orig = [row0[0]]
        seg_texts = [row0[4]] if row0[4].strip() else []

        for new_id, row in group[1:]:
            orig_i, curr_start, curr_end, _spk, curr_text = row
            gap = curr_start - seg_end
            can_merge = (
                new_id == seg_ids[-1] + 1
                and gap <= max_merge_gap
                and (curr_end - seg_start) <= max_duration
            )
            if can_merge:
                seg_end = max(seg_end, curr_end)
                seg_ids.append(new_id)
                seg_orig.append(orig_i)
                if curr_text.strip():
                    seg_texts.append(curr_text)
                continue
            merged_rows.append(
                (seg_start, seg_end, spk, _join_merge_texts(seg_texts), list(dict.fromkeys(seg_orig)))
            )
            seg_start = curr_start
            seg_end = curr_end
            seg_ids = [new_id]
            seg_orig = [orig_i]
            seg_texts = [curr_text] if curr_text.strip() else []
        merged_rows.append(
            (seg_start, seg_end, spk, _join_merge_texts(seg_texts), list(dict.fromkeys(seg_orig)))
        )

    merged_rows.sort(key=lambda item: (item[0], item[1], item[2]))
    out: list[Turn] = []
    members: list[list[int]] = []
    for start, end, spk, text, origs in merged_rows:
        src = turns[origs[-1]]
        out.append(
            Turn(
                start=start,
                end=end,
                speaker_id=spk,
                text=text,
                asr_status=AsrStatus.FINAL if text else AsrStatus.EMPTY,
                source=src.source,
                confidence=src.confidence,
            )
        )
        members.append(origs)
    return out, members


def _split_on_joiner(text: str, n: int) -> list[str] | None:
    if n <= 0:
        return None
    if n == 1:
        return [text]
    parts = [p for p in text.split(JOINER) if p]
    if len(parts) == n:
        return parts
    return None


def _duration_split(text: str, unit_turn_indices: list[int], turns: list[Turn]) -> dict[int, str]:
    durs = [max(1e-6, turns[i].duration) for i in unit_turn_indices]
    total = sum(durs)
    chars = list(text)
    n = len(chars)
    out: dict[int, str] = {}
    cursor = 0
    for k, (idx, dur) in enumerate(zip(unit_turn_indices, durs)):
        if k == len(unit_turn_indices) - 1:
            piece = "".join(chars[cursor:])
        else:
            take = int(round(n * (dur / total)))
            piece = "".join(chars[cursor : cursor + take])
            cursor += take
        out[idx] = piece.strip(JOINER).replace(JOINER + JOINER, JOINER)
    return out


def _is_partial_turn_slice(
    turn: Turn, *, unit_start: float, unit_end: float, n_indices: int
) -> bool:
    if n_indices != 1:
        return False
    return float(unit_start) > float(turn.start) + 1e-6 or float(unit_end) < float(turn.end) - 1e-6


def _merge_slice_text(existing: str, piece: str, turn: Turn | None) -> str:
    """Merge slice texts of one long turn without duplicating full-turn content.

    A split unit's hyp may already carry the complete turn text (e.g. the moss
    hyp built from Mode-C text, which is never split). In that case the piece
    (or the existing dest value) supersedes partial slices instead of joining.
    """
    turn_text = (turn.text or "") if turn is not None else ""
    p_norm = normalize_for_cer(piece)
    e_norm = normalize_for_cer(existing)
    t_norm = normalize_for_cer(turn_text)
    if t_norm and p_norm == t_norm:
        return piece
    if t_norm and e_norm == t_norm:
        return existing
    if e_norm and p_norm == e_norm:
        return existing
    return join_turn_texts([existing, piece])


def assign_unit_text(
    dest: dict[int, str],
    *,
    turn_indices: list[int],
    text: str,
    turns: list[Turn],
    unit_start: float,
    unit_end: float,
    written: set[int],
) -> None:
    """Map unit text onto original turns, concatenating slices of one long turn."""
    mapped = distribute_unit_text(turn_indices, text, turns)
    n_indices = len(list(dict.fromkeys(turn_indices)))
    for idx, piece in mapped.items():
        turn = turns[idx] if 0 <= idx < len(turns) else None
        is_slice = turn is not None and _is_partial_turn_slice(
            turn, unit_start=unit_start, unit_end=unit_end, n_indices=n_indices
        )
        if is_slice and idx in written:
            dest[idx] = _merge_slice_text(dest.get(idx, ""), piece, turn)
        else:
            dest[idx] = piece
        written.add(idx)


def distribute_unit_text(
    unit_turn_indices: list[int],
    text: str,
    turns: list[Turn],
) -> dict[int, str]:
    """Map unit-level text back to member turns.

    Prefer splitting on the moss/unit joiner when piece count matches;
    otherwise fall back to relative duration and strip leaked joiners.
    """
    if not unit_turn_indices:
        return {}
    if len(unit_turn_indices) == 1:
        return {unit_turn_indices[0]: text}
    parts = _split_on_joiner(text, len(unit_turn_indices))
    if parts is not None:
        return {idx: part for idx, part in zip(unit_turn_indices, parts)}
    return _duration_split(text, unit_turn_indices, turns)
