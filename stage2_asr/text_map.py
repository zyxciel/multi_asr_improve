from __future__ import annotations

from stage2_asr.types import AsrStatus, AsrUnit, Turn

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


def stitch_member_texts(pieces: list[str]) -> str:
    """Reassemble mapped-back fragments without injecting a sentence joiner.

    Duration-split pieces such as 以前那个温 + 度的问题 must become one sentence.
    """
    return "".join(t.strip() for t in pieces if (t or "").strip())


def merged_turns_from_units(
    turns: list[Turn],
    texts: dict[int, str],
    units: list[AsrUnit],
) -> list[Turn]:
    """One turn per ASR unit; text is stitched member-turn text."""
    out: list[Turn] = []
    for unit in units:
        pieces: list[str] = []
        source = "fused"
        confidence = 1.0
        for i in unit.turn_indices:
            if 0 <= i < len(turns):
                pieces.append(str(texts.get(i, turns[i].text) or ""))
                source = turns[i].source
                confidence = turns[i].confidence
            elif i in texts:
                pieces.append(str(texts[i] or ""))
        text = stitch_member_texts(pieces)
        out.append(
            Turn(
                start=float(unit.start),
                end=float(unit.end),
                speaker_id=str(unit.speaker_id),
                text=text,
                asr_status=AsrStatus.FINAL if text else AsrStatus.EMPTY,
                source=source,
                confidence=confidence,
            )
        )
    return out


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
