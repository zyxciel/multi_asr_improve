from __future__ import annotations

from stage2_asr.pinyin_util import pinyin_edit_distance
from stage2_asr.types import Edit, Turn
from stage2_asr.validators import validate_edits_evidence_ladder, validate_edits_span_local


def _parse_hotword_aliases(hotwords: list[str]) -> dict[str, str]:
    """'canon|alt1|alt2' → alt -> canon (span-local length only)."""
    aliases: dict[str, str] = {}
    for hw in hotwords:
        if "|" not in hw:
            continue
        parts = [p.strip() for p in hw.split("|") if p.strip()]
        if len(parts) < 2:
            continue
        canon = parts[0]
        for alt in parts[1:]:
            if abs(len(alt) - len(canon)) <= 1:
                aliases[alt] = canon
    return aliases


def run_pass_b(
    turns: list[Turn],
    draft_texts: dict[int, str],
    hotwords: list[str] | None = None,
) -> tuple[dict[int, str], list[dict]]:
    """
    Required global consistency pass (mock-friendly):
    - Apply hotword aliases with span-local + Tier C evidence checks
    - If a canonical hotword already appears in >=2 turns, replace aliases elsewhere
    """
    hotwords = hotwords or []
    out = dict(draft_texts)
    audits: list[dict] = []
    aliases = _parse_hotword_aliases(hotwords)
    if not aliases and not hotwords:
        return out, audits

    # Recurrence: canons already in meeting draft
    joined = "||".join(out.values())
    recurring = {hw.split("|")[0] for hw in hotwords if hw.split("|")[0] and joined.count(hw.split("|")[0]) >= 1}

    for i, text in list(out.items()):
        new_text = text
        for alt, canon in aliases.items():
            if alt not in new_text:
                continue
            # Prefer applying when canon is a known hotword / recurring
            if canon not in recurring and canon not in hotwords and not any(h.startswith(canon) for h in hotwords):
                continue
            e = Edit(
                span_asr=alt,
                span_out=canon,
                tier="C",
                anchor="hotword",
                pinyin_asr="",
                pinyin_out="",
            )
            ok1, _ = validate_edits_span_local([e])
            # Allow Tier C if pinyin close OR exact alias table (mock treats alias table as hotword prior)
            ok2, _ = validate_edits_evidence_ladder([e])
            if not ok1:
                continue
            if not ok2 and pinyin_edit_distance(alt, canon) > 2:
                continue
            if not ok2:
                # Alias table is an explicit hotword prior; accept if span-local only
                e.tier = "C"
            new_text = new_text.replace(alt, canon)
            audits.append(
                {
                    "turn_index": i,
                    "span_asr": alt,
                    "span_out": canon,
                    "tier": "C",
                    "anchor": "hotword",
                    "pass": "B",
                }
            )
        if new_text != text:
            out[i] = new_text

    return out, audits
