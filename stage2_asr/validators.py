from __future__ import annotations

from typing import Any

from stage2_asr.pinyin_util import pinyin_edit_distance, pinyin_equal
from stage2_asr.types import Edit, LlmJudgment


def span_local_char_count_ok(edit: Edit, max_delta: int = 1) -> bool:
    return abs(len(edit.span_out) - len(edit.span_asr)) <= max_delta


def validate_edits_span_local(edits: list[Edit], max_delta: int = 1) -> tuple[bool, str | None]:
    for e in edits:
        if e.tier == "punct":
            continue
        if not span_local_char_count_ok(e, max_delta=max_delta):
            return False, f"span_local_char_count failed for {e.span_asr!r} -> {e.span_out!r}"
    return True, None


def validate_edits_evidence_ladder(
    edits: list[Edit],
    *,
    max_pinyin_edits: int = 2,
    allowed_tiers: set[str] | None = None,
) -> tuple[bool, str | None]:
    """Enforce Tier B exact pinyin / Tier C fuzzy pinyin + anchor (no context-only)."""
    for e in edits:
        if allowed_tiers is not None and e.tier not in allowed_tiers:
            return False, f"tier {e.tier!r} not allowed in this pass"
        if e.tier in {"A", "punct"}:
            continue
        if e.tier == "B":
            if not pinyin_equal(e.span_asr, e.span_out):
                return False, f"Tier B requires exact pinyin: {e.span_asr!r} vs {e.span_out!r}"
        elif e.tier == "C":
            if not e.anchor:
                return False, "Tier C requires anchor"
            if e.anchor not in {"hyp", "neighbor_draft", "meeting_draft", "hotword"}:
                return False, f"bad Tier C anchor {e.anchor}"
            dist = pinyin_edit_distance(e.span_asr, e.span_out)
            if dist > max_pinyin_edits:
                return False, f"Tier C pinyin edit distance {dist} > {max_pinyin_edits}"
        else:
            return False, f"unknown tier {e.tier}"
    return True, None


def validate_judgment_schema(payload: dict[str, Any], require_anchor_for_c: bool = True) -> tuple[bool, str | None]:
    if "text" not in payload or "base_model" not in payload:
        return False, "missing text or base_model"
    edits = payload.get("edits", [])
    if not isinstance(edits, list):
        return False, "edits must be list"
    for e in edits:
        if not isinstance(e, dict):
            return False, "edit must be dict"
        if "tier" not in e or "span_asr" not in e or "span_out" not in e:
            return False, "edit missing tier/span fields"
        tier = e.get("tier")
        if tier not in {"A", "B", "C", "punct"}:
            return False, f"bad tier {tier}"
        if require_anchor_for_c and tier == "C" and not e.get("anchor"):
            return False, "Tier C requires anchor"
    return True, None


def judgment_from_payload(payload: dict[str, Any]) -> LlmJudgment:
    edits = [
        Edit(
            span_asr=str(e.get("span_asr", "")),
            span_out=str(e.get("span_out", "")),
            tier=str(e.get("tier", "A")),
            pinyin_asr=str(e.get("pinyin_asr", "")),
            pinyin_out=str(e.get("pinyin_out", "")),
            anchor=e.get("anchor"),
        )
        for e in payload.get("edits", [])
    ]
    return LlmJudgment(
        text=str(payload.get("text", "")),
        base_model=str(payload.get("base_model", "")),
        edits=edits,
        overlap=bool(payload.get("overlap", False)),
        raw=payload,
    )
