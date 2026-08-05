from __future__ import annotations

from stage2_asr.pinyin_util import pinyin_edit_distance
from stage2_asr.types import Edit, Hypothesis, PipelineConfig, Turn
from stage2_asr.validators import (
    judgment_from_payload,
    validate_edits_evidence_ladder,
    validate_edits_span_local,
    validate_judgment_schema,
)

_PASS_B_TIERS = {"B", "C", "punct"}


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


def _apply_hotword_aliases(
    draft_texts: dict[int, str],
    hotwords: list[str],
) -> tuple[dict[int, str], list[dict]]:
    out = dict(draft_texts)
    audits: list[dict] = []
    aliases = _parse_hotword_aliases(hotwords)
    if not aliases and not hotwords:
        return out, audits

    joined = "||".join(out.values())
    recurring = {
        hw.split("|")[0]
        for hw in hotwords
        if hw.split("|")[0] and joined.count(hw.split("|")[0]) >= 1
    }

    for i, text in list(out.items()):
        new_text = text
        for alt, canon in aliases.items():
            if alt not in new_text:
                continue
            if (
                canon not in recurring
                and canon not in hotwords
                and not any(h.startswith(canon) for h in hotwords)
            ):
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
            ok2, _ = validate_edits_evidence_ladder([e])
            if not ok1:
                continue
            if not ok2 and pinyin_edit_distance(alt, canon) > 2:
                continue
            new_text = new_text.replace(alt, canon)
            audits.append(
                {
                    "turn_index": i,
                    "span_asr": alt,
                    "span_out": canon,
                    "tier": "C",
                    "anchor": "hotword",
                    "pass": "B",
                    "path": "hotword_alias",
                }
            )
        if new_text != text:
            out[i] = new_text
    return out, audits


def _meeting_draft(turns: list[Turn], texts: dict[int, str]) -> list[dict]:
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


def _try_pass_b_judge(
    llm_judge,
    *,
    hyps: list[Hypothesis],
    neighbors: list[dict],
    hotwords: list[str],
    overlap: bool,
    heavy_overlap: bool,
    unit_id: str,
) -> tuple[dict | None, str | None]:
    try:
        raw = llm_judge.judge(
            hypotheses=hyps,
            neighbor_draft=neighbors,
            hotwords=hotwords,
            overlap=overlap,
            heavy_overlap=heavy_overlap,
            unit_id=unit_id,
        )
    except Exception as exc:  # noqa: BLE001
        return None, str(exc)
    ok, err = validate_judgment_schema(raw)
    if not ok:
        return None, err
    judgment = judgment_from_payload(raw)
    if any(e.tier == "A" for e in judgment.edits):
        return None, "pass_b_rejects_tier_a"
    ok2, err2 = validate_edits_span_local(judgment.edits)
    if not ok2:
        return None, err2
    ok3, err3 = validate_edits_evidence_ladder(
        judgment.edits, allowed_tiers=_PASS_B_TIERS
    )
    if not ok3:
        return None, err3
    return raw, None


def run_pass_b(
    turns: list[Turn],
    draft_texts: dict[int, str],
    hotwords: list[str] | None = None,
    *,
    llm_judge=None,
    fallback_judge=None,
    config: PipelineConfig | None = None,
    overlap_turn_indices: set[int] | None = None,
    heavy_overlap_turn_indices: set[int] | None = None,
    moss_texts: dict[int, str] | None = None,
) -> tuple[dict[int, str], list[dict]]:
    """
    Required global consistency pass:
    1) Hotword alias fast path (span-local + Tier C)
    2) Optional LLM Tier B/C scan with full meeting_draft as neighbors
    3) MOSS-aware: overlap turns prefer / force moss provisional text
    4) Optional fallback_judge (e.g. DeepSeek) after primary retries fail
    """
    cfg = config or PipelineConfig()
    hotwords = hotwords or []
    overlap_turn_indices = overlap_turn_indices or set()
    heavy_overlap_turn_indices = heavy_overlap_turn_indices or set()
    moss_texts = moss_texts or {}

    out, audits = _apply_hotword_aliases(draft_texts, hotwords)

    if llm_judge is None:
        # Without LLM, still enforce moss base on overlap when draft equals a foreign hyp
        # is not knowable here; only restore moss if draft is empty but moss exists.
        for i in sorted(overlap_turn_indices | heavy_overlap_turn_indices):
            moss = moss_texts.get(i)
            if moss and not (out.get(i) or "").strip():
                out[i] = moss
                audits.append(
                    {
                        "turn_index": i,
                        "pass": "B",
                        "path": "moss_force",
                        "reason": "overlap_empty_restore_moss",
                    }
                )
        return out, audits

    meeting = _meeting_draft(turns, out)
    for i, turn in enumerate(turns):
        text = out.get(i, turn.text)
        if not text:
            continue
        if turn.duration + 1e-9 < cfg.min_asr_seconds:
            continue

        overlap = i in overlap_turn_indices
        heavy = i in heavy_overlap_turn_indices
        hyps: list[Hypothesis] = [Hypothesis("draft", text)]
        if i in moss_texts:
            hyps.insert(0, Hypothesis("moss", moss_texts[i]))

        neighbors = [row for row in meeting if row["turn_index"] != i]
        # Cap neighbors similarly to Pass A (~4096 tokens ≈ 8192 chars).
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

        unit_id = f"pass_b_t{i}"
        accepted = None
        last_err = None
        used_fallback = False
        for attempt in range(cfg.llm_max_retries + 1):
            raw, err = _try_pass_b_judge(
                llm_judge,
                hyps=hyps,
                neighbors=capped,
                hotwords=hotwords,
                overlap=overlap,
                heavy_overlap=heavy,
                unit_id=unit_id,
            )
            if raw is None:
                last_err = err
                continue
            accepted = raw
            break

        if accepted is None and fallback_judge is not None:
            used_fallback = True
            raw, err = _try_pass_b_judge(
                fallback_judge,
                hyps=hyps,
                neighbors=capped,
                hotwords=hotwords,
                overlap=overlap,
                heavy_overlap=heavy,
                unit_id=unit_id,
            )
            if raw is None:
                last_err = err or last_err
            else:
                accepted = raw

        if accepted is None:
            audits.append(
                {
                    "turn_index": i,
                    "pass": "B",
                    "path": "llm",
                    "fallback": True,
                    "fallback_judge": getattr(fallback_judge, "name", None)
                    if used_fallback
                    else None,
                    "last_error": last_err,
                }
            )
            continue

        judgment = judgment_from_payload(accepted)
        new_text = judgment.text
        if overlap or heavy:
            # MOSS-aware: reject non-moss/non-draft base with no Tier B/C evidence
            # (do not wipe Pass A repairs by forcing raw moss back).
            if judgment.base_model not in {"moss", "draft"} and not judgment.edits:
                audits.append(
                    {
                        "turn_index": i,
                        "pass": "B",
                        "path": "moss_aware_reject",
                        "reason": "overlap_non_moss_base_without_edits",
                        "base_model": judgment.base_model,
                    }
                )
                continue

        if new_text != text:
            out[i] = new_text
            for e in judgment.edits:
                audits.append(
                    {
                        "turn_index": i,
                        "pass": "B",
                        "path": "llm",
                        "span_asr": e.span_asr,
                        "span_out": e.span_out,
                        "tier": e.tier,
                        "anchor": e.anchor,
                        "fallback_judge_ok": used_fallback or None,
                    }
                )
            if not judgment.edits:
                audits.append(
                    {
                        "turn_index": i,
                        "pass": "B",
                        "path": "llm",
                        "tier": "B",
                        "anchor": "meeting_draft",
                        "span_asr": text,
                        "span_out": new_text,
                        "fallback_judge_ok": used_fallback or None,
                    }
                )
            # Refresh meeting draft for subsequent turns
            meeting = _meeting_draft(turns, out)

    return out, audits
