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


def _cap_neighbors(meeting: list[dict], turn_index: int, cfg: PipelineConfig) -> list[dict]:
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


def _hyps_for_turn(i: int, text: str, moss_texts: dict[int, str]) -> list[Hypothesis]:
    hyps: list[Hypothesis] = [Hypothesis("draft", text)]
    if i in moss_texts:
        hyps.insert(0, Hypothesis("moss", moss_texts[i]))
    return hyps


def _validate_pass_b_raw(raw) -> tuple[dict | None, str | None]:
    if isinstance(raw, BaseException):
        return None, str(raw)
    if not isinstance(raw, dict):
        return None, f"expected dict judgment, got {type(raw).__name__}"
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
    return _validate_pass_b_raw(raw)


def _apply_pass_b_accepted(
    *,
    out: dict[int, str],
    audits: list[dict],
    i: int,
    text: str,
    accepted: dict,
    overlap: bool,
    heavy: bool,
    used_fallback: bool,
    batched: bool = False,
) -> bool:
    """Apply a validated Pass B judgment. Returns True if the turn text changed."""
    extra = {"batched": True} if batched else {}
    judgment = judgment_from_payload(accepted)
    new_text = judgment.text
    if overlap or heavy:
        if judgment.base_model not in {"moss", "draft"} and not judgment.edits:
            audits.append(
                {
                    "turn_index": i,
                    "pass": "B",
                    "path": "moss_aware_reject",
                    "reason": "overlap_non_moss_base_without_edits",
                    "base_model": judgment.base_model,
                    **extra,
                }
            )
            return False
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
                    **extra,
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
                    **extra,
                }
            )
        return True
    return False


def _run_pass_b_batched(
    *,
    turns: list[Turn],
    out: dict[int, str],
    audits: list[dict],
    hotwords: list[str],
    llm_judge,
    fallback_judge,
    cfg: PipelineConfig,
    overlap_turn_indices: set[int],
    heavy_overlap_turn_indices: set[int],
    moss_texts: dict[int, str],
) -> tuple[dict[int, str], list[dict]]:
    """Snapshot meeting_draft once; judge_many in chunks (no in-pass cascade)."""
    meeting = _meeting_draft(turns, out)
    batch_size = max(1, int(getattr(cfg, "pass_b_batch_size", 1) or 1))
    prepared: list[dict] = []
    for i, turn in enumerate(turns):
        text = out.get(i, turn.text)
        if not text:
            continue
        if turn.duration + 1e-9 < cfg.min_asr_seconds:
            continue
        prepared.append(
            {
                "i": i,
                "text": text,
                "overlap": i in overlap_turn_indices,
                "heavy": i in heavy_overlap_turn_indices,
                "hyps": _hyps_for_turn(i, text, moss_texts),
                "neighbors": _cap_neighbors(meeting, i, cfg),
                "unit_id": f"pass_b_t{i}",
                "last_err": None,
            }
        )

    def _jobs_for(chunk: list[dict]) -> list[dict]:
        return [
            {
                "hypotheses": p["hyps"],
                "neighbor_draft": p["neighbors"],
                "hotwords": hotwords,
                "overlap": p["overlap"],
                "heavy_overlap": p["heavy"],
                "unit_id": p["unit_id"],
            }
            for p in chunk
        ]

    def _accept_or_defer(chunk: list[dict], raws: list, *, attempt: int) -> list[dict]:
        deferred: list[dict] = []
        for p, raw in zip(chunk, raws):
            accepted, err = _validate_pass_b_raw(raw)
            if accepted is None:
                p["last_err"] = err
                p["retries"] = attempt
                deferred.append(p)
                continue
            _apply_pass_b_accepted(
                out=out,
                audits=audits,
                i=p["i"],
                text=p["text"],
                accepted=accepted,
                overlap=p["overlap"],
                heavy=p["heavy"],
                used_fallback=bool(p.get("used_fallback")),
                batched=True,
            )
        return deferred

    pending = list(prepared)
    while pending:
        chunk = pending[:batch_size]
        pending = pending[batch_size:]
        still = _accept_or_defer(
            chunk,
            llm_judge.judge_many(_jobs_for(chunk), max_workers=batch_size),
            attempt=1,
        )
        for attempt in range(2, cfg.llm_max_retries + 2):
            if not still:
                break
            still = _accept_or_defer(
                still,
                llm_judge.judge_many(_jobs_for(still), max_workers=batch_size),
                attempt=attempt,
            )

        if still and fallback_judge is not None and hasattr(fallback_judge, "judge_many"):
            before = list(still)
            for p in before:
                p["used_fallback"] = True
            still = _accept_or_defer(
                before,
                fallback_judge.judge_many(_jobs_for(before), max_workers=batch_size),
                attempt=max(int(before[0].get("retries") or 0), 1),
            )
            recovered = {p["i"] for p in before} - {p["i"] for p in still}
            for a in audits:
                if a.get("turn_index") in recovered and a.get("path") == "llm":
                    a["fallback_judge_ok"] = True
                    a["fallback_judge"] = getattr(fallback_judge, "name", "fallback")
        elif still and fallback_judge is not None:
            next_still: list[dict] = []
            for p in still:
                p["used_fallback"] = True
                raw, err = _try_pass_b_judge(
                    fallback_judge,
                    hyps=p["hyps"],
                    neighbors=p["neighbors"],
                    hotwords=hotwords,
                    overlap=p["overlap"],
                    heavy_overlap=p["heavy"],
                    unit_id=p["unit_id"],
                )
                if raw is None:
                    p["last_err"] = err or p.get("last_err")
                    next_still.append(p)
                    continue
                _apply_pass_b_accepted(
                    out=out,
                    audits=audits,
                    i=p["i"],
                    text=p["text"],
                    accepted=raw,
                    overlap=p["overlap"],
                    heavy=p["heavy"],
                    used_fallback=True,
                    batched=True,
                )
            still = next_still

        for p in still:
            audits.append(
                {
                    "turn_index": p["i"],
                    "pass": "B",
                    "path": "llm",
                    "fallback": True,
                    "batched": True,
                    "fallback_judge": getattr(fallback_judge, "name", None)
                    if p.get("used_fallback")
                    else None,
                    "last_error": p.get("last_err"),
                }
            )
    return out, audits


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
       (sequential by default; `--pass-b-batch-size N` snapshots neighbors and batches)
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

    batch_size = max(1, int(getattr(cfg, "pass_b_batch_size", 1) or 1))
    if batch_size > 1 and hasattr(llm_judge, "judge_many"):
        return _run_pass_b_batched(
            turns=turns,
            out=out,
            audits=audits,
            hotwords=hotwords,
            llm_judge=llm_judge,
            fallback_judge=fallback_judge,
            cfg=cfg,
            overlap_turn_indices=overlap_turn_indices,
            heavy_overlap_turn_indices=heavy_overlap_turn_indices,
            moss_texts=moss_texts,
        )

    meeting = _meeting_draft(turns, out)
    for i, turn in enumerate(turns):
        text = out.get(i, turn.text)
        if not text:
            continue
        if turn.duration + 1e-9 < cfg.min_asr_seconds:
            continue

        overlap = i in overlap_turn_indices
        heavy = i in heavy_overlap_turn_indices
        hyps = _hyps_for_turn(i, text, moss_texts)
        capped = _cap_neighbors(meeting, i, cfg)

        unit_id = f"pass_b_t{i}"
        accepted = None
        last_err = None
        used_fallback = False
        for _attempt in range(cfg.llm_max_retries + 1):
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

        changed = _apply_pass_b_accepted(
            out=out,
            audits=audits,
            i=i,
            text=text,
            accepted=accepted,
            overlap=overlap,
            heavy=heavy,
            used_fallback=used_fallback,
            batched=False,
        )
        if changed:
            meeting = _meeting_draft(turns, out)

    return out, audits
