from __future__ import annotations

from stage2_asr.agreement import all_hyps_agree, pick_best_hyp
from stage2_asr.types import AsrUnit, Hypothesis, PipelineConfig, Turn
from stage2_asr.validators import (
    judgment_from_payload,
    validate_edits_evidence_ladder,
    validate_edits_span_local,
    validate_judgment_schema,
)


def _neighbor_draft(
    turns: list[Turn],
    unit: AsrUnit,
    draft_texts: dict[int, str],
    config: PipelineConfig,
) -> list[dict]:
    """Nearest turns within ±window, capped by neighbor_max_turns and ~4096-token char budget."""
    center = 0.5 * (unit.start + unit.end)
    cands: list[tuple[float, int, Turn]] = []
    for i, t in enumerate(turns):
        if abs(0.5 * (t.start + t.end) - center) > config.neighbor_window_seconds:
            continue
        if i in unit.turn_indices:
            continue
        dist = abs(0.5 * (t.start + t.end) - center)
        cands.append((dist, i, t))
    cands.sort(key=lambda x: x[0])
    out: list[dict] = []
    char_budget = 4096 * 2
    used = 0
    for _, i, t in cands[: config.neighbor_max_turns]:
        text = draft_texts.get(i, t.text)
        piece = {
            "turn_index": i,
            "start": t.start,
            "end": t.end,
            "speaker_id": t.speaker_id,
            "text": text,
        }
        cost = len(text) + 32
        if out and used + cost > char_budget:
            break
        out.append(piece)
        used += cost
    return out


def run_pass_a_for_unit(
    *,
    unit: AsrUnit,
    turns: list[Turn],
    hyps: list[Hypothesis],
    draft_texts: dict[int, str],
    llm_judge,
    hotwords: list[str],
    config: PipelineConfig,
) -> tuple[str, dict]:
    """Return (final_text, audit_record)."""
    prefer_moss = unit.contains_overlap or unit.heavy_overlap
    audit: dict = {
        "unit_id": unit.unit_id,
        "heavy_overlap": unit.heavy_overlap,
        "skipped_llm": False,
        "retries": 0,
        "fallback": False,
    }

    if (not unit.contains_overlap) and all_hyps_agree(hyps):
        best = pick_best_hyp(hyps, prefer_moss_on_overlap=False)
        text = best.text if best else ""
        audit["skipped_llm"] = True
        audit["reason"] = "cer0_agree"
        return text, audit

    neighbors = _neighbor_draft(turns, unit, draft_texts, config)
    last_err = None
    for attempt in range(config.llm_max_retries + 1):
        raw = llm_judge.judge(
            hypotheses=hyps,
            neighbor_draft=neighbors,
            hotwords=hotwords,
            overlap=unit.contains_overlap,
            heavy_overlap=unit.heavy_overlap,
            unit_id=unit.unit_id,
        )
        ok, err = validate_judgment_schema(raw)
        if not ok:
            last_err = err
            audit["retries"] = attempt + 1
            continue
        judgment = judgment_from_payload(raw)
        ok2, err2 = validate_edits_span_local(judgment.edits)
        if not ok2:
            last_err = err2
            audit["retries"] = attempt + 1
            continue
        ok3, err3 = validate_edits_evidence_ladder(judgment.edits)
        if not ok3:
            last_err = err3
            audit["retries"] = attempt + 1
            continue
        if prefer_moss and judgment.base_model != "moss" and any(h.model == "moss" for h in hyps):
            if not judgment.edits:
                moss = pick_best_hyp(hyps, prefer_moss_on_overlap=True)
                text = moss.text if moss else judgment.text
                audit["forced_moss_base"] = True
                audit["judgment"] = judgment.raw
                return text, audit
        audit["judgment"] = judgment.raw
        return judgment.text, audit

    best = pick_best_hyp(hyps, prefer_moss_on_overlap=prefer_moss)
    audit["fallback"] = True
    audit["last_error"] = last_err
    return (best.text if best else ""), audit
