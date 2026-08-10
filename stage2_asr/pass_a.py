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
    # Design cap is 4096 tokens; approximate with ~2 chars/token.
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


def _try_judge(
    llm_judge,
    *,
    hyps: list[Hypothesis],
    neighbors: list[dict],
    hotwords: list[str],
    unit: AsrUnit,
) -> tuple[dict | None, str | None]:
    try:
        raw = llm_judge.judge(
            hypotheses=hyps,
            neighbor_draft=neighbors,
            hotwords=hotwords,
            overlap=unit.contains_overlap,
            heavy_overlap=unit.heavy_overlap,
            unit_id=unit.unit_id,
        )
    except Exception as exc:  # noqa: BLE001 — fallback path must stay resilient
        return None, str(exc)
    ok, err = validate_judgment_schema(raw)
    if not ok:
        return None, err
    judgment = judgment_from_payload(raw)
    ok2, err2 = validate_edits_span_local(judgment.edits)
    if not ok2:
        return None, err2
    ok3, err3 = validate_edits_evidence_ladder(judgment.edits)
    if not ok3:
        return None, err3
    return raw, None


def run_pass_a_for_unit(
    *,
    unit: AsrUnit,
    turns: list[Turn],
    hyps: list[Hypothesis],
    draft_texts: dict[int, str],
    llm_judge,
    hotwords: list[str],
    config: PipelineConfig,
    fallback_judge=None,
) -> tuple[str, dict]:
    """Return (final_text, audit_record)."""
    prefer_moss = unit.contains_overlap or unit.heavy_overlap
    audit: dict = {
        "unit_id": unit.unit_id,
        "heavy_overlap": unit.heavy_overlap,
        "skipped_llm": False,
        "retries": 0,
        "fallback": False,
        "fallback_judge": None,
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
        raw, err = _try_judge(
            llm_judge,
            hyps=hyps,
            neighbors=neighbors,
            hotwords=hotwords,
            unit=unit,
        )
        if raw is None:
            last_err = err
            audit["retries"] = attempt + 1
            continue
        judgment = judgment_from_payload(raw)
        if prefer_moss and judgment.base_model != "moss" and any(h.model == "moss" for h in hyps):
            if not judgment.edits:
                moss = pick_best_hyp(hyps, prefer_moss_on_overlap=True)
                text = moss.text if moss else judgment.text
                audit["forced_moss_base"] = True
                audit["judgment"] = judgment.raw
                return text, audit
        audit["judgment"] = judgment.raw
        return judgment.text, audit

    if fallback_judge is not None:
        audit["fallback_judge"] = getattr(fallback_judge, "name", "fallback")
        raw, err = _try_judge(
            fallback_judge,
            hyps=hyps,
            neighbors=neighbors,
            hotwords=hotwords,
            unit=unit,
        )
        if raw is not None:
            judgment = judgment_from_payload(raw)
            if prefer_moss and judgment.base_model != "moss" and any(
                h.model == "moss" for h in hyps
            ):
                if not judgment.edits:
                    moss = pick_best_hyp(hyps, prefer_moss_on_overlap=True)
                    text = moss.text if moss else judgment.text
                    audit["forced_moss_base"] = True
                    audit["judgment"] = judgment.raw
                    return text, audit
            audit["judgment"] = judgment.raw
            audit["fallback_judge_ok"] = True
            return judgment.text, audit
        last_err = err or last_err

    best = pick_best_hyp(hyps, prefer_moss_on_overlap=prefer_moss)
    audit["fallback"] = True
    audit["last_error"] = last_err
    return (best.text if best else ""), audit


def _finalize_from_raw(
    *,
    raw: dict,
    unit: AsrUnit,
    hyps: list[Hypothesis],
    prefer_moss: bool,
    audit: dict,
) -> tuple[str, dict]:
    judgment = judgment_from_payload(raw)
    if prefer_moss and judgment.base_model != "moss" and any(h.model == "moss" for h in hyps):
        if not judgment.edits:
            moss = pick_best_hyp(hyps, prefer_moss_on_overlap=True)
            text = moss.text if moss else judgment.text
            audit["forced_moss_base"] = True
            audit["judgment"] = judgment.raw
            return text, audit
    audit["judgment"] = judgment.raw
    return judgment.text, audit


def run_pass_a_batch(
    *,
    items: list[dict],
    turns: list[Turn],
    draft_texts: dict[int, str],
    llm_judge,
    hotwords: list[str],
    config: PipelineConfig,
    fallback_judge=None,
) -> list[tuple[str, dict]]:
    """
    Run Pass A for several units with a shared draft snapshot for neighbors.

    items: [{unit, hyps}, ...]
    When config.pass_a_batch_size > 1 and judge exposes judge_many (vLLM),
    first-attempt and retry LLM calls are issued via judge_many (batched).
    """
    batch_size = max(1, int(getattr(config, "pass_a_batch_size", 1) or 1))
    # Fast path: sequential (preserves per-unit draft updates between units).
    if batch_size <= 1 or len(items) <= 1 or not hasattr(llm_judge, "judge_many"):
        out_seq: list[tuple[str, dict]] = []
        for item in items:
            text, audit = run_pass_a_for_unit(
                unit=item["unit"],
                turns=turns,
                hyps=item["hyps"],
                draft_texts=draft_texts,
                llm_judge=llm_judge,
                hotwords=hotwords,
                config=config,
                fallback_judge=fallback_judge,
            )
            out_seq.append((text, audit))
            for ti, piece in _distribute_placeholder(item["unit"], text, turns):
                draft_texts[ti] = piece
        return out_seq

    # Snapshot neighbors from current draft for the whole micro-batch.
    prepared: list[dict] = []
    results: list[tuple[str, dict] | None] = [None] * len(items)

    for i, item in enumerate(items):
        unit: AsrUnit = item["unit"]
        hyps: list[Hypothesis] = item["hyps"]
        prefer_moss = unit.contains_overlap or unit.heavy_overlap
        audit: dict = {
            "unit_id": unit.unit_id,
            "heavy_overlap": unit.heavy_overlap,
            "skipped_llm": False,
            "retries": 0,
            "fallback": False,
            "fallback_judge": None,
            "batched": True,
        }
        if (not unit.contains_overlap) and all_hyps_agree(hyps):
            best = pick_best_hyp(hyps, prefer_moss_on_overlap=False)
            results[i] = (best.text if best else "", {**audit, "skipped_llm": True, "reason": "cer0_agree"})
            continue
        neighbors = _neighbor_draft(turns, unit, draft_texts, config)
        prepared.append(
            {
                "index": i,
                "unit": unit,
                "hyps": hyps,
                "prefer_moss": prefer_moss,
                "audit": audit,
                "neighbors": neighbors,
            }
        )

    def _jobs_for(chunk: list[dict]) -> list[dict]:
        return [
            {
                "hypotheses": p["hyps"],
                "neighbor_draft": p["neighbors"],
                "hotwords": hotwords,
                "overlap": p["unit"].contains_overlap,
                "heavy_overlap": p["unit"].heavy_overlap,
                "unit_id": p["unit"].unit_id,
            }
            for p in chunk
        ]

    def _accept_or_defer(
        chunk: list[dict],
        raws: list,
        *,
        attempt: int,
    ) -> list[dict]:
        """Finalize valid judgments; return items that still need another try."""
        deferred: list[dict] = []
        for p, raw in zip(chunk, raws):
            i = p["index"]
            audit = p["audit"]
            if isinstance(raw, BaseException):
                audit["retries"] = attempt
                audit["last_error"] = str(raw)
                deferred.append(p)
                continue
            ok, err = validate_judgment_schema(raw)
            if not ok:
                audit["retries"] = attempt
                audit["last_error"] = err
                deferred.append(p)
                continue
            judgment = judgment_from_payload(raw)
            ok2, err2 = validate_edits_span_local(judgment.edits)
            if not ok2:
                audit["retries"] = attempt
                audit["last_error"] = err2
                deferred.append(p)
                continue
            ok3, err3 = validate_edits_evidence_ladder(judgment.edits)
            if not ok3:
                audit["retries"] = attempt
                audit["last_error"] = err3
                deferred.append(p)
                continue
            text, audit2 = _finalize_from_raw(
                raw=raw,
                unit=p["unit"],
                hyps=p["hyps"],
                prefer_moss=p["prefer_moss"],
                audit=audit,
            )
            results[i] = (text, audit2)
        return deferred

    # Chunked first attempt + re-batched retries (avoid serial 1/1 tail).
    pending = list(prepared)
    while pending:
        chunk = pending[:batch_size]
        pending = pending[batch_size:]
        still = _accept_or_defer(
            chunk,
            llm_judge.judge_many(_jobs_for(chunk), max_workers=batch_size),
            attempt=1,
        )
        # attempt 0 was the first batch; remaining retries are 2..llm_max_retries+1
        for attempt in range(2, config.llm_max_retries + 2):
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
                p["audit"]["fallback_judge"] = getattr(fallback_judge, "name", "fallback")
            still = _accept_or_defer(
                before,
                fallback_judge.judge_many(_jobs_for(before), max_workers=batch_size),
                attempt=max(int(before[0]["audit"].get("retries") or 0), 1),
            )
            recovered = {p["index"] for p in before} - {p["index"] for p in still}
            for idx in recovered:
                text, audit = results[idx]  # type: ignore[misc]
                audit["fallback_judge_ok"] = True
                results[idx] = (text, audit)
        elif still and fallback_judge is not None:
            next_still: list[dict] = []
            for p in still:
                audit = p["audit"]
                audit["fallback_judge"] = getattr(fallback_judge, "name", "fallback")
                raw, err = _try_judge(
                    fallback_judge,
                    hyps=p["hyps"],
                    neighbors=p["neighbors"],
                    hotwords=hotwords,
                    unit=p["unit"],
                )
                if raw is None:
                    audit["last_error"] = err or audit.get("last_error")
                    next_still.append(p)
                    continue
                text, audit2 = _finalize_from_raw(
                    raw=raw,
                    unit=p["unit"],
                    hyps=p["hyps"],
                    prefer_moss=p["prefer_moss"],
                    audit=audit,
                )
                audit2["fallback_judge_ok"] = True
                results[p["index"]] = (text, audit2)
            still = next_still

        for p in still:
            best = pick_best_hyp(p["hyps"], prefer_moss_on_overlap=p["prefer_moss"])
            audit = p["audit"]
            audit["fallback"] = True
            audit["batched"] = True
            results[p["index"]] = ((best.text if best else ""), audit)

    # Apply draft updates in original order.
    out: list[tuple[str, dict]] = []
    for i, item in enumerate(items):
        assert results[i] is not None
        text, audit = results[i]  # type: ignore[misc]
        out.append((text, audit))
        for ti, piece in _distribute_placeholder(item["unit"], text, turns):
            draft_texts[ti] = piece
    return out


def _distribute_placeholder(unit: AsrUnit, text: str, turns: list[Turn]):
    """Yield (turn_index, text_piece) by relative duration (same as pipeline)."""
    idxs = unit.turn_indices
    if not idxs:
        return
    if len(idxs) == 1:
        yield idxs[0], text
        return
    durs = [max(1e-6, turns[i].duration) for i in idxs]
    total = sum(durs)
    chars = list(text)
    n = len(chars)
    cursor = 0
    for k, (idx, dur) in enumerate(zip(idxs, durs)):
        if k == len(idxs) - 1:
            yield idx, "".join(chars[cursor:])
        else:
            take = int(round(n * (dur / total)))
            yield idx, "".join(chars[cursor : cursor + take])
            cursor += take
