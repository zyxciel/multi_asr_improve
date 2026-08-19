from __future__ import annotations

from typing import Any

from stage2_asr.text_map import distribute_unit_text
from stage2_asr.types import Hypothesis, PipelineConfig, Turn

ALLOWED_KINDS = frozenset({"punc", "entity", "codeswitch", "itn"})


def apply_polish_edits(text: str, edits: list[dict]) -> tuple[str, list[dict]]:
    """Apply span replacements on `text`; return (new_text, located edits).

    Locations (`start_char`, `end_char`) are 0-based offsets in the *input* text.
    Overlapping or missing spans are skipped. Apply right-to-left so earlier
    offsets stay valid.
    """
    located: list[dict] = []
    occupied: list[tuple[int, int]] = []

    def _overlaps(start: int, end: int) -> bool:
        for a, b in occupied:
            if start < b and end > a:
                return True
        return False

    for raw in edits:
        if not isinstance(raw, dict):
            continue
        kind = str(raw.get("kind", "")).lower()
        if kind not in ALLOWED_KINDS:
            continue
        span_asr = str(raw.get("span_asr", ""))
        span_out = str(raw.get("span_out", ""))
        if span_asr == span_out:
            continue

        start_hint = raw.get("start_char")
        if span_asr == "":
            if not isinstance(start_hint, int) or not (0 <= start_hint <= len(text)):
                continue
            start, end = start_hint, start_hint
        else:
            start = None
            if isinstance(start_hint, int) and start_hint >= 0:
                stop = start_hint + len(span_asr)
                if stop <= len(text) and text[start_hint:stop] == span_asr:
                    start = start_hint
            if start is None:
                found = text.find(span_asr)
                if found < 0:
                    continue
                start = found
            end = start + len(span_asr)

        if _overlaps(start, end):
            continue
        occupied.append((start, end))
        item = {
            "span_asr": span_asr,
            "span_out": span_out,
            "kind": kind,
            "start_char": start,
            "end_char": end,
        }
        if raw.get("anchor"):
            item["anchor"] = str(raw["anchor"])
        located.append(item)

    out = text
    for loc in sorted(located, key=lambda x: (x["start_char"], x["end_char"]), reverse=True):
        out = out[: loc["start_char"]] + loc["span_out"] + out[loc["end_char"] :]
    return out, located


def hyps_by_turn_from_records(
    records: list[dict],
    turns: list[Turn],
) -> dict[int, list[Hypothesis]]:
    """Map saved asr_hypotheses.json records onto per-turn hyp lists."""
    out: dict[int, list[Hypothesis]] = {}
    for record in records or []:
        if not isinstance(record, dict):
            continue
        indices = [int(i) for i in (record.get("turn_indices") or [])]
        if not indices:
            continue
        unit_id = str(record.get("unit_id") or "")
        for raw in record.get("hyps") or []:
            if not isinstance(raw, dict):
                continue
            unit_text = str(raw.get("text") or "")
            model = str(raw.get("model") or "")
            mapped = distribute_unit_text(indices, unit_text, turns)
            for i in indices:
                piece = str(mapped.get(i) or unit_text)
                out.setdefault(i, []).append(
                    Hypothesis(
                        model=model,
                        text=piece,
                        lid=raw.get("lid"),
                        meta={
                            "unit_id": unit_id,
                            "unit_text": unit_text,
                        },
                    )
                )
    return out


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


def _validate_polish_raw(raw: Any) -> tuple[dict | None, str | None]:
    if isinstance(raw, BaseException):
        return None, str(raw)
    if not isinstance(raw, dict):
        return None, f"expected dict polish payload, got {type(raw).__name__}"
    if "text" not in raw:
        return None, "missing text"
    edits = raw.get("edits", [])
    if not isinstance(edits, list):
        return None, "edits must be list"
    for e in edits:
        if not isinstance(e, dict):
            return None, "edit must be dict"
        if "span_asr" not in e or "span_out" not in e or "kind" not in e:
            return None, "edit missing span_asr/span_out/kind"
        kind = str(e.get("kind", "")).lower()
        if kind not in ALLOWED_KINDS:
            return None, f"bad kind {kind!r}"
    return raw, None


def _try_polish(
    llm_judge,
    *,
    text: str,
    neighbors: list[dict],
    hotwords: list[str],
    turn_index: int,
    unit_id: str,
    hypotheses: list[Hypothesis] | None = None,
) -> tuple[dict | None, str | None]:
    try:
        raw = llm_judge.polish(
            text=text,
            neighbor_draft=neighbors,
            hotwords=hotwords,
            turn_index=turn_index,
            unit_id=unit_id,
            hypotheses=hypotheses or [],
        )
    except Exception as exc:  # noqa: BLE001
        return None, str(exc)
    return _validate_polish_raw(raw)


def _apply_polish_accepted(
    *,
    out: dict[int, str],
    audits: list[dict],
    turn: Turn,
    i: int,
    text: str,
    accepted: dict,
    batched: bool = False,
) -> bool:
    extra = {"batched": True} if batched else {}
    edits = accepted.get("edits") or []
    if not edits:
        if str(accepted.get("text", "")) != text:
            audits.append(
                {
                    "turn_index": i,
                    "pass": "polish",
                    "path": "empty_edits_reject",
                    "span_asr": text,
                    "span_out": accepted.get("text"),
                    **extra,
                }
            )
        return False
    new_text, located = apply_polish_edits(text, edits)
    if new_text == text:
        return False
    out[i] = new_text
    for loc in located:
        audits.append(
            {
                "turn_index": i,
                "pass": "polish",
                "path": "llm",
                "kind": loc["kind"],
                "span_asr": loc["span_asr"],
                "span_out": loc["span_out"],
                "start_char": loc["start_char"],
                "end_char": loc["end_char"],
                "turn_start": turn.start,
                "turn_end": turn.end,
                "speaker_id": turn.speaker_id,
                **extra,
            }
        )
        if loc.get("anchor"):
            audits[-1]["anchor"] = loc["anchor"]
    return True


def _jobs_for(chunk: list[dict], hotwords: list[str]) -> list[dict]:
    return [
        {
            "text": p["text"],
            "neighbor_draft": p["neighbors"],
            "hotwords": hotwords,
            "turn_index": p["i"],
            "unit_id": p["unit_id"],
            "hypotheses": p.get("hyps") or [],
        }
        for p in chunk
    ]


def _run_polish_batched(
    *,
    turns: list[Turn],
    out: dict[int, str],
    audits: list[dict],
    hotwords: list[str],
    llm_judge,
    cfg: PipelineConfig,
    hyp_by_turn: dict[int, list[Hypothesis]] | None = None,
) -> tuple[dict[int, str], list[dict]]:
    meeting = _meeting_draft(turns, out)
    batch_size = max(1, int(getattr(cfg, "pass_a_batch_size", 1) or 1))
    hyp_by_turn = hyp_by_turn or {}
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
                "turn": turn,
                "text": text,
                "neighbors": _cap_neighbors(meeting, i, cfg),
                "hyps": hyp_by_turn.get(i, []),
                "unit_id": f"polish_t{i}",
                "last_err": None,
            }
        )

    def _accept_or_defer(chunk: list[dict], raws: list, *, attempt: int) -> list[dict]:
        deferred: list[dict] = []
        for p, raw in zip(chunk, raws):
            accepted, err = _validate_polish_raw(raw)
            if accepted is None:
                p["last_err"] = err
                p["retries"] = attempt
                deferred.append(p)
                continue
            _apply_polish_accepted(
                out=out,
                audits=audits,
                turn=p["turn"],
                i=p["i"],
                text=p["text"],
                accepted=accepted,
                batched=True,
            )
        return deferred

    pending = list(prepared)
    while pending:
        chunk = pending[:batch_size]
        pending = pending[batch_size:]
        still = _accept_or_defer(
            chunk,
            llm_judge.polish_many(_jobs_for(chunk, hotwords), max_workers=batch_size),
            attempt=1,
        )
        for attempt in range(2, cfg.llm_max_retries + 2):
            if not still:
                break
            still = _accept_or_defer(
                still,
                llm_judge.polish_many(_jobs_for(still, hotwords), max_workers=batch_size),
                attempt=attempt,
            )
        for p in still:
            audits.append(
                {
                    "turn_index": p["i"],
                    "pass": "polish",
                    "path": "llm",
                    "fallback": True,
                    "batched": True,
                    "last_error": p.get("last_err"),
                }
            )
    return out, audits


def run_polish(
    turns: list[Turn],
    texts: dict[int, str],
    *,
    llm_judge=None,
    hotwords: list[str] | None = None,
    config: PipelineConfig | None = None,
    hyp_by_turn: dict[int, list[Hypothesis]] | None = None,
) -> tuple[dict[int, str], list[dict]]:
    """
    Display polish + hyp/context/world recovery on phonetic-final ASR.

    Pass A/B span-local and pinyin caps do not apply. Validated span edits
    are applied; judgment.text is untrusted when edits are empty.
    """
    cfg = config or PipelineConfig()
    hotwords = hotwords or []
    hyp_by_turn = hyp_by_turn or {}
    out = dict(texts)
    audits: list[dict] = []

    if llm_judge is None or not hasattr(llm_judge, "polish"):
        return out, audits

    batch_size = max(1, int(getattr(cfg, "pass_a_batch_size", 1) or 1))
    if batch_size > 1 and hasattr(llm_judge, "polish_many"):
        return _run_polish_batched(
            turns=turns,
            out=out,
            audits=audits,
            hotwords=hotwords,
            llm_judge=llm_judge,
            cfg=cfg,
            hyp_by_turn=hyp_by_turn,
        )

    meeting = _meeting_draft(turns, out)
    for i, turn in enumerate(turns):
        text = out.get(i, turn.text)
        if not text:
            continue
        if turn.duration + 1e-9 < cfg.min_asr_seconds:
            continue

        neighbors = _cap_neighbors(meeting, i, cfg)
        unit_id = f"polish_t{i}"
        accepted = None
        last_err = None
        for _attempt in range(cfg.llm_max_retries + 1):
            raw, err = _try_polish(
                llm_judge,
                text=text,
                neighbors=neighbors,
                hotwords=hotwords,
                turn_index=i,
                unit_id=unit_id,
                hypotheses=hyp_by_turn.get(i, []),
            )
            if raw is None:
                last_err = err
                continue
            accepted = raw
            break

        if accepted is None:
            audits.append(
                {
                    "turn_index": i,
                    "pass": "polish",
                    "path": "llm",
                    "fallback": True,
                    "last_error": last_err,
                }
            )
            continue

        changed = _apply_polish_accepted(
            out=out,
            audits=audits,
            turn=turn,
            i=i,
            text=text,
            accepted=accepted,
            batched=False,
        )
        if changed:
            meeting = _meeting_draft(turns, out)

    return out, audits
