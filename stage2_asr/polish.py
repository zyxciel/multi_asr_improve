from __future__ import annotations

import re
from typing import Any

from stage2_asr.neighbors import cap_neighbors, meeting_draft
from stage2_asr.text_map import distribute_unit_text
from stage2_asr.types import Hypothesis, PipelineConfig, Turn

ALLOWED_KINDS = frozenset({"punc", "entity", "codeswitch"})
ALLOWED_ANCHORS = frozenset({"hyp", "neighbor_draft", "meeting_draft", "hotword"})
# CN→CN entity repairs may grow/shrink by 1–2 characters (爱情→娃娃亲).
CJK_CHAR_SLACK = 2

_CJK_RE = re.compile(r"[\u4e00-\u9fff]")
_LATIN_RE = re.compile(r"[A-Za-z]")
_DIGIT_RE = re.compile(r"\d")


def _is_word_char(ch: str) -> bool:
    return ("\u4e00" <= ch <= "\u9fff") or (ch.isascii() and ch.isalnum())


def _core(s: str) -> str:
    return "".join(ch for ch in s if _is_word_char(ch))


def _cjk_count(s: str) -> int:
    return len(_CJK_RE.findall(s))


def _latin_count(s: str) -> int:
    return len(_LATIN_RE.findall(s))


def _digits(s: str) -> list[str]:
    return _DIGIT_RE.findall(s)


def _latin_letters(s: str) -> str:
    return "".join(ch.lower() for ch in s if ch.isascii() and ch.isalpha())


def _is_cjk_only(s: str) -> bool:
    core = _core(s)
    return bool(core) and _cjk_count(core) == len(core)


def _is_latin_only(s: str) -> bool:
    core = _core(s)
    return bool(core) and _cjk_count(core) == 0 and _latin_count(core) > 0


def _is_mixed(s: str) -> bool:
    return _cjk_count(s) > 0 and _latin_count(s) > 0


def _codeswitch_scripts(asr: str, out: str) -> bool:
    """CJK ↔ Latin, or CJK ↔ mixed CN–EN (温度的问题 → Windows产品)."""
    a_cjk, a_lat, a_mix = _is_cjk_only(asr), _is_latin_only(asr), _is_mixed(asr)
    o_cjk, o_lat, o_mix = _is_cjk_only(out), _is_latin_only(out), _is_mixed(out)
    return bool(
        (a_cjk and (o_lat or o_mix))
        or (a_lat and (o_cjk or o_mix))
        or (a_mix and (o_cjk or o_lat or o_mix))
    )


def _same_latin_identity(asr: str, out: str) -> bool:
    a = _latin_letters(asr)
    b = _latin_letters(out)
    return bool(a) and a == b and _cjk_count(asr) == 0 and _cjk_count(out) == 0


def _is_repeat_collapse(asr: str, out: str) -> bool:
    if not out or len(out) >= len(asr):
        return False
    return len(asr) % len(out) == 0 and asr == out * (len(asr) // len(out))


def hyp_text_blob(hypotheses) -> str:
    parts: list[str] = []
    for h in hypotheses or []:
        if h is None:
            continue
        if isinstance(h, dict):
            parts.append(str(h.get("text") or ""))
            meta = h.get("meta") if isinstance(h.get("meta"), dict) else {}
            parts.append(str(meta.get("unit_text") or ""))
        else:
            parts.append(str(getattr(h, "text", "") or ""))
            meta = getattr(h, "meta", None) or {}
            if isinstance(meta, dict):
                parts.append(str(meta.get("unit_text") or ""))
    return " ".join(parts)


def neighbor_text_blob(neighbors) -> str:
    if not neighbors:
        return ""
    if isinstance(neighbors, str):
        return neighbors
    parts: list[str] = []
    for row in neighbors:
        if isinstance(row, dict):
            parts.append(str(row.get("text") or ""))
        else:
            parts.append(str(row))
    return " ".join(parts)


def _hotword_blob(hotwords) -> str:
    return " ".join(str(h) for h in (hotwords or []))


def _span_out_in_source(
    span_out: str,
    anchor: str,
    *,
    hyp_blob: str,
    neighbor_blob: str,
    hotword_blob: str,
) -> bool:
    needle = span_out.strip()
    if not needle:
        return False
    if anchor == "hyp":
        return needle in hyp_blob or needle.lower() in hyp_blob.lower()
    if anchor in {"neighbor_draft", "meeting_draft"}:
        return needle in neighbor_blob or needle.lower() in neighbor_blob.lower()
    if anchor == "hotword":
        return needle in hotword_blob or needle.lower() in hotword_blob.lower()
    return False


def validate_polish_edits(
    edits: list,
    *,
    text: str,
    hypotheses: list | None = None,
    neighbors: list | None = None,
    hotwords: list[str] | None = None,
) -> tuple[bool, str | None]:
    """Reject undisciplined polish edits. Empty edits are schema-ok (applied later)."""
    if not isinstance(edits, list):
        return False, "edits must be list"
    hyp_blob = hyp_text_blob(hypotheses)
    neighbor_blob = neighbor_text_blob(neighbors)
    hw_blob = _hotword_blob(hotwords)

    for e in edits:
        if not isinstance(e, dict):
            return False, "edit must be dict"
        kind = str(e.get("kind", "")).lower()
        if kind not in ALLOWED_KINDS:
            return False, f"bad kind {kind!r}"
        span_asr = str(e.get("span_asr", ""))
        span_out = str(e.get("span_out", ""))
        if span_asr == span_out:
            continue
        if span_asr == "":
            return False, "insertions forbidden"
        if span_asr not in text:
            return False, "span_asr not in text"
        if _digits(span_asr) != _digits(span_out):
            return False, "number rewrite forbidden"
        if _is_repeat_collapse(span_asr, span_out):
            return False, "repeat collapse forbidden"
        cjk_pair = _is_cjk_only(span_asr) and _is_cjk_only(span_out)
        if cjk_pair:
            delta = abs(len(_core(span_asr)) - len(_core(span_out)))
            if delta > CJK_CHAR_SLACK:
                return False, "cjk substitution exceeds ±2 character slack"
        else:
            if span_out.startswith(span_asr) and _core(span_out[len(span_asr) :]):
                return False, "added content forbidden"
            if span_asr.startswith(span_out) and _core(span_asr[len(span_out) :]):
                return False, "deleted content forbidden"
        if kind == "punc":
            if _core(span_asr) != _core(span_out):
                return False, "punc cannot change words"
            continue
        if not cjk_pair:
            if _is_latin_only(span_asr) and _is_latin_only(span_out):
                if not _same_latin_identity(span_asr, span_out):
                    return False, "latin rewrite must keep the same letters"
            elif not _codeswitch_scripts(span_asr, span_out):
                return False, "unsupported script mix"
        if _same_latin_identity(span_asr, span_out):
            continue
        evidence = str(e.get("evidence") or "").strip()
        if not evidence:
            return False, "missing evidence"
        anchor = str(e.get("anchor") or "")
        if anchor not in ALLOWED_ANCHORS:
            return False, f"bad/missing anchor {anchor!r}"
        if not _span_out_in_source(
            span_out,
            anchor,
            hyp_blob=hyp_blob,
            neighbor_blob=neighbor_blob,
            hotword_blob=hw_blob,
        ):
            return False, f"span_out not found in {anchor} evidence"
    return True, None


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
        if raw.get("evidence"):
            item["evidence"] = str(raw["evidence"])
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


def hyps_by_merged_from_records(
    records: list[dict],
    member_indices: list[list[int]],
) -> dict[int, list[Hypothesis]]:
    """Attach unit hyps to merged rows whose members intersect the unit."""
    out: dict[int, list[Hypothesis]] = {}
    for i, members in enumerate(member_indices or []):
        mset = {int(x) for x in members}
        hyps: list[Hypothesis] = []
        for record in records or []:
            if not isinstance(record, dict):
                continue
            indices = [int(x) for x in (record.get("turn_indices") or [])]
            if not mset.intersection(indices):
                continue
            unit_id = str(record.get("unit_id") or "")
            for raw in record.get("hyps") or []:
                if not isinstance(raw, dict):
                    continue
                unit_text = str(raw.get("text") or "")
                hyps.append(
                    Hypothesis(
                        model=str(raw.get("model") or ""),
                        text=unit_text,
                        lid=raw.get("lid"),
                        meta={"unit_id": unit_id, "unit_text": unit_text},
                    )
                )
        if hyps:
            out[i] = hyps
    return out


def _validate_polish_raw(
    raw: Any,
    *,
    text: str = "",
    hypotheses: list | None = None,
    neighbors: list | None = None,
    hotwords: list[str] | None = None,
) -> tuple[dict | None, str | None]:
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
    ok, err = validate_polish_edits(
        edits,
        text=text,
        hypotheses=hypotheses,
        neighbors=neighbors,
        hotwords=hotwords,
    )
    if not ok:
        return None, err
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
    return _validate_polish_raw(
        raw,
        text=text,
        hypotheses=hypotheses,
        neighbors=neighbors,
        hotwords=hotwords,
    )


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
        if loc.get("evidence"):
            audits[-1]["evidence"] = loc["evidence"]
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
    meeting = meeting_draft(turns, out)
    batch_size = max(1, int(getattr(cfg, "polish_batch_size", 1) or 1))
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
                "neighbors": cap_neighbors(meeting, i, cfg),
                "hyps": hyp_by_turn.get(i, []),
                "unit_id": f"polish_t{i}",
                "last_err": None,
            }
        )

    def _accept_or_defer(chunk: list[dict], raws: list, *, attempt: int) -> list[dict]:
        deferred: list[dict] = []
        for p, raw in zip(chunk, raws):
            accepted, err = _validate_polish_raw(
                raw,
                text=p["text"],
                hypotheses=p.get("hyps") or [],
                neighbors=p["neighbors"],
                hotwords=hotwords,
            )
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
    """Minimal evidenced polish on the phonetic-final ASR text.

    Pass A/B span-local and pinyin caps do not apply. Validated span edits
    only; judgment.text is untrusted when edits are empty. ITN is disabled.
    """
    cfg = config or PipelineConfig()
    hotwords = hotwords or []
    hyp_by_turn = hyp_by_turn or {}
    out = dict(texts)
    audits: list[dict] = []

    if llm_judge is None or not hasattr(llm_judge, "polish"):
        return out, audits

    batch_size = max(1, int(getattr(cfg, "polish_batch_size", 1) or 1))
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

    meeting = meeting_draft(turns, out)
    for i, turn in enumerate(turns):
        text = out.get(i, turn.text)
        if not text:
            continue
        if turn.duration + 1e-9 < cfg.min_asr_seconds:
            continue

        neighbors = cap_neighbors(meeting, i, cfg)
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
            meeting = meeting_draft(turns, out)

    return out, audits
