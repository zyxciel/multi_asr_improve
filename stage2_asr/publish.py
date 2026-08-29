from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

from stage2_asr.publish_itn import itn_edit_allowed
from stage2_asr.types import PipelineConfig, Turn

ALLOWED_KINDS = frozenset({"filler", "repair", "punc", "latex", "itn"})
MARK_RE = re.compile(r"⟦t(\d+)(?:\|[^⟧]*)?⟧")
LATIN_RUN_RE = re.compile(r"[A-Za-z]{2,}")
TEX_CMD_RE = re.compile(r"\\([a-zA-Z]+)")
MATH_HINTS = ("平方", "squared", "下标", "subscript")
FORMULA_KINDS = frozenset({"symbol", "formula"})

FILLERS = frozenset(
    {
        "嗯",
        "啊",
        "呃",
        "那个",
        "就是说",
        "um",
        "uh",
        "ah",
        "er",
        "you know",
    }
)

_WORD_RE = re.compile(r"[\u4e00-\u9fffA-Za-z0-9]")


def _core(s: str) -> str:
    return "".join(_WORD_RE.findall(s or ""))


def _digits(s: str) -> list[str]:
    return re.findall(r"\d", s or "")


def _sanitize_speaker(speaker_id: Any) -> str:
    raw = re.sub(r"[⟦⟧|]", "", str(speaker_id or "").strip())
    raw = re.sub(r"\s+", " ", raw).strip()
    return raw[:64] or "?"


def marker_for(i: int, speaker_id: str | None = None) -> str:
    return f"⟦t{int(i)}|{_sanitize_speaker(speaker_id)}⟧"


def concat_meeting(texts: dict[int, str], speakers: dict[int, str] | None = None) -> str:
    speakers = speakers or {}
    parts: list[str] = []
    for i in sorted(texts):
        parts.append(marker_for(i, speakers.get(i)))
        parts.append(str(texts[i] or ""))
    return "".join(parts)


def split_meeting(meeting: str) -> dict[int, str]:
    out: dict[int, str] = {}
    matches = list(MARK_RE.finditer(meeting or ""))
    for idx, m in enumerate(matches):
        start = m.end()
        end = matches[idx + 1].start() if idx + 1 < len(matches) else len(meeting)
        out[int(m.group(1))] = meeting[start:end]
    return out


def latin_runs(s: str) -> set[str]:
    return {m.group(0).lower() for m in LATIN_RUN_RE.finditer(s or "")}


def _marker_spans(meeting: str) -> list[tuple[int, int]]:
    return [(m.start(), m.end()) for m in MARK_RE.finditer(meeting or "")]


def _owning_turn_span(meeting: str, start: int, end: int) -> tuple[int, int] | None:
    markers = list(MARK_RE.finditer(meeting or ""))
    for idx, m in enumerate(markers):
        t_start = m.end()
        t_end = markers[idx + 1].start() if idx + 1 < len(markers) else len(meeting)
        if t_start <= start and end <= t_end:
            return t_start, t_end
    return None


def _overlaps(start: int, end: int, occupied: list[tuple[int, int]]) -> bool:
    for a, b in occupied:
        if start < b and end > a:
            return True
    return False


def _is_latin_letter(ch: str) -> bool:
    return bool(ch) and ch.isascii() and ch.isalpha()


def _whole_latin_token(meeting: str, start: int, end: int) -> bool:
    if start > 0 and _is_latin_letter(meeting[start - 1]):
        return False
    if end < len(meeting) and _is_latin_letter(meeting[end]):
        return False
    return True


def _span_is_latin(span: str) -> bool:
    core = (span or "").strip()
    return bool(core) and all(ch.isascii() and (ch.isalpha() or ch.isspace() or ch in "'-") for ch in core)


def _is_cjk(ch: str) -> bool:
    return bool(ch) and "\u4e00" <= ch <= "\u9fff"


def _content_char(ch: str) -> bool:
    return _is_cjk(ch) or _is_latin_letter(ch) or (bool(ch) and ch.isdigit())


def _span_is_multi_cjk(span: str) -> bool:
    core = (span or "").strip()
    return len(core) >= 2 and all(_is_cjk(ch) or ch.isspace() for ch in core)


def _whole_cjk_filler(meeting: str, end: int) -> bool:
    """Reject 那个/就是说 when they prefix a following noun (以前那个温度)."""
    if end < len(meeting) and _content_char(meeting[end]):
        return False
    return True


def _filler_span_ok(meeting: str, start: int, end: int, span: str) -> bool:
    if _span_is_latin(span):
        return _whole_latin_token(meeting, start, end)
    if _span_is_multi_cjk(span):
        return _whole_cjk_filler(meeting, end)
    return True


def _locate(
    meeting: str,
    span: str,
    start_hint: Any = None,
    *,
    filler_token: bool = False,
) -> tuple[int, int] | None:
    if span == "":
        return None

    def _ok(start: int, stop: int) -> bool:
        if not filler_token:
            return True
        return _filler_span_ok(meeting, start, stop, span)

    if isinstance(start_hint, int) and start_hint >= 0:
        stop = start_hint + len(span)
        if stop <= len(meeting) and meeting[start_hint:stop] == span and _ok(start_hint, stop):
            return start_hint, stop
    start = 0
    while True:
        found = meeting.find(span, start)
        if found < 0:
            return None
        end = found + len(span)
        if _ok(found, end):
            return found, end
        start = found + 1


def _glossary_latex_cmds(terms: list) -> set[str]:
    cmds: set[str] = set()
    for term in terms or []:
        if not isinstance(term, dict):
            continue
        latex = str(term.get("latex") or "")
        cmds.update(TEX_CMD_RE.findall(latex))
    return cmds


def _glossary_formula_surfaces(terms: list) -> set[str]:
    out: set[str] = set()
    for term in terms or []:
        if not isinstance(term, dict):
            continue
        if str(term.get("kind") or "") not in FORMULA_KINDS:
            continue
        out.add(str(term.get("surface") or "").lower())
        for alias in term.get("aliases") or []:
            out.add(str(alias).lower())
    return out


def _language_break(span_asr: str, span_out: str, *, kind: str, glossary_terms: list) -> bool:
    src_lat = latin_runs(span_asr)
    dst_lat = latin_runs(span_out)
    lost = src_lat - dst_lat
    if not lost:
        return False
    if kind == "itn":
        return False
    if kind == "repair":
        return False
    if kind == "latex":
        formula = _glossary_formula_surfaces(glossary_terms)
        hints = any(h in span_asr for h in MATH_HINTS)
        if hints and all(len(x) < 2 or x in formula for x in lost):
            return False
        if lost <= formula:
            return False
        return True
    return True


def _filler_ok(span_asr: str, span_out: str, glossary_terms: list) -> bool:
    if span_out != "":
        return False
    token = (span_asr or "").strip().strip("，,。.!？?")
    extra = {
        str(t.get("surface") or "").strip()
        for t in (glossary_terms or [])
        if isinstance(t, dict) and str(t.get("kind") or "") == "filler"
    }
    return token in FILLERS or token.lower() in {x.lower() for x in FILLERS} or token in extra


def _latex_ok(span_asr: str, span_out: str, glossary_terms: list) -> tuple[bool, str | None]:
    if "$" not in span_out:
        return False, "latex span_out must contain $"
    allowed = _glossary_latex_cmds(glossary_terms)
    for cmd in TEX_CMD_RE.findall(span_out):
        if cmd not in allowed:
            return False, f"unknown tex command \\{cmd}"
    if _language_break(span_asr, span_out, kind="latex", glossary_terms=glossary_terms):
        return False, "latex cannot wrap non-formula latin"
    if not any(h in span_asr for h in MATH_HINTS) and not _glossary_formula_surfaces(
        glossary_terms
    ).intersection({span_asr.lower(), *latin_runs(span_asr)}):
        if latin_runs(span_asr) and not any(h in span_asr for h in MATH_HINTS):
            # allow if no long latin product names
            if any(len(r) >= 2 for r in latin_runs(span_asr)):
                return False, "latex on non-formula term"
    return True, None


def validate_publish_payload(raw: Any, *, meeting: str) -> tuple[bool, str | None]:
    if not isinstance(raw, dict):
        return False, "expected dict"
    edits = raw.get("edits", [])
    if not isinstance(edits, list):
        return False, "edits must be list"
    markers = _marker_spans(meeting)
    for e in edits:
        if not isinstance(e, dict):
            return False, "edit must be dict"
        if "span_asr" not in e or "span_out" not in e or "kind" not in e:
            return False, "edit missing span_asr/span_out/kind"
        kind = str(e.get("kind", "")).lower()
        if kind not in ALLOWED_KINDS:
            return False, f"bad kind {kind!r}"
        span_asr = str(e.get("span_asr", ""))
        if any(m in span_asr for m in ("⟦", "⟧")) or MARK_RE.search(span_asr):
            return False, "edit touches turn marker"
        loc = _locate(meeting, span_asr, e.get("start_char")) if span_asr else None
        if loc and _overlaps(loc[0], loc[1], markers):
            return False, "edit overlaps turn marker"
    return True, None


def filter_publish_edits(
    edits: list,
    *,
    meeting: str,
    glossary_terms: list | None = None,
) -> tuple[list[dict], str | None]:
    glossary_terms = glossary_terms or []
    kept: list[dict] = []
    occupied: list[tuple[int, int]] = []
    markers = _marker_spans(meeting)
    occupied.extend(markers)

    for raw in edits or []:
        if not isinstance(raw, dict):
            continue
        kind = str(raw.get("kind", "")).lower()
        if kind not in ALLOWED_KINDS:
            continue
        span_asr = str(raw.get("span_asr", ""))
        span_out = str(raw.get("span_out", ""))
        if kind != "filler" and span_asr == span_out:
            continue
        if kind == "filler" and span_asr == "":
            continue
        loc = _locate(
            meeting,
            span_asr,
            raw.get("start_char"),
            filler_token=kind == "filler",
        )
        if loc is None:
            continue
        start, end = loc
        if _overlaps(start, end, occupied):
            continue
        if kind == "filler":
            if not _filler_ok(span_asr, span_out, glossary_terms):
                continue
            bounds = _owning_turn_span(meeting, start, end)
            if bounds is not None:
                t0, t1 = bounds
                body = meeting[t0:t1]
                rel_s = start - t0
                rel_e = end - t0
                new_body = body[:rel_s] + span_out + body[rel_e:]
                if _core(body) and not _core(new_body):
                    continue
        elif kind == "repair":
            if span_out not in span_asr or len(span_out) >= len(span_asr):
                continue
            if _language_break(span_asr, span_out, kind="repair", glossary_terms=glossary_terms):
                continue
        elif kind == "punc":
            if _core(span_asr) != _core(span_out):
                continue
            if _digits(span_asr) != _digits(span_out):
                continue
        elif kind == "latex":
            ok, _err = _latex_ok(span_asr, span_out, glossary_terms)
            if not ok:
                continue
        elif kind == "itn":
            ok, _err = itn_edit_allowed(span_asr, span_out)
            if not ok:
                continue
        occupied.append((start, end))
        item = {
            "span_asr": span_asr,
            "span_out": span_out,
            "kind": kind,
            "start_char": start,
            "end_char": end,
        }
        kept.append(item)
    return kept, None


def apply_publish_edits(meeting: str, edits: list[dict]) -> tuple[str, list[dict]]:
    located = [e for e in edits if isinstance(e, dict) and "start_char" in e and "end_char" in e]
    if any("start_char" not in e for e in edits if isinstance(e, dict)):
        located, _ = filter_publish_edits(edits, meeting=meeting, glossary_terms=[])
    out = meeting
    for loc in sorted(located, key=lambda x: (x["start_char"], x["end_char"]), reverse=True):
        out = out[: loc["start_char"]] + loc["span_out"] + out[loc["end_char"] :]
    return out, located


def load_glossary(path: Path | None) -> dict:
    empty = {"terms": [], "keywords": [], "rare_words": []}
    if path is None or not path.exists():
        return empty
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError, UnicodeDecodeError):
        return empty
    if not isinstance(data, dict):
        return empty
    terms = data.get("terms") if isinstance(data.get("terms"), list) else []
    return {
        "terms": terms,
        "keywords": data.get("keywords") if isinstance(data.get("keywords"), list) else [],
        "rare_words": data.get("rare_words") if isinstance(data.get("rare_words"), list) else [],
    }


def merge_glossary(seed: dict, extract: dict | None) -> dict:
    seed_terms = [t for t in (seed.get("terms") or []) if isinstance(t, dict)]
    seen = {str(t.get("surface") or "") for t in seed_terms if t.get("surface")}
    merged_terms = [{**t, "source": t.get("source") or "seed"} for t in seed_terms]
    extract = extract if isinstance(extract, dict) else {}
    for t in extract.get("new_terms") or []:
        if not isinstance(t, dict):
            continue
        surface = str(t.get("surface") or "").strip()
        if not surface or surface in seen:
            continue
        seen.add(surface)
        merged_terms.append(
            {
                "surface": surface,
                "aliases": list(t.get("aliases") or []),
                "kind": str(t.get("kind") or "other"),
                "latex": t.get("latex"),
                "source": "extract",
            }
        )
    keywords = extract.get("keywords") if isinstance(extract.get("keywords"), list) else []
    rare = extract.get("rare_words") if isinstance(extract.get("rare_words"), list) else []
    return {"terms": merged_terms, "keywords": keywords, "rare_words": rare}


def _latin_ok_after(src: str, dst: str, applied: list[dict]) -> bool:
    lost = latin_runs(src) - latin_runs(dst)
    for e in applied:
        if e.get("kind") in {"itn", "latex", "filler"}:
            lost -= latin_runs(str(e.get("span_asr") or ""))
    return not lost


def run_publish(
    turns: list[Turn],
    texts: dict[int, str],
    *,
    llm_judge=None,
    hotwords: list[str] | None = None,
    glossary: dict | None = None,
    config: PipelineConfig | None = None,
) -> tuple[dict[int, str], list[dict], dict, dict | None]:
    """Display publish: span edits, extract, optional faithfulness gate."""
    cfg = config or PipelineConfig()
    hotwords = hotwords or []
    seed = glossary if isinstance(glossary, dict) else {"terms": []}
    terms = seed.get("terms") if isinstance(seed.get("terms"), list) else []
    out = dict(texts)
    speakers = {
        i: (turns[i].speaker_id if 0 <= i < len(turns) else "?") for i in out
    }
    audits: list[dict] = []
    eval_payload: dict | None = None

    if llm_judge is None or not hasattr(llm_judge, "publish"):
        return out, audits, merge_glossary(seed, None), None

    original_meeting = concat_meeting(out, speakers)
    accepted = None
    last_err = None
    for _attempt in range(int(cfg.llm_max_retries) + 1):
        try:
            raw = llm_judge.publish(
                meeting=original_meeting,
                hotwords=hotwords,
                glossary=seed,
                unit_id="publish_meeting",
            )
        except Exception as exc:  # noqa: BLE001
            last_err = str(exc)
            continue
        ok, err = validate_publish_payload(raw, meeting=original_meeting)
        if not ok:
            last_err = err
            continue
        accepted = raw
        break
    if accepted is None:
        audits.append(
            {
                "pass": "publish",
                "path": "llm",
                "fallback": True,
                "last_error": last_err,
            }
        )
        published_meeting = original_meeting
        located: list[dict] = []
    else:
        kept, _ = filter_publish_edits(
            accepted.get("edits") or [],
            meeting=original_meeting,
            glossary_terms=terms,
        )
        published_meeting, located = apply_publish_edits(original_meeting, kept)
        if not _latin_ok_after(original_meeting, published_meeting, located):
            published_meeting = original_meeting
            located = []
            audits.append({"pass": "publish", "path": "latin_invariant_revert"})
        else:
            split = split_meeting(published_meeting)
            for i in out:
                if i in split:
                    out[i] = split[i]
            for loc in located:
                audits.append({"pass": "publish", "path": "llm", **loc})

    do_eval = bool(getattr(cfg, "publish_eval", True))
    if do_eval and hasattr(llm_judge, "eval_publish"):
        eval_payload = None
        last_err = None
        for _attempt in range(int(cfg.llm_max_retries) + 1):
            try:
                eval_payload = llm_judge.eval_publish(
                    original=original_meeting,
                    published=concat_meeting(out, speakers),
                    unit_id="publish_eval",
                    enable_thinking=bool(getattr(cfg, "publish_eval_thinking", True)),
                )
            except Exception as exc:  # noqa: BLE001
                last_err = str(exc)
                eval_payload = None
                continue
            if isinstance(eval_payload, dict) and "faithful" in eval_payload:
                break
            last_err = "bad eval payload"
            eval_payload = None
        if eval_payload is None:
            audits.append(
                {
                    "pass": "publish_eval",
                    "path": "llm",
                    "fallback": True,
                    "last_error": last_err,
                }
            )
        else:
            audits.append(
                {
                    "pass": "publish_eval",
                    "path": "llm",
                    "faithful": bool(eval_payload.get("faithful")),
                }
            )
            if eval_payload.get("faithful") is False:
                out = dict(texts)
                published_meeting = original_meeting
                audits.append({"pass": "publish_eval", "path": "reverted"})

    extract_raw = None
    if hasattr(llm_judge, "extract_terms"):
        for _attempt in range(int(cfg.llm_max_retries) + 1):
            try:
                extract_raw = llm_judge.extract_terms(
                    meeting=concat_meeting(out, speakers),
                    glossary=seed,
                    unit_id="publish_extract",
                )
            except Exception:  # noqa: BLE001
                extract_raw = None
                continue
            if isinstance(extract_raw, dict):
                break
            extract_raw = None
        if not isinstance(extract_raw, dict):
            audits.append({"pass": "extract", "path": "llm", "fallback": True})
            extract_raw = None
        else:
            audits.append({"pass": "extract", "path": "llm"})

    glossary_out = merge_glossary(seed, extract_raw)
    return out, audits, glossary_out, eval_payload


def render_transcript(turns: list[Turn], texts: dict[int, str]) -> str:
    lines: list[str] = []
    for i, turn in enumerate(turns):
        text = texts.get(i, turn.text)
        lines.append(f"## Speaker {turn.speaker_id} [{turn.start:.2f}–{turn.end:.2f}]")
        lines.append(text or "")
        lines.append("")
    return "\n".join(lines).rstrip() + "\n"
