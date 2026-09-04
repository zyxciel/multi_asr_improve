from __future__ import annotations

import re
from typing import Any

from stage2_asr.agreement import normalize_for_cer
from stage2_asr.llm_retry import sleep_before_retry
from stage2_asr.neighbors import cap_neighbors, meeting_draft
from stage2_asr.pinyin_util import pinyin_edit_distance
from stage2_asr.polish_cluster import (
    EntitySubset,
    build_homophone_clusters,
    cluster_allow_list,
    cluster_channel_edit,
    leftover_mentions,
    parse_partition_payload,
    revert_subset_edits,
    subset_edit_texts_unique,
)
from stage2_asr.text_map import distribute_unit_text
from stage2_asr.types import Hypothesis, PipelineConfig, Turn

ALLOWED_KINDS = frozenset({"punc", "entity", "codeswitch"})
ALLOWED_ANCHORS = frozenset(
    {"hyp", "neighbor_draft", "meeting_draft", "meeting_hyp", "hotword"}
)
# CN→CN entity repairs may grow/shrink by 1–2 characters (爱情→娃娃亲).
CJK_CHAR_SLACK = 2
PINYIN_NEAR = 2
_MEETING_HYP_PROMPT_CHARS = 4096

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


def _parse_hotword_index(hotwords: list[str] | None) -> tuple[list[str], dict[str, str]]:
    """'canon|alt1|alt2' → canons + alt→canon. Length caps from Pass B do not apply."""
    canons: list[str] = []
    alias_to_canon: dict[str, str] = {}
    for hw in hotwords or []:
        parts = [p.strip() for p in str(hw).split("|") if p.strip()]
        if not parts:
            continue
        canon = parts[0]
        canons.append(canon)
        for alt in parts[1:]:
            alias_to_canon[alt] = canon
    return canons, alias_to_canon


def _canon_of(span_out: str, canons: list[str]) -> str | None:
    needle = span_out.strip()
    if not needle:
        return None
    for c in canons:
        if needle == c or needle.lower() == c.lower():
            return c
    return None


def _lexicon_hit(span_asr: str, span_out: str, hotwords: list[str] | None) -> bool:
    canons, alias_to_canon = _parse_hotword_index(hotwords)
    canon = _canon_of(span_out, canons)
    if canon is None:
        return False
    if alias_to_canon.get(span_asr) == canon or alias_to_canon.get(span_asr.lower()) == canon:
        return True
    return pinyin_edit_distance(span_asr, canon) <= PINYIN_NEAR


def _in_blob(needle: str, blob: str) -> bool:
    if not needle or not blob:
        return False
    return needle in blob or needle.lower() in blob.lower()


def meeting_hyp_blob(hyp_by_turn: dict[int, list] | None) -> str:
    parts: list[str] = []
    for hyps in (hyp_by_turn or {}).values():
        blob = hyp_text_blob(hyps)
        if blob.strip():
            parts.append(blob)
    return " ".join(parts)


def format_meeting_hyps_prompt(
    hyp_by_turn: dict[int, list] | None,
    *,
    skip_index: int,
    this_text: str,
    char_cap: int = _MEETING_HYP_PROMPT_CHARS,
) -> str:
    """Other-turn ASR forms, unique vs this turn, capped for the polish prompt."""
    this_n = normalize_for_cer(this_text)
    seen: set[str] = set()
    lines: list[str] = []
    for i, hyps in sorted((hyp_by_turn or {}).items()):
        if int(i) == int(skip_index):
            continue
        for h in hyps or []:
            if h is None:
                continue
            if isinstance(h, dict):
                model = str(h.get("model") or "?")
                text = str(h.get("text") or "").strip()
            else:
                model = str(getattr(h, "model", "?") or "?")
                text = str(getattr(h, "text", "") or "").strip()
            if not text:
                continue
            key = normalize_for_cer(text)
            if not key or key == this_n or key in seen:
                continue
            seen.add(key)
            lines.append(f"- t{i}/{model}: {text}")
    if not lines:
        return "(none)"
    blob = "\n".join(lines)
    if len(blob) > char_cap:
        return blob[:char_cap] + "\n..."
    return blob


def _hyps_agree_with_draft(hyps: list | None, text: str) -> bool:
    nonempty: list[str] = []
    for h in hyps or []:
        if h is None:
            continue
        raw = h.get("text") if isinstance(h, dict) else getattr(h, "text", "")
        piece = str(raw or "").strip()
        if piece:
            nonempty.append(piece)
    if len(nonempty) < 2:
        return False
    draft_n = normalize_for_cer(text)
    return all(normalize_for_cer(t) == draft_n for t in nonempty)


def _span_out_in_source(
    span_out: str,
    anchor: str,
    *,
    hyp_blob: str,
    neighbor_blob: str,
    meeting_draft_blob: str,
    meeting_hyp_blob: str,
    hotword_blob: str,
) -> bool:
    needle = span_out.strip()
    if not needle:
        return False
    if anchor == "hyp":
        return _in_blob(needle, hyp_blob)
    if anchor == "neighbor_draft":
        return _in_blob(needle, neighbor_blob)
    if anchor == "meeting_draft":
        return _in_blob(needle, meeting_draft_blob)
    if anchor == "meeting_hyp":
        return _in_blob(needle, meeting_hyp_blob)
    if anchor == "hotword":
        return _in_blob(needle, hotword_blob)
    return False


def validate_polish_edits(
    edits: list,
    *,
    text: str,
    hypotheses: list | None = None,
    neighbors: list | None = None,
    hotwords: list[str] | None = None,
    meeting_hyps: list | dict | None = None,
    meeting_drafts: list | None = None,
) -> tuple[bool, str | None]:
    """Reject undisciplined polish edits. Empty edits are schema-ok (applied later)."""
    if not isinstance(edits, list):
        return False, "edits must be list"
    hyp_blob = hyp_text_blob(hypotheses)
    neighbor_blob = neighbor_text_blob(neighbors)
    meeting_hyp_src = (
        meeting_hyp_blob(meeting_hyps)
        if isinstance(meeting_hyps, dict)
        else hyp_text_blob(meeting_hyps)
    )
    meeting_draft_blob = neighbor_text_blob(meeting_drafts)
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
        lex_hit = _lexicon_hit(span_asr, span_out, hotwords)
        cjk_pair = _is_cjk_only(span_asr) and _is_cjk_only(span_out)
        if cjk_pair:
            delta = abs(len(_core(span_asr)) - len(_core(span_out)))
            if delta > CJK_CHAR_SLACK and not lex_hit:
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
        if anchor == "hotword":
            if not lex_hit:
                return False, "span_out is not a pinyin-near/aliased hotword canonical"
        elif not _span_out_in_source(
            span_out,
            anchor,
            hyp_blob=hyp_blob,
            neighbor_blob=neighbor_blob,
            meeting_draft_blob=meeting_draft_blob,
            meeting_hyp_blob=meeting_hyp_src,
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


# --- Homophone-cluster wiring (spec §2–§4) ---


def _pretty_cluster_mappings(
    subsets: list[EntitySubset], allow: dict[str, str]
) -> str:
    """Render approved mappings for the polish prompt: `张三风|涨三丰 → 涨三丰`.

    Only surfaces still present in the (conflict-filtered) allow-list are
    listed, so the prompt never grants a mapping the allow-list dropped.
    """
    lines: list[str] = []
    for sub in subsets:
        if not sub.same_entity or sub.canonical is None:
            continue
        surfaces = sorted(s for s in sub.surfaces if allow.get(s) == sub.canonical)
        if not surfaces:
            continue
        lines.append(f"{'|'.join(surfaces)} → {sub.canonical}")
    return "\n".join(lines) if lines else "(none)"


def _cluster_partition(
    llm_judge,
    hyp_records: list[dict] | None,
    cfg: PipelineConfig,
    audits: list[dict],
) -> tuple[dict[str, str], list[EntitySubset], str]:
    """Run cluster construction + one partition call per kept cluster.

    Returns (allow_list, approved_subsets, cluster_mappings_pretty).
    Skips (current behavior) when the config disables it, the judge lacks
    `partition_cluster`, or there are no hyp records. Invalid JSON / errors
    retry with the polish backoff, then yield no subsets for that cluster.
    """
    if not getattr(cfg, "polish_cluster", True):
        return {}, [], "(none)"
    if not hasattr(llm_judge, "partition_cluster"):
        return {}, [], "(none)"
    if not hyp_records:
        return {}, [], "(none)"

    clusters = build_homophone_clusters(hyp_records)
    allow: dict[str, str] = {}
    conflicts: set[str] = set()
    approved: list[EntitySubset] = []
    backoff = float(getattr(cfg, "llm_retry_backoff_s", 0.0))
    for cluster in clusters:
        raw = None
        last_err = None
        ok = False
        for attempt in range(cfg.llm_max_retries + 1):
            sleep_before_retry(attempt, backoff)
            try:
                raw = llm_judge.partition_cluster(
                    cluster=cluster, unit_id=cluster.cluster_id
                )
            except Exception as exc:  # noqa: BLE001
                raw = None
                last_err = str(exc)
                continue
            if isinstance(raw, dict):
                ok = True
                break
            last_err = f"expected dict partition payload, got {type(raw).__name__}"
            raw = None
        subsets = parse_partition_payload(raw, cluster) if ok else []
        covered: set[str] = set()
        for sub in subsets:
            covered |= set(sub.surfaces)
        row: dict[str, Any] = {
            "pass": "polish_cluster",
            "path": "partition",
            "cluster_id": cluster.cluster_id,
            "surfaces": list(cluster.surfaces),
            "subsets": [
                {
                    "surfaces": sorted(sub.surfaces),
                    "canonical": sub.canonical,
                    "same_entity": sub.same_entity,
                    "reason": sub.reason,
                }
                for sub in subsets
            ],
            "uncovered": sorted(set(cluster.surfaces) - covered),
            "tone_mismatch_pairs": [list(p) for p in cluster.tone_mismatch_pairs],
            "ok": ok,
        }
        if not ok:
            row["fallback"] = True
            row["last_error"] = last_err
        audits.append(row)

        approved.extend(
            sub for sub in subsets if sub.same_entity and sub.canonical is not None
        )
        # Merge this cluster's allow-list; a surface mapped to two different
        # canonicals (across subsets/clusters) is dropped as a conflict.
        for surface, canonical in cluster_allow_list(subsets).items():
            if surface in conflicts:
                continue
            if surface in allow and allow[surface] != canonical:
                del allow[surface]
                conflicts.add(surface)
            else:
                allow[surface] = canonical
    return allow, approved, _pretty_cluster_mappings(approved, allow)


def _cluster_subset_sweep(
    out: dict[int, str], audits: list[dict], subsets: list[EntitySubset]
) -> dict[int, str]:
    """Per approved subset: hard-check landed edits, revert that subset only
    if its cluster-channel edits disagree, then report leftover mentions."""
    applied = [
        a
        for a in audits
        if a.get("pass") == "polish"
        and a.get("path") == "llm"
        and not a.get("fallback")
        and isinstance(a.get("turn_index"), int)
        and isinstance(a.get("start_char"), int)
    ]
    for sub in subsets:
        if not sub.same_entity or sub.canonical is None:
            continue
        applied_for_s = [
            a for a in applied if str(a.get("span_asr")) in sub.surfaces
        ]
        if applied_for_s and not subset_edit_texts_unique(applied_for_s, sub.canonical):
            out = revert_subset_edits(out, applied_for_s, sub)
            audits.append(
                {
                    "pass": "polish_cluster",
                    "path": "subset_revert",
                    "surfaces": sorted(sub.surfaces),
                    "canonical": sub.canonical,
                    "reason": "cluster-channel edits landed non-unique writings",
                }
            )
        audits.extend(leftover_mentions(out, sub, applied_for_s))
    return out


def _validate_polish_raw(
    raw: Any,
    *,
    text: str = "",
    hypotheses: list | None = None,
    neighbors: list | None = None,
    hotwords: list[str] | None = None,
    meeting_hyps: list | dict | None = None,
    meeting_drafts: list | None = None,
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
        meeting_hyps=meeting_hyps,
        meeting_drafts=meeting_drafts,
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
    meeting_hyps: list | dict | None = None,
    meeting_drafts: list | None = None,
    meeting_hyps_prompt: str = "(none)",
    cluster_mappings: str = "(none)",
) -> tuple[dict | None, str | None]:
    try:
        raw = llm_judge.polish(
            text=text,
            neighbor_draft=neighbors,
            hotwords=hotwords,
            turn_index=turn_index,
            unit_id=unit_id,
            hypotheses=hypotheses or [],
            meeting_hyps=meeting_hyps_prompt,
            cluster_mappings=cluster_mappings,
        )
    except Exception as exc:  # noqa: BLE001
        return None, str(exc)
    return _validate_polish_raw(
        raw,
        text=text,
        hypotheses=hypotheses,
        neighbors=neighbors,
        hotwords=hotwords,
        meeting_hyps=meeting_hyps,
        meeting_drafts=meeting_drafts,
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
    allow: dict[str, str] | None = None,
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
        audits[-1]["cluster_channel"] = bool(cluster_channel_edit(loc, allow or {}))
    return True


def _jobs_for(
    chunk: list[dict], hotwords: list[str], cluster_mappings: str = "(none)"
) -> list[dict]:
    return [
        {
            "text": p["text"],
            "neighbor_draft": p["neighbors"],
            "hotwords": hotwords,
            "turn_index": p["i"],
            "unit_id": p["unit_id"],
            "hypotheses": p.get("hyps") or [],
            "meeting_hyps": p.get("meeting_hyps_prompt") or "(none)",
            "cluster_mappings": cluster_mappings,
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
    allow: dict[str, str] | None = None,
    cluster_mappings: str = "(none)",
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
        hyps = hyp_by_turn.get(i, [])
        hyp_prompt = format_meeting_hyps_prompt(
            hyp_by_turn, skip_index=i, this_text=text
        )
        if _hyps_agree_with_draft(hyps, text) and hyp_prompt == "(none)":
            audits.append(
                {
                    "turn_index": i,
                    "pass": "polish",
                    "path": "hyps_agree_skip",
                    "batched": True,
                }
            )
            continue
        prepared.append(
            {
                "i": i,
                "turn": turn,
                "text": text,
                "neighbors": cap_neighbors(meeting, i, cfg),
                "hyps": hyps,
                "meeting_hyps_prompt": hyp_prompt,
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
                meeting_hyps=hyp_by_turn,
                meeting_drafts=meeting,
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
                allow=allow,
            )
        return deferred

    pending = list(prepared)
    while pending:
        chunk = pending[:batch_size]
        pending = pending[batch_size:]
        still = _accept_or_defer(
            chunk,
            llm_judge.polish_many(
                _jobs_for(chunk, hotwords, cluster_mappings), max_workers=batch_size
            ),
            attempt=1,
        )
        backoff = float(getattr(cfg, "llm_retry_backoff_s", 0.0))
        for attempt in range(2, cfg.llm_max_retries + 2):
            if not still:
                break
            sleep_before_retry(attempt - 1, backoff)
            still = _accept_or_defer(
                still,
                llm_judge.polish_many(
                    _jobs_for(still, hotwords, cluster_mappings),
                    max_workers=batch_size,
                ),
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
    hyp_records: list[dict] | None = None,
) -> tuple[dict[int, str], list[dict]]:
    """Minimal evidenced polish on the phonetic-final ASR text.

    Pass A/B span-local and pinyin caps do not apply. Validated span edits
    only; judgment.text is untrusted when edits are empty. ITN is disabled.

    When `hyp_records` are given (and `cfg.polish_cluster` is on and the
    judge exposes `partition_cluster`), homophone clusters are partitioned
    first; approved entity subsets feed the prompt's cluster-mappings block,
    cluster-channel tagging, and the post-pass subset hard check / leftover
    report.
    """
    cfg = config or PipelineConfig()
    hotwords = hotwords or []
    hyp_by_turn = hyp_by_turn or {}
    out = dict(texts)
    audits: list[dict] = []

    if llm_judge is None or not hasattr(llm_judge, "polish"):
        return out, audits

    allow, approved_subsets, cluster_mappings = _cluster_partition(
        llm_judge, hyp_records, cfg, audits
    )

    batch_size = max(1, int(getattr(cfg, "polish_batch_size", 1) or 1))
    if batch_size > 1 and hasattr(llm_judge, "polish_many"):
        out, audits = _run_polish_batched(
            turns=turns,
            out=out,
            audits=audits,
            hotwords=hotwords,
            llm_judge=llm_judge,
            cfg=cfg,
            hyp_by_turn=hyp_by_turn,
            allow=allow,
            cluster_mappings=cluster_mappings,
        )
        out = _cluster_subset_sweep(out, audits, approved_subsets)
        return out, audits

    meeting = meeting_draft(turns, out)
    for i, turn in enumerate(turns):
        text = out.get(i, turn.text)
        if not text:
            continue
        if turn.duration + 1e-9 < cfg.min_asr_seconds:
            continue
        hyps = hyp_by_turn.get(i, [])
        hyp_prompt = format_meeting_hyps_prompt(
            hyp_by_turn, skip_index=i, this_text=text
        )
        if _hyps_agree_with_draft(hyps, text) and hyp_prompt == "(none)":
            audits.append(
                {
                    "turn_index": i,
                    "pass": "polish",
                    "path": "hyps_agree_skip",
                }
            )
            continue

        neighbors = cap_neighbors(meeting, i, cfg)
        unit_id = f"polish_t{i}"
        accepted = None
        last_err = None
        backoff = float(getattr(cfg, "llm_retry_backoff_s", 0.0))
        for _attempt in range(cfg.llm_max_retries + 1):
            sleep_before_retry(_attempt, backoff)
            raw, err = _try_polish(
                llm_judge,
                text=text,
                neighbors=neighbors,
                hotwords=hotwords,
                turn_index=i,
                unit_id=unit_id,
                hypotheses=hyps,
                meeting_hyps=hyp_by_turn,
                meeting_drafts=meeting,
                meeting_hyps_prompt=hyp_prompt,
                cluster_mappings=cluster_mappings,
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
            allow=allow,
        )
        if changed:
            meeting = meeting_draft(turns, out)

    out = _cluster_subset_sweep(out, audits, approved_subsets)
    return out, audits
