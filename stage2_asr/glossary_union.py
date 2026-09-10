"""Corpus-level glossary union: merge per-sample publish glossaries."""

from __future__ import annotations

import json
import unicodedata
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

from stage2_asr.publish import load_glossary

_CJK_RANGES = (
    (0x3400, 0x4DBF),
    (0x4E00, 0x9FFF),
    (0xF900, 0xFAFF),
)


def _has_cjk(text: str) -> bool:
    for ch in text:
        o = ord(ch)
        if any(lo <= o <= hi for lo, hi in _CJK_RANGES):
            return True
    return False


def surface_key(surface: str) -> str:
    """NFKC + strip; Latin-only keys are case-folded; CJK stays exact."""
    s = unicodedata.normalize("NFKC", str(surface or "")).strip()
    if not s:
        return ""
    if _has_cjk(s):
        return s
    return s.casefold()


def _item_surface(item: Any) -> str:
    if isinstance(item, dict):
        return str(item.get("surface") or "").strip()
    if isinstance(item, str):
        return item.strip()
    return ""


def _as_items(raw: Any) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    if not isinstance(raw, list):
        return out
    for item in raw:
        if isinstance(item, str):
            surface = item.strip()
            if surface:
                out.append({"surface": surface})
            continue
        if not isinstance(item, dict):
            continue
        surface = _item_surface(item)
        if not surface:
            continue
        out.append(item)
    return out


def _pick_surface(originals: Counter[str], *, latin: bool) -> str:
    items = list(originals.items())
    if not items:
        return ""
    if latin:
        items.sort(key=lambda kv: (-kv[1], -sum(c.isupper() for c in kv[0]), kv[0]))
    else:
        items.sort(key=lambda kv: (-kv[1], kv[0]))
    return items[0][0]


def _load_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError, UnicodeDecodeError):
        return {}
    return data if isinstance(data, dict) else {}


def _load_turns(sample_dir: Path) -> list[dict[str, Any]]:
    data = _load_json(sample_dir / "mode_c_published.json")
    turns = data.get("turns")
    return turns if isinstance(turns, list) else []


def _load_units(sample_dir: Path) -> list[dict[str, Any]]:
    data = _load_json(sample_dir / "asr_units.json")
    units = data.get("units")
    return [u for u in units if isinstance(u, dict)] if isinstance(units, list) else []


def _load_hyps_by_unit(sample_dir: Path) -> dict[str, list[dict[str, str]]]:
    data = _load_json(sample_dir / "asr_hypotheses.json")
    records = data.get("records")
    if not isinstance(records, list):
        return {}
    out: dict[str, list[dict[str, str]]] = {}
    for rec in records:
        if not isinstance(rec, dict):
            continue
        unit_id = str(rec.get("unit_id") or "")
        if not unit_id:
            continue
        hyps: list[dict[str, str]] = []
        for h in rec.get("hyps") or []:
            if not isinstance(h, dict):
                continue
            text = str(h.get("text") or "")
            model = str(h.get("model") or "")
            if model or text:
                hyps.append({"model": model, "text": text})
        out[unit_id] = hyps
    return out


def _find_spans(text: str, surface: str) -> list[tuple[int, int]]:
    if not text or not surface:
        return []
    if _has_cjk(surface):
        hay, needle = text, surface
    else:
        hay, needle = text.casefold(), surface.casefold()
    out: list[tuple[int, int]] = []
    start = 0
    step = max(1, len(needle))
    while True:
        i = hay.find(needle, start)
        if i < 0:
            break
        out.append((i, i + len(surface)))
        start = i + step
    return out


def _context(text: str, start: int, end: int, context_chars: int) -> str:
    lo = max(0, start - context_chars)
    hi = min(len(text), end + context_chars)
    return text[lo:hi]


def _overlap(a0: float, a1: float, b0: float, b1: float) -> float:
    return min(a1, b1) - max(a0, b0)


def _best_unit(turn: dict[str, Any], units: list[dict[str, Any]]) -> dict[str, Any] | None:
    try:
        t0 = float(turn.get("start"))
        t1 = float(turn.get("end"))
    except (TypeError, ValueError):
        return None
    best: dict[str, Any] | None = None
    best_ov = 0.0
    for unit in units:
        try:
            u0 = float(unit.get("start"))
            u1 = float(unit.get("end"))
        except (TypeError, ValueError):
            continue
        ov = _overlap(t0, t1, u0, u1)
        if ov > best_ov:
            best_ov = ov
            best = unit
    return best


def _sample_id(sample_dir: Path, work_root: Path) -> str:
    rel = sample_dir.resolve().relative_to(work_root.resolve())
    posix = rel.as_posix()
    return work_root.name if posix == "." else posix


def _crop_rel(
    sample_dir: Path, work_root: Path, unit_id: str | None
) -> str | None:
    if not unit_id:
        return None
    crop = sample_dir / "crops" / f"{unit_id}.wav"
    if not crop.is_file():
        return None
    return crop.resolve().relative_to(work_root.resolve()).as_posix()


def _mentions_in_turns(turns: list[dict[str, Any]], surface: str) -> int:
    n = 0
    for turn in turns:
        text = str(turn.get("text") or "") if isinstance(turn, dict) else ""
        n += len(_find_spans(text, surface))
    return n


def _occurrences_for_surface(
    *,
    surface: str,
    sample_id: str,
    sample_dir: Path,
    work_root: Path,
    turns: list[dict[str, Any]],
    units: list[dict[str, Any]],
    hyps_by_unit: dict[str, list[dict[str, str]]],
    context_chars: int,
) -> list[dict[str, Any]]:
    hits: list[dict[str, Any]] = []
    for ti, turn in enumerate(turns):
        if not isinstance(turn, dict):
            continue
        text = str(turn.get("text") or "")
        spans = _find_spans(text, surface)
        if not spans:
            continue
        unit = _best_unit(turn, units)
        unit_id = str(unit.get("unit_id") or "") if unit else ""
        try:
            start = float(turn.get("start"))
        except (TypeError, ValueError):
            start = float("nan")
        try:
            end = float(turn.get("end"))
        except (TypeError, ValueError):
            end = float("nan")
        for cs, ce in spans:
            hits.append(
                {
                    "sample_id": sample_id,
                    "turn_index": ti,
                    "start": start,
                    "end": end,
                    "speaker_id": str(turn.get("speaker_id") or "?"),
                    "char_start": cs,
                    "char_end": ce,
                    "context": _context(text, cs, ce, context_chars),
                    "unit_id": unit_id or None,
                    "crop": _crop_rel(sample_dir, work_root, unit_id or None),
                    "hyps": list(hyps_by_unit.get(unit_id, [])),
                }
            )
    return hits


def _new_bucket() -> dict[str, Any]:
    return {
        "originals": Counter(),
        "sample_ids": [],
        "aliases": set(),
        "kinds": Counter(),
        "latex": None,
        "scores": [],
        "seen": set(),
    }


def _ingest(
    bucket: dict[str, Any],
    item: dict[str, Any],
    *,
    sample_id: str,
    as_term: bool,
) -> None:
    surface = _item_surface(item)
    if not surface:
        return
    bucket["originals"][surface] += 1
    if sample_id not in bucket["seen"]:
        bucket["sample_ids"].append(sample_id)
        bucket["seen"].add(sample_id)
    if as_term:
        for alias in item.get("aliases") or []:
            a = str(alias).strip()
            if a:
                bucket["aliases"].add(a)
        kind = str(item.get("kind") or "other").strip() or "other"
        bucket["kinds"][kind] += 1
        latex = item.get("latex")
        if bucket["latex"] in (None, "") and latex not in (None, ""):
            bucket["latex"] = latex
    score = item.get("score")
    if isinstance(score, (int, float)):
        bucket["scores"].append(float(score))


def _row_from_bucket(
    bucket: dict[str, Any],
    *,
    latin: bool,
    n_mentions: int,
    extra: dict[str, Any] | None = None,
) -> dict[str, Any]:
    surface = _pick_surface(bucket["originals"], latin=latin)
    row: dict[str, Any] = {
        "surface": surface,
        "n_docs": len(bucket["sample_ids"]),
        "n_mentions": n_mentions,
        "sample_ids": list(bucket["sample_ids"]),
    }
    if bucket["kinds"]:
        row["kind"] = bucket["kinds"].most_common(1)[0][0]
        row["aliases"] = sorted(bucket["aliases"])
        row["latex"] = bucket["latex"]
    if bucket["scores"]:
        row["score_max"] = max(bucket["scores"])
        row["score_mean"] = round(sum(bucket["scores"]) / len(bucket["scores"]), 6)
    if extra:
        row.update(extra)
    return row


def _sort_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return sorted(
        rows,
        key=lambda r: (-int(r.get("n_docs") or 0), -int(r.get("n_mentions") or 0), str(r.get("surface") or "")),
    )


def discover_glossary_samples(work_root: Path) -> list[Path]:
    root = Path(work_root)
    if not root.is_dir():
        return []
    return sorted(p.parent for p in root.rglob("glossary.json") if p.is_file())


def union_corpus_glossary(
    work_root: Path,
    *,
    context_chars: int = 80,
) -> dict[str, Any]:
    """Union per-sample glossary.json under work_root; attach rare_word occurrences."""
    root = Path(work_root)
    context_chars = max(0, int(context_chars))
    term_buckets: dict[str, dict[str, Any]] = defaultdict(_new_bucket)
    kw_buckets: dict[str, dict[str, Any]] = defaultdict(_new_bucket)
    rare_buckets: dict[str, dict[str, Any]] = defaultdict(_new_bucket)
    sample_payloads: list[dict[str, Any]] = []

    sample_dirs = discover_glossary_samples(root)
    for sample_dir in sample_dirs:
        sid = _sample_id(sample_dir, root)
        gloss = load_glossary(sample_dir / "glossary.json")
        turns = _load_turns(sample_dir)
        units = _load_units(sample_dir)
        hyps_by_unit = _load_hyps_by_unit(sample_dir)
        sample_payloads.append(
            {
                "sample_id": sid,
                "sample_dir": sample_dir,
                "turns": turns,
                "units": units,
                "hyps_by_unit": hyps_by_unit,
            }
        )
        for item in _as_items(gloss.get("terms")):
            key = surface_key(_item_surface(item))
            if key:
                _ingest(term_buckets[key], item, sample_id=sid, as_term=True)
        for item in _as_items(gloss.get("keywords")):
            key = surface_key(_item_surface(item))
            if key:
                _ingest(kw_buckets[key], item, sample_id=sid, as_term=False)
        for item in _as_items(gloss.get("rare_words")):
            key = surface_key(_item_surface(item))
            if key:
                _ingest(rare_buckets[key], item, sample_id=sid, as_term=False)

    turns_by_sample = {s["sample_id"]: s["turns"] for s in sample_payloads}

    def mentions(bucket: dict[str, Any], key: str) -> int:
        surface = _pick_surface(bucket["originals"], latin=not _has_cjk(key))
        n = 0
        for sid in bucket["sample_ids"]:
            n += _mentions_in_turns(turns_by_sample.get(sid, []), surface)
        return n

    terms: list[dict[str, Any]] = []
    for key, bucket in term_buckets.items():
        terms.append(
            _row_from_bucket(
                bucket,
                latin=not _has_cjk(key),
                n_mentions=mentions(bucket, key),
            )
        )
    keywords: list[dict[str, Any]] = []
    for key, bucket in kw_buckets.items():
        keywords.append(
            _row_from_bucket(
                bucket,
                latin=not _has_cjk(key),
                n_mentions=mentions(bucket, key),
            )
        )

    payload_by_id = {s["sample_id"]: s for s in sample_payloads}
    rare_words: list[dict[str, Any]] = []
    for key, bucket in rare_buckets.items():
        latin = not _has_cjk(key)
        surface = _pick_surface(bucket["originals"], latin=latin)
        occ: list[dict[str, Any]] = []
        n_mentions = 0
        for sid in bucket["sample_ids"]:
            sample = payload_by_id[sid]
            hits = _occurrences_for_surface(
                surface=surface,
                sample_id=sid,
                sample_dir=sample["sample_dir"],
                work_root=root,
                turns=sample["turns"],
                units=sample["units"],
                hyps_by_unit=sample["hyps_by_unit"],
                context_chars=context_chars,
            )
            occ.extend(hits)
            n_mentions += len(hits)
        rare_words.append(
            _row_from_bucket(
                bucket,
                latin=latin,
                n_mentions=n_mentions,
                extra={"occurrences": occ},
            )
        )

    return {
        "meta": {
            "n_samples": len(sample_dirs),
            "work_root": str(root),
            "context_chars": context_chars,
        },
        "terms": _sort_rows(terms),
        "keywords": _sort_rows(keywords),
        "rare_words": _sort_rows(rare_words),
    }


def to_seed_glossary(corpus: dict[str, Any]) -> dict[str, Any]:
    """Strip analysis fields so the payload can be passed to --glossary."""
    terms: list[dict[str, Any]] = []
    for t in corpus.get("terms") or []:
        if not isinstance(t, dict) or not str(t.get("surface") or "").strip():
            continue
        terms.append(
            {
                "surface": str(t["surface"]).strip(),
                "aliases": list(t.get("aliases") or []),
                "kind": str(t.get("kind") or "other"),
                "latex": t.get("latex"),
            }
        )
    keywords: list[dict[str, Any]] = []
    for k in corpus.get("keywords") or []:
        if not isinstance(k, dict) or not str(k.get("surface") or "").strip():
            continue
        score = k.get("score_max", k.get("score", 0.0))
        try:
            score_f = float(score)
        except (TypeError, ValueError):
            score_f = 0.0
        keywords.append({"surface": str(k["surface"]).strip(), "score": score_f})
    rare: list[dict[str, Any]] = []
    for r in corpus.get("rare_words") or []:
        if not isinstance(r, dict) or not str(r.get("surface") or "").strip():
            continue
        rare.append(
            {
                "surface": str(r["surface"]).strip(),
                "count": int(r.get("n_mentions") or r.get("count") or 0),
            }
        )
    return {"terms": terms, "keywords": keywords, "rare_words": rare}


def write_corpus_glossary(corpus: dict[str, Any], out: Path) -> dict[str, Path]:
    out = Path(out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(corpus, ensure_ascii=False, indent=2), encoding="utf-8")
    seed_path = out.with_name(f"{out.stem}.seed.json")
    seed_path.write_text(
        json.dumps(to_seed_glossary(corpus), ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    return {"corpus": out, "seed": seed_path}
