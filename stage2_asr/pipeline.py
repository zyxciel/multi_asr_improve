from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np

from stage2_asr.audio_io import crop_unit_wav, load_wav_mono16k
from stage2_asr.pass_a import run_pass_a_for_unit
from stage2_asr.pass_b import run_pass_b
from stage2_asr.types import AsrStatus, AsrUnit, Hypothesis, PipelineConfig, Turn
from stage2_asr.units import build_asr_units
from stage2_asr.validate import validate_turns


def load_mode_c(path: Path) -> tuple[list[Turn], dict[str, Any]]:
    """
    Load Mode-C fusion JSON.

    Supported shapes:
    - {"meta": {...}, "turns": [{start, end, text, speaker_id, ...}, ...]}
    - [{start, end, text, speaker_id}, ...]  (plain turn list)
    """
    data = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(data, list):
        raw_doc: dict[str, Any] = {"turns": data}
        turn_dicts = data
    elif isinstance(data, dict):
        raw_doc = data
        turn_dicts = data.get("turns", [])
    else:
        raise ValueError(f"unsupported mode_c JSON root type: {type(data).__name__}")
    if not isinstance(turn_dicts, list):
        raise ValueError("mode_c turns must be a JSON array")
    turns = [Turn.from_dict(t) for t in turn_dicts if isinstance(t, dict)]
    return turns, raw_doc


def _load_audio_optional(audio_path: Path, sr: int) -> np.ndarray | None:
    """Load mono 16 kHz float32 via audio_io; None if missing/unreadable (mock-friendly)."""
    if not audio_path.exists():
        return None
    try:
        audio, _ = load_wav_mono16k(audio_path, target_sr=sr)
        return audio
    except Exception:
        return None


def _load_units(path: Path) -> list[AsrUnit] | None:
    if not path.exists():
        return None
    payload = json.loads(path.read_text(encoding="utf-8"))
    raw = payload.get("units") if isinstance(payload, dict) else payload
    if not isinstance(raw, list) or not raw:
        return None
    return [AsrUnit.from_dict(u) for u in raw]


def _save_units(path: Path, units: list[AsrUnit], skipped: list[Any]) -> None:
    path.write_text(
        json.dumps(
            {"units": [u.to_dict() for u in units], "skipped_invalid_ts": skipped},
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )


def _load_pass_stats(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}
    return data if isinstance(data, dict) else {}


def _merge_pass_stats(path: Path, updates: dict[str, Any]) -> dict[str, Any]:
    merged = _load_pass_stats(path)
    merged.update(updates)
    path.write_text(json.dumps(merged, ensure_ascii=False, indent=2), encoding="utf-8")
    return merged


def _rewrite_edits_pass_b(edits_path: Path, pass_b_audits: list[dict]) -> None:
    """Keep Pass A lines; replace any prior Pass B lines with this run's audits."""
    pass_a: list[dict] = []
    if edits_path.exists():
        for line in edits_path.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            try:
                obj = json.loads(line)
            except json.JSONDecodeError:
                continue
            if isinstance(obj, dict) and obj.get("pass") == "A":
                pass_a.append(obj)
    with edits_path.open("w", encoding="utf-8") as f:
        for a in pass_a:
            f.write(json.dumps(a, ensure_ascii=False) + "\n")
        for a in pass_b_audits:
            f.write(json.dumps(a, ensure_ascii=False) + "\n")


def _cache_path(work_dir: Path, unit_id: str, runner_name: str) -> Path:
    d = work_dir / "asr_cache"
    d.mkdir(parents=True, exist_ok=True)
    return d / f"{unit_id}__{runner_name}.json"


def _distribute_text(unit_turn_indices: list[int], text: str, turns: list[Turn]) -> dict[int, str]:
    """Map unit text back to member turns by relative duration."""
    if not unit_turn_indices:
        return {}
    if len(unit_turn_indices) == 1:
        return {unit_turn_indices[0]: text}
    durs = [max(1e-6, turns[i].duration) for i in unit_turn_indices]
    total = sum(durs)
    chars = list(text)
    n = len(chars)
    out: dict[int, str] = {}
    cursor = 0
    for k, (idx, dur) in enumerate(zip(unit_turn_indices, durs)):
        if k == len(unit_turn_indices) - 1:
            out[idx] = "".join(chars[cursor:])
        else:
            take = int(round(n * (dur / total)))
            out[idx] = "".join(chars[cursor : cursor + take])
            cursor += take
    return out


def _summarize_pass_a(audits: list[dict]) -> dict[str, Any]:
    n = len(audits)
    skipped = sum(1 for a in audits if a.get("skipped_llm"))
    retries_total = sum(int(a.get("retries") or 0) for a in audits)
    fallback = sum(1 for a in audits if a.get("fallback"))
    fallback_judge_ok = sum(1 for a in audits if a.get("fallback_judge_ok"))
    judged = n - skipped
    invalid_rate = (retries_total / max(judged, 1)) if judged else 0.0
    return {
        "n_units_judged_or_skipped": n,
        "skipped_llm": skipped,
        "retries_total": retries_total,
        "fallback_to_best_hyp": fallback,
        "fallback_judge_success": fallback_judge_ok,
        "approx_retry_rate": invalid_rate,
        "suggest_raise_temperature": invalid_rate > 0.05,
    }


def _summarize_pass_b(audits: list[dict]) -> dict[str, Any]:
    return {
        "n_audits": len(audits),
        "hotword_alias": sum(1 for a in audits if a.get("path") == "hotword_alias"),
        "llm_edits": sum(1 for a in audits if a.get("path") == "llm" and not a.get("fallback")),
        "llm_fallback": sum(1 for a in audits if a.get("path") == "llm" and a.get("fallback")),
        "fallback_judge_success": sum(1 for a in audits if a.get("fallback_judge_ok")),
        "moss_aware_reject": sum(1 for a in audits if a.get("path") == "moss_aware_reject"),
        "moss_force": sum(1 for a in audits if a.get("path") == "moss_force"),
    }


def _unit_by_id(units: list) -> dict[str, Any]:
    return {str(unit.unit_id): unit for unit in units}


def _load_hyp_records(path: Path) -> list[dict]:
    if not path.exists():
        return []
    payload = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(payload, list):
        return payload
    if isinstance(payload, dict) and isinstance(payload.get("records"), list):
        return payload["records"]
    return []


def _save_hyp_records(path: Path, records: list[dict], *, stage: str, asr_models: list[str]) -> None:
    path.write_text(
        json.dumps(
            {
                "meta": {
                    "stage": stage,
                    "asr_models": asr_models,
                    "record_count": len(records),
                },
                "records": records,
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )


def _record_index(records: list[dict]) -> dict[str, dict]:
    out: dict[str, dict] = {}
    for record in records:
        unit_id = str(record.get("unit_id", ""))
        if unit_id:
            out[unit_id] = record
    return out


def _merge_hyps(existing: list[dict], new: list[dict]) -> list[dict]:
    by_model = {str(item.get("model")): item for item in existing if item.get("model")}
    for item in new:
        model = str(item.get("model", ""))
        if model:
            by_model[model] = item
    return [by_model[key] for key in sorted(by_model)]


def _rebuild_overlap_metadata(
    records: list[dict],
    turns: list[Turn],
) -> tuple[set[int], set[int], dict[int, str]]:
    overlap_turn_indices: set[int] = set()
    heavy_overlap_turn_indices: set[int] = set()
    moss_texts: dict[int, str] = {}
    for record in records:
        turn_indices = [int(i) for i in record.get("turn_indices") or []]
        if record.get("contains_overlap"):
            overlap_turn_indices.update(turn_indices)
        if record.get("heavy_overlap"):
            heavy_overlap_turn_indices.update(turn_indices)
        moss = next(
            (
                item
                for item in (record.get("hyps") or [])
                if isinstance(item, dict) and item.get("model") == "moss"
            ),
            None,
        )
        if moss is not None:
            moss_texts.update(
                _distribute_text(
                    turn_indices,
                    str(moss.get("text", "")),
                    turns,
                )
            )
    return overlap_turn_indices, heavy_overlap_turn_indices, moss_texts


def run_pipeline(
    *,
    input_json: Path,
    audio_path: Path,
    work_dir: Path,
    asr_runner,
    llm_judge,
    config: PipelineConfig | None = None,
    hotwords: list[str] | None = None,
    fallback_judge=None,
    stage: str = "all",
    asr_models: list[str] | None = None,
) -> dict[str, Any]:
    """
    Stage modes:
    - asr: build units + run selected ASR models, persist asr_hypotheses/asr_cache only.
    - pass_a: run Pass A only from saved ASR hypotheses/cache.
    - pass_b: run Pass B only from mode_c_draft.json.
    - llm: run Pass A then Pass B from saved ASR hypotheses/cache (no ASR inference).
    - all: full pipeline in one run.
    """
    cfg = config or PipelineConfig()
    hotwords = hotwords or []
    stage = str(stage).lower()
    allowed_stages = {"all", "asr", "pass_a", "pass_b", "llm"}
    if stage not in allowed_stages:
        raise ValueError(f"unknown stage {stage!r}; expected one of {sorted(allowed_stages)}")
    normalized_models = [m.strip().lower() for m in (asr_models or ["moss", "qwen", "firered"]) if m.strip()]
    if not normalized_models:
        normalized_models = ["moss", "qwen", "firered"]
    supported_models = {"moss", "qwen", "firered"}
    bad = [m for m in normalized_models if m not in supported_models]
    if bad:
        raise ValueError(f"unsupported asr models: {bad}; expected subset of {sorted(supported_models)}")

    work_dir.mkdir(parents=True, exist_ok=True)

    raw_turns, raw_doc = load_mode_c(input_json)
    turns, skipped = validate_turns(raw_turns, cfg)
    audio = _load_audio_optional(audio_path, cfg.sample_rate)
    units_path = work_dir / "asr_units.json"

    # LLM-only stages reload persisted units so unit_id stays aligned with asr_hypotheses.
    if stage in {"pass_a", "pass_b", "llm"}:
        loaded = _load_units(units_path)
        if loaded is not None:
            units = loaded
        else:
            units = build_asr_units(turns, cfg, audio=audio, sample_rate=cfg.sample_rate)
            _save_units(units_path, units, skipped)
    else:
        units = build_asr_units(turns, cfg, audio=audio, sample_rate=cfg.sample_rate)
        _save_units(units_path, units, skipped)

    asr_runner_name = f"{getattr(asr_runner, 'name', 'asr')}__{'-'.join(sorted(normalized_models))}"
    hyp_path = work_dir / "asr_hypotheses.json"
    existing_records = _load_hyp_records(hyp_path)
    existing_by_unit = _record_index(existing_records)
    units_by_id = _unit_by_id(units)

    if stage in {"all", "asr"}:
        built_records: list[dict] = []
        for unit in units:
            if unit.skip_asr:
                prior = existing_by_unit.get(unit.unit_id, {})
                built_records.append(
                    {
                        "unit_id": unit.unit_id,
                        "turn_indices": unit.turn_indices,
                        "contains_overlap": bool(unit.contains_overlap),
                        "heavy_overlap": bool(unit.heavy_overlap),
                        "skipped": True,
                        "reason": unit.skip_reason,
                        "hyps": prior.get("hyps", []),
                    }
                )
                continue

            cache = _cache_path(work_dir, unit.unit_id, asr_runner_name)
            if cache.exists():
                cached = json.loads(cache.read_text(encoding="utf-8"))
                hyps = [
                    Hypothesis(
                        model=h["model"],
                        text=h["text"],
                        lid=h.get("lid"),
                        meta=h.get("meta") or {},
                    )
                    for h in cached
                ]
            else:
                crop_path = None
                if audio_path.exists():
                    try:
                        crop_path = str(
                            crop_unit_wav(
                                audio_path,
                                unit.start,
                                unit.end,
                                work_dir=work_dir,
                                unit_id=unit.unit_id,
                                sr=cfg.sample_rate,
                            )
                        )
                    except Exception:
                        crop_path = None
                selected_for_call = set(normalized_models)
                if unit.heavy_overlap and "moss" not in selected_for_call:
                    # Heavy overlap still needs an anchor hypothesis; moss is CPU-cheap.
                    selected_for_call = set(selected_for_call) | {"moss"}
                try:
                    hyps = asr_runner.transcribe_unit(
                        unit,
                        turns,
                        str(audio_path),
                        moss_exclusive=unit.heavy_overlap,
                        crop_path=crop_path,
                        selected_models=selected_for_call,
                    )
                except TypeError:
                    hyps = asr_runner.transcribe_unit(
                        unit,
                        turns,
                        str(audio_path),
                        moss_exclusive=unit.heavy_overlap,
                        crop_path=crop_path,
                    )
                    hyps = [h for h in hyps if h.model in selected_for_call]
                cache.write_text(
                    json.dumps([h.to_dict() for h in hyps], ensure_ascii=False, indent=2),
                    encoding="utf-8",
                )

            new_hyps = [h.to_dict() for h in hyps]
            # Accumulate hyps across staged ASR runs and re-runs of stage=all.
            prior_hyps = existing_by_unit.get(unit.unit_id, {}).get("hyps")
            merged_hyps = _merge_hyps(prior_hyps or [], new_hyps)
            built_records.append(
                {
                    "unit_id": unit.unit_id,
                    "turn_indices": unit.turn_indices,
                    "contains_overlap": bool(unit.contains_overlap),
                    "heavy_overlap": bool(unit.heavy_overlap),
                    "skipped": False,
                    "hyps": merged_hyps,
                }
            )

        _save_hyp_records(
            hyp_path,
            built_records,
            stage=stage,
            asr_models=sorted(normalized_models),
        )
        if stage == "asr":
            return {
                "stage": "asr",
                "n_turns": len(turns),
                "n_units": len(units),
                "asr_hypotheses_path": hyp_path,
                "asr_models": sorted(normalized_models),
            }

        hyp_records = built_records
    else:
        hyp_records = existing_records
        if not hyp_records:
            raise FileNotFoundError(
                f"{hyp_path} missing; run stage=asr first to prepare ASR hypotheses."
            )

    overlap_turn_indices, heavy_overlap_turn_indices, moss_texts = _rebuild_overlap_metadata(
        hyp_records, turns
    )

    if stage == "pass_b":
        draft_path = work_dir / "mode_c_draft.json"
        if not draft_path.exists():
            raise FileNotFoundError(
                f"{draft_path} missing; run stage=pass_a (or all/llm) first."
            )
        draft_doc = json.loads(draft_path.read_text(encoding="utf-8"))
        draft_turns = [Turn.from_dict(t) for t in draft_doc.get("turns", [])]
        draft_texts = {i: t.text for i, t in enumerate(draft_turns)}
        final_map, pass_b_audits = run_pass_b(
            turns,
            draft_texts,
            hotwords=hotwords,
            llm_judge=llm_judge,
            fallback_judge=fallback_judge,
            config=cfg,
            overlap_turn_indices=overlap_turn_indices,
            heavy_overlap_turn_indices=heavy_overlap_turn_indices,
            moss_texts=moss_texts,
        )
        final_turns = []
        for i, t in enumerate(turns):
            text = final_map.get(i, t.text)
            final_turns.append(
                Turn(
                    start=t.start,
                    end=t.end,
                    speaker_id=t.speaker_id,
                    text=text,
                    asr_status=AsrStatus.FINAL if text else t.asr_status,
                    source=t.source,
                    confidence=t.confidence,
                )
            )
        final_doc = {
            "meta": {**(raw_doc.get("meta") or {}), "stage": "pass_b_final"},
            "turns": [t.to_dict() for t in final_turns],
        }
        final_path = work_dir / "mode_c_asr_final.json"
        final_path.write_text(json.dumps(final_doc, ensure_ascii=False, indent=2), encoding="utf-8")
        edits_path = work_dir / "llm_edits.jsonl"
        _rewrite_edits_pass_b(edits_path, pass_b_audits)
        stats_path = work_dir / "pass_stats.json"
        pass_stats = _merge_pass_stats(stats_path, {"pass_b": _summarize_pass_b(pass_b_audits)})
        return {
            "stage": "pass_b",
            "final_path": final_path,
            "stats_path": stats_path,
            "n_turns": len(final_turns),
            "n_units": len(units),
            "pass_stats": pass_stats,
        }

    draft_texts: dict[int, str] = {i: t.text for i, t in enumerate(turns)}
    pass_a_audits: list[dict] = []
    for record in hyp_records:
        unit_id = str(record.get("unit_id", ""))
        if not unit_id or unit_id not in units_by_id:
            continue
        unit = units_by_id[unit_id]
        if bool(record.get("skipped")):
            continue
        hyps = [
            Hypothesis(
                model=h["model"],
                text=h["text"],
                lid=h.get("lid"),
                meta=h.get("meta") or {},
            )
            for h in (record.get("hyps") or [])
            if isinstance(h, dict)
        ]
        if not hyps:
            continue

        text, audit = run_pass_a_for_unit(
            unit=unit,
            turns=turns,
            hyps=hyps,
            draft_texts=draft_texts,
            llm_judge=llm_judge,
            hotwords=hotwords,
            config=cfg,
            fallback_judge=fallback_judge,
        )
        pass_a_audits.append(audit)
        for ti, piece in _distribute_text(unit.turn_indices, text, turns).items():
            draft_texts[ti] = piece

    _save_hyp_records(
        hyp_path,
        hyp_records,
        stage=stage if stage in {"pass_a", "llm"} else "all",
        asr_models=sorted(normalized_models),
    )

    draft_turns = []
    for i, t in enumerate(turns):
        nt = Turn(
            start=t.start,
            end=t.end,
            speaker_id=t.speaker_id,
            text=draft_texts.get(i, t.text),
            asr_status=AsrStatus.PROVISIONAL if draft_texts.get(i, t.text) else t.asr_status,
            source=t.source,
            confidence=t.confidence,
        )
        draft_turns.append(nt)

    draft_doc = {
        "meta": {**(raw_doc.get("meta") or {}), "stage": "pass_a_draft"},
        "turns": [t.to_dict() for t in draft_turns],
    }
    draft_path = work_dir / "mode_c_draft.json"
    draft_path.write_text(json.dumps(draft_doc, ensure_ascii=False, indent=2), encoding="utf-8")
    if stage == "pass_a":
        edits_path = work_dir / "llm_edits.jsonl"
        with edits_path.open("w", encoding="utf-8") as f:
            for a in pass_a_audits:
                f.write(json.dumps({"pass": "A", **a}, ensure_ascii=False) + "\n")
        stats_path = work_dir / "pass_stats.json"
        pass_stats = _merge_pass_stats(stats_path, {"pass_a": _summarize_pass_a(pass_a_audits)})
        return {
            "stage": "pass_a",
            "draft_path": draft_path,
            "stats_path": stats_path,
            "n_turns": len(draft_turns),
            "n_units": len(units),
            "pass_stats": pass_stats,
        }

    final_map, pass_b_audits = run_pass_b(
        turns,
        draft_texts,
        hotwords=hotwords,
        llm_judge=llm_judge,
        fallback_judge=fallback_judge,
        config=cfg,
        overlap_turn_indices=overlap_turn_indices,
        heavy_overlap_turn_indices=heavy_overlap_turn_indices,
        moss_texts=moss_texts,
    )
    final_turns = []
    for i, t in enumerate(turns):
        text = final_map.get(i, t.text)
        final_turns.append(
            Turn(
                start=t.start,
                end=t.end,
                speaker_id=t.speaker_id,
                text=text,
                asr_status=AsrStatus.FINAL if text else t.asr_status,
                source=t.source,
                confidence=t.confidence,
            )
        )

    final_doc = {
        "meta": {**(raw_doc.get("meta") or {}), "stage": "pass_b_final"},
        "turns": [t.to_dict() for t in final_turns],
    }
    final_path = work_dir / "mode_c_asr_final.json"
    final_path.write_text(json.dumps(final_doc, ensure_ascii=False, indent=2), encoding="utf-8")

    edits_path = work_dir / "llm_edits.jsonl"
    with edits_path.open("w", encoding="utf-8") as f:
        for a in pass_a_audits:
            f.write(json.dumps({"pass": "A", **a}, ensure_ascii=False) + "\n")
        for a in pass_b_audits:
            f.write(json.dumps(a, ensure_ascii=False) + "\n")

    pass_stats = {
        "pass_a": _summarize_pass_a(pass_a_audits),
        "pass_b": _summarize_pass_b(pass_b_audits),
    }
    stats_path = work_dir / "pass_stats.json"
    stats_path.write_text(json.dumps(pass_stats, ensure_ascii=False, indent=2), encoding="utf-8")

    return {
        "stage": stage,
        "final_path": final_path,
        "draft_path": draft_path,
        "stats_path": stats_path,
        "n_turns": len(final_turns),
        "n_units": len(units),
        "pass_stats": pass_stats,
    }
