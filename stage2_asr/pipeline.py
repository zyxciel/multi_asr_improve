from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

import numpy as np

from stage2_asr.audio_io import crop_unit_wav, load_wav_mono16k
from stage2_asr.llm_log import LlmInferLogger
from stage2_asr.pass_a import run_pass_a_batch, run_pass_a_for_unit
from stage2_asr.pass_b import run_pass_b
from stage2_asr.polish import hyps_by_turn_from_records, hyps_by_unit_from_records, run_polish
from stage2_asr.text_map import distribute_unit_text, join_turn_texts, merged_turns_from_units
from stage2_asr.types import AsrStatus, AsrUnit, Hypothesis, PipelineConfig, Turn
from stage2_asr.units import build_asr_units
from stage2_asr.validate import validate_turns


def _log(msg: str) -> None:
    """Progress to stderr so stdout stays machine-readable JSON."""
    print(msg, file=sys.stderr, flush=True)


def _attach_llm_logger(judge, logger: LlmInferLogger | None) -> None:
    if judge is None or logger is None:
        return
    if hasattr(judge, "log_fn"):
        judge.log_fn = logger.log



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


def _write_merged_mode_c(
    path: Path,
    *,
    units: list[AsrUnit],
    turns: list[Turn],
    texts: dict[int, str],
    raw_doc: dict[str, Any],
    stage: str,
) -> list[Turn]:
    merged = merged_turns_from_units(turns, texts, units)
    rows: list[dict[str, Any]] = []
    for t, unit in zip(merged, units):
        row = t.to_dict()
        row["unit_id"] = unit.unit_id
        row["turn_indices"] = unit.turn_indices
        rows.append(row)
    doc = {
        "meta": {
            **(raw_doc.get("meta") or {}),
            "stage": stage,
            "grid": "asr_units",
            "n_source_turns": len(turns),
        },
        "turns": rows,
    }
    path.write_text(json.dumps(doc, ensure_ascii=False, indent=2), encoding="utf-8")
    return merged


def _cache_path(work_dir: Path, unit_id: str, runner_name: str) -> Path:
    d = work_dir / "asr_cache"
    d.mkdir(parents=True, exist_ok=True)
    return d / f"{unit_id}__{runner_name}.json"


def _distribute_text(unit_turn_indices: list[int], text: str, turns: list[Turn]) -> dict[int, str]:
    """Map unit text back to member turns (joiner split, else relative duration)."""
    return distribute_unit_text(unit_turn_indices, text, turns)


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
        "n_batched": sum(1 for a in audits if a.get("batched")),
    }


def _summarize_polish(audits: list[dict]) -> dict[str, Any]:
    kinds: dict[str, int] = {}
    anchors: dict[str, int] = {}
    for a in audits:
        if a.get("path") == "llm" and a.get("kind"):
            k = str(a["kind"])
            kinds[k] = kinds.get(k, 0) + 1
        if a.get("path") == "llm" and a.get("anchor"):
            an = str(a["anchor"])
            anchors[an] = anchors.get(an, 0) + 1
    return {
        "n_audits": len(audits),
        "llm_edits": sum(1 for a in audits if a.get("path") == "llm" and not a.get("fallback")),
        "empty_edits_reject": sum(1 for a in audits if a.get("path") == "empty_edits_reject"),
        "fallback": sum(1 for a in audits if a.get("fallback")),
        "by_kind": kinds,
        "by_anchor": anchors,
        "n_batched": sum(1 for a in audits if a.get("batched")),
    }


def _rewrite_edits_keep_other_passes(
    edits_path: Path, pass_name: str, new_audits: list[dict]
) -> None:
    """Keep lines from other passes; replace this pass's lines."""
    kept: list[dict] = []
    if edits_path.exists():
        for line in edits_path.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            try:
                obj = json.loads(line)
            except json.JSONDecodeError:
                continue
            if isinstance(obj, dict) and obj.get("pass") != pass_name:
                kept.append(obj)
    with edits_path.open("w", encoding="utf-8") as f:
        for a in kept:
            f.write(json.dumps(a, ensure_ascii=False) + "\n")
        for a in new_audits:
            row = a if isinstance(a, dict) and a.get("pass") == pass_name else {"pass": pass_name, **a}
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def _persist_polish(
    *,
    turns: list[Turn],
    texts: dict[int, str],
    raw_doc: dict[str, Any],
    work_dir: Path,
    llm_judge,
    hotwords: list[str],
    cfg: PipelineConfig,
    n_units: int,
    llm_log_path: Path | None,
    hyp_records: list[dict] | None = None,
    units: list[AsrUnit] | None = None,
    grid: str | None = None,
) -> dict[str, Any]:
    batch_note = max(1, int(getattr(cfg, "polish_batch_size", 1) or 1))
    _log(
        f"[polish] start: {len(turns)} turns "
        f"batch_size={batch_note} work_dir={work_dir}"
    )
    if units:
        hyp_by_turn = hyps_by_unit_from_records(hyp_records or [], units)
    else:
        hyp_by_turn = hyps_by_turn_from_records(hyp_records or [], turns)
    polished_map, audits = run_polish(
        turns,
        texts,
        llm_judge=llm_judge,
        hotwords=hotwords,
        config=cfg,
        hyp_by_turn=hyp_by_turn,
    )
    _log(f"[polish] done: {len(audits)} audits")
    polished_turns = []
    for i, t in enumerate(turns):
        text = polished_map.get(i, t.text)
        polished_turns.append(
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
    meta = {**(raw_doc.get("meta") or {}), "stage": "polish"}
    if grid:
        meta["grid"] = grid
    if units:
        rows = []
        for t, unit in zip(polished_turns, units):
            row = t.to_dict()
            row["unit_id"] = unit.unit_id
            row["turn_indices"] = unit.turn_indices
            rows.append(row)
        turn_rows = rows
    else:
        turn_rows = [t.to_dict() for t in polished_turns]
    polished_doc = {"meta": meta, "turns": turn_rows}
    polished_path = work_dir / "mode_c_polished.json"
    polished_path.write_text(
        json.dumps(polished_doc, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    _rewrite_edits_keep_other_passes(work_dir / "llm_edits.jsonl", "polish", audits)
    stats_path = work_dir / "pass_stats.json"
    pass_stats = _merge_pass_stats(stats_path, {"polish": _summarize_polish(audits)})
    _log(f"[polish] wrote {polished_path}")
    return {
        "polished_path": polished_path,
        "stats_path": stats_path,
        "llm_log_path": llm_log_path,
        "n_turns": len(polished_turns),
        "n_units": n_units,
        "pass_stats": pass_stats,
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


def _moss_hypothesis_from_turns(unit: AsrUnit, turns: list[Turn]) -> Hypothesis | None:
    texts: list[str] = []
    for i in unit.turn_indices:
        if 0 <= i < len(turns) and (turns[i].text or "").strip():
            texts.append(turns[i].text)
    joined = join_turn_texts(texts)
    if not joined:
        return None
    return Hypothesis(
        model="moss",
        text=joined,
        meta={"moss_merged": len(texts) > 1},
    )


def _ensure_moss_hyp(
    hyps: list[Hypothesis],
    unit: AsrUnit,
    turns: list[Turn],
    *,
    selected: set[str],
) -> list[Hypothesis]:
    """Guarantee a moss hyp from Mode-C text when moss is requested (no GPU)."""
    if "moss" not in selected:
        return hyps
    if any(h.model == "moss" and (h.text or "").strip() for h in hyps):
        return hyps
    moss = _moss_hypothesis_from_turns(unit, turns)
    if moss is None:
        return hyps
    return [h for h in hyps if h.model != "moss"] + [moss]


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
    - polish: merge same-speaker ASR units from mode_c_asr_final.json, then polish; writes mode_c_polished.json on the unit grid.
    - all: full pipeline in one run (ASR + Pass A/B + polish).
    """
    cfg = config or PipelineConfig()
    hotwords = hotwords or []
    stage = str(stage).lower()
    allowed_stages = {"all", "asr", "pass_a", "pass_b", "llm", "polish"}
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

    llm_logger: LlmInferLogger | None = None
    llm_log_path: Path | None = None
    if stage in {"all", "pass_a", "pass_b", "llm", "polish"}:
        llm_log_path = work_dir / "llm_infer.jsonl"
        llm_logger = LlmInferLogger(llm_log_path)
        _attach_llm_logger(llm_judge, llm_logger)
        _attach_llm_logger(fallback_judge, llm_logger)
        _log(f"[llm] infer log -> {llm_log_path}")

    try:
        return _run_pipeline_body(
            input_json=input_json,
            audio_path=audio_path,
            work_dir=work_dir,
            asr_runner=asr_runner,
            llm_judge=llm_judge,
            cfg=cfg,
            hotwords=hotwords,
            fallback_judge=fallback_judge,
            stage=stage,
            normalized_models=normalized_models,
            llm_log_path=llm_log_path,
        )
    finally:
        if llm_logger is not None:
            llm_logger.close()


def _run_pipeline_body(
    *,
    input_json: Path,
    audio_path: Path,
    work_dir: Path,
    asr_runner,
    llm_judge,
    cfg: PipelineConfig,
    hotwords: list[str],
    fallback_judge,
    stage: str,
    normalized_models: list[str],
    llm_log_path: Path | None,
) -> dict[str, Any]:
    if stage == "polish":
        final_path = work_dir / "mode_c_asr_final.json"
        if not final_path.exists():
            raise FileNotFoundError(
                f"{final_path} missing; run stage=pass_b (or all/llm) first."
            )
        final_doc = json.loads(final_path.read_text(encoding="utf-8"))
        final_turns = [Turn.from_dict(t) for t in final_doc.get("turns", [])]
        texts = {i: t.text for i, t in enumerate(final_turns)}
        loaded = _load_units(work_dir / "asr_units.json")
        n_units = len(loaded) if loaded is not None else 0
        polish_turns = final_turns
        polish_texts = texts
        polish_units = None
        polish_grid = None
        if loaded:
            polish_turns = _write_merged_mode_c(
                work_dir / "mode_c_asr_final_merged.json",
                units=loaded,
                turns=final_turns,
                texts=texts,
                raw_doc=final_doc if isinstance(final_doc, dict) else {},
                stage="pass_b_final_merged",
            )
            polish_texts = {i: t.text for i, t in enumerate(polish_turns)}
            polish_units = loaded
            polish_grid = "asr_units"
            n_units = len(loaded)
        extra = _persist_polish(
            turns=polish_turns,
            texts=polish_texts,
            raw_doc=final_doc if isinstance(final_doc, dict) else {},
            work_dir=work_dir,
            llm_judge=llm_judge,
            hotwords=hotwords,
            cfg=cfg,
            n_units=n_units,
            llm_log_path=llm_log_path,
            hyp_records=_load_hyp_records(work_dir / "asr_hypotheses.json"),
            units=polish_units,
            grid=polish_grid,
        )
        return {"stage": "polish", "final_path": final_path, **extra}

    raw_turns, raw_doc = load_mode_c(input_json)
    turns, skipped = validate_turns(raw_turns, cfg)
    audio = _load_audio_optional(audio_path, cfg.sample_rate)
    units_path = work_dir / "asr_units.json"

    # Reload persisted units so staged ASR (qwen/firered, then moss) keeps the same unit_id keys.
    reuse_units = stage in {"pass_a", "pass_b", "llm"} or (
        stage == "asr" and units_path.exists()
    )
    if reuse_units:
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
        asr_units = [u for u in units if not u.skip_asr]
        n_asr = len(asr_units)
        _log(
            f"[stage={stage}] ASR start: {n_asr} units "
            f"(models={sorted(normalized_models)}, work_dir={work_dir})"
        )
        built_records: list[dict] = []
        asr_i = 0
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

            asr_i += 1
            selected_for_call = set(normalized_models)
            if unit.heavy_overlap:
                selected_for_call.add("moss")
            cache = _cache_path(work_dir, unit.unit_id, asr_runner_name)
            hyps: list[Hypothesis] | None = None
            if cache.exists():
                cached = json.loads(cache.read_text(encoding="utf-8"))
                cached_hyps = [
                    Hypothesis(
                        model=h["model"],
                        text=h["text"],
                        lid=h.get("lid"),
                        meta=h.get("meta") or {},
                    )
                    for h in cached
                    if isinstance(h, dict)
                ]
                cached_models = {h.model for h in cached_hyps if (h.text or "").strip()}
                # Empty/stale moss caches must not block injecting Mode-C text.
                if cached_models or "moss" not in selected_for_call:
                    _log(f"[asr] {asr_i}/{n_asr} {unit.unit_id} cache-hit")
                    hyps = cached_hyps
                else:
                    _log(f"[asr] {asr_i}/{n_asr} {unit.unit_id} ignore-empty-cache")
            if hyps is None:
                crop_path = None
                crop_note = "no-crop"
                if audio_path.exists():
                    try:
                        crop_file = work_dir / "crops" / f"{unit.unit_id}.wav"
                        reused = crop_file.is_file() and crop_file.stat().st_size > 0
                        crop = crop_unit_wav(
                            audio_path,
                            unit.start,
                            unit.end,
                            work_dir=work_dir,
                            unit_id=unit.unit_id,
                            sr=cfg.sample_rate,
                            reuse_existing=True,
                        )
                        crop_path = str(crop)
                        crop_note = "reuse-crop" if reused else "new-crop"
                    except Exception as exc:  # noqa: BLE001
                        crop_path = None
                        crop_note = f"crop-failed:{exc}"
                _log(
                    f"[asr] {asr_i}/{n_asr} {unit.unit_id} transcribe "
                    f"({crop_note}) models={sorted(selected_for_call)}"
                )
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
            hyps = _ensure_moss_hyp(
                hyps, unit, turns, selected=selected_for_call
            )
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

        _log(f"[stage={stage}] ASR done: wrote {hyp_path}")
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
        _log(f"[stage={stage}] loaded {len(hyp_records)} ASR hyp records from {hyp_path}")

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
        _log(
            f"[pass_b] start: {len(draft_turns)} turns "
            f"batch_size={cfg.pass_b_batch_size} work_dir={work_dir}"
        )
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
        _log(f"[pass_b] done: {len(pass_b_audits)} audits")
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
        final_merged_path = work_dir / "mode_c_asr_final_merged.json"
        _write_merged_mode_c(
            final_merged_path,
            units=units,
            turns=final_turns,
            texts={i: t.text for i, t in enumerate(final_turns)},
            raw_doc=raw_doc,
            stage="pass_b_final_merged",
        )
        edits_path = work_dir / "llm_edits.jsonl"
        _rewrite_edits_keep_other_passes(edits_path, "B", pass_b_audits)
        stats_path = work_dir / "pass_stats.json"
        pass_stats = _merge_pass_stats(stats_path, {"pass_b": _summarize_pass_b(pass_b_audits)})
        _log(f"[pass_b] wrote {final_path}")
        return {
            "stage": "pass_b",
            "final_path": final_path,
            "final_merged_path": final_merged_path,
            "stats_path": stats_path,
            "llm_log_path": llm_log_path,
            "n_turns": len(final_turns),
            "n_units": len(units),
            "pass_stats": pass_stats,
        }

    draft_texts: dict[int, str] = {i: t.text for i, t in enumerate(turns)}
    pass_a_items: list[dict] = []
    for record in hyp_records:
        unit_id = str(record.get("unit_id", ""))
        if not unit_id or unit_id not in units_by_id:
            continue
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
        pass_a_items.append({"unit": units_by_id[unit_id], "hyps": hyps})

    n_pass_a = len(pass_a_items)
    _log(
        f"[pass_a] start: {n_pass_a} units "
        f"batch_size={cfg.pass_a_batch_size} work_dir={work_dir}"
    )
    pass_a_results = run_pass_a_batch(
        items=pass_a_items,
        turns=turns,
        draft_texts=draft_texts,
        llm_judge=llm_judge,
        hotwords=hotwords,
        config=cfg,
        fallback_judge=fallback_judge,
    )
    pass_a_audits = [audit for _, audit in pass_a_results]
    for ai, (item, (text, audit)) in enumerate(zip(pass_a_items, pass_a_results), start=1):
        _log(
            f"[pass_a] {ai}/{n_pass_a} {item['unit'].unit_id} "
            f"skipped={bool(audit.get('skipped_llm'))} fallback={bool(audit.get('fallback'))}"
        )
    _log(f"[pass_a] done: {len(pass_a_audits)} judged/skipped")

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
    draft_merged_path = work_dir / "mode_c_draft_merged.json"
    _write_merged_mode_c(
        draft_merged_path,
        units=units,
        turns=draft_turns,
        texts={i: t.text for i, t in enumerate(draft_turns)},
        raw_doc=raw_doc,
        stage="pass_a_draft_merged",
    )
    _log(f"[pass_a] wrote {draft_path}")
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
            "draft_merged_path": draft_merged_path,
            "stats_path": stats_path,
            "llm_log_path": llm_log_path,
            "n_turns": len(draft_turns),
            "n_units": len(units),
            "pass_stats": pass_stats,
        }

    _log(
        f"[pass_b] start: {len(turns)} turns "
        f"batch_size={cfg.pass_b_batch_size} work_dir={work_dir}"
    )
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
    _log(f"[pass_b] done: {len(pass_b_audits)} audits")
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
    final_merged_path = work_dir / "mode_c_asr_final_merged.json"
    merged_final = _write_merged_mode_c(
        final_merged_path,
        units=units,
        turns=final_turns,
        texts={i: t.text for i, t in enumerate(final_turns)},
        raw_doc=raw_doc,
        stage="pass_b_final_merged",
    )

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
    _log(f"[stage={stage}] done: final={final_path} draft={draft_path}")

    extra: dict[str, Any] = {}
    if stage == "all":
        extra = _persist_polish(
            turns=merged_final,
            texts={i: t.text for i, t in enumerate(merged_final)},
            raw_doc=final_doc,
            work_dir=work_dir,
            llm_judge=llm_judge,
            hotwords=hotwords,
            cfg=cfg,
            n_units=len(units),
            llm_log_path=llm_log_path,
            hyp_records=hyp_records,
            units=units,
            grid="asr_units",
        )

    return {
        "stage": stage,
        "final_path": final_path,
        "draft_path": draft_path,
        "draft_merged_path": draft_merged_path,
        "final_merged_path": final_merged_path,
        "stats_path": extra.get("stats_path", stats_path),
        "llm_log_path": llm_log_path,
        "n_turns": len(final_turns),
        "n_units": len(units),
        "pass_stats": extra.get("pass_stats", pass_stats),
        **({"polished_path": extra["polished_path"]} if extra.get("polished_path") else {}),
    }
