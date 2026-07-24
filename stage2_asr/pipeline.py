from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np

from stage2_asr.audio_io import crop_unit_wav
from stage2_asr.pass_a import run_pass_a_for_unit
from stage2_asr.pass_b import run_pass_b
from stage2_asr.types import AsrStatus, Hypothesis, PipelineConfig, Turn
from stage2_asr.units import build_asr_units
from stage2_asr.validate import validate_turns


def load_mode_c(path: Path) -> tuple[list[Turn], dict[str, Any]]:
    data = json.loads(path.read_text(encoding="utf-8"))
    turns = [Turn.from_dict(t) for t in data.get("turns", [])]
    return turns, data


def _load_audio_optional(audio_path: Path, sr: int) -> np.ndarray | None:
    # v0 mock: do not require real wav decode; return silence if file missing
    if not audio_path.exists():
        return None
    try:
        import wave

        with wave.open(str(audio_path), "rb") as wf:
            n = wf.getnframes()
            raw = wf.readframes(n)
            audio = np.frombuffer(raw, dtype=np.int16).astype(np.float32) / 32768.0
            return audio
    except Exception:
        return None


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
    # Character allocation proportional to duration
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


def run_pipeline(
    *,
    input_json: Path,
    audio_path: Path,
    work_dir: Path,
    asr_runner,
    llm_judge,
    config: PipelineConfig | None = None,
    hotwords: list[str] | None = None,
) -> dict[str, Any]:
    cfg = config or PipelineConfig()
    hotwords = hotwords or []
    work_dir.mkdir(parents=True, exist_ok=True)

    raw_turns, raw_doc = load_mode_c(input_json)
    turns, skipped = validate_turns(raw_turns, cfg)
    audio = _load_audio_optional(audio_path, cfg.sample_rate)
    units = build_asr_units(turns, cfg, audio=audio, sample_rate=cfg.sample_rate)

    (work_dir / "asr_units.json").write_text(
        json.dumps(
            {"units": [u.to_dict() for u in units], "skipped_invalid_ts": skipped},
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )

    hyp_records: list[dict] = []
    draft_texts: dict[int, str] = {i: t.text for i, t in enumerate(turns)}
    pass_a_audits: list[dict] = []

    for unit in units:
        if unit.skip_asr:
            hyp_records.append({"unit_id": unit.unit_id, "skipped": True, "reason": unit.skip_reason, "hyps": []})
            continue

        cache = _cache_path(work_dir, unit.unit_id, getattr(asr_runner, "name", "asr"))
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
            try:
                hyps = asr_runner.transcribe_unit(
                    unit,
                    turns,
                    str(audio_path),
                    moss_exclusive=unit.heavy_overlap,
                    crop_path=crop_path,
                )
            except TypeError:
                hyps = asr_runner.transcribe_unit(
                    unit,
                    turns,
                    str(audio_path),
                    moss_exclusive=unit.heavy_overlap,
                )
            cache.write_text(
                json.dumps([h.to_dict() for h in hyps], ensure_ascii=False, indent=2),
                encoding="utf-8",
            )

        hyp_records.append(
            {
                "unit_id": unit.unit_id,
                "heavy_overlap": unit.heavy_overlap,
                "hyps": [h.to_dict() for h in hyps],
            }
        )

        text, audit = run_pass_a_for_unit(
            unit=unit,
            turns=turns,
            hyps=hyps,
            draft_texts=draft_texts,
            llm_judge=llm_judge,
            hotwords=hotwords,
            config=cfg,
        )
        pass_a_audits.append(audit)
        for ti, piece in _distribute_text(unit.turn_indices, text, turns).items():
            draft_texts[ti] = piece

    (work_dir / "asr_hypotheses.json").write_text(
        json.dumps(hyp_records, ensure_ascii=False, indent=2), encoding="utf-8"
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

    final_map, pass_b_audits = run_pass_b(turns, draft_texts, hotwords=hotwords)
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

    return {
        "final_path": final_path,
        "draft_path": draft_path,
        "n_turns": len(final_turns),
        "n_units": len(units),
    }
