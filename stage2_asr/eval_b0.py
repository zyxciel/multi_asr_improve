"""Eval B0: MOSS-from-fusion baseline metrics."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from stage2_asr.eval_metrics import cer, corpus_cer, cp_cer
from stage2_asr.types import Turn


def _load_turns(path: Path) -> list[Turn]:
    data = json.loads(path.read_text(encoding="utf-8"))
    return [Turn.from_dict(t) for t in data.get("turns", [])]


def evaluate_b0(*, hyp_path: Path, ref_path: Path) -> dict[str, Any]:
    """Compare Mode-C (hyp) turn texts to a reference JSON with the same turn order."""
    hyp_turns = _load_turns(hyp_path)
    ref_turns = _load_turns(ref_path)
    n = min(len(hyp_turns), len(ref_turns))
    pairs = [(ref_turns[i].text, hyp_turns[i].text) for i in range(n)]
    per_turn: list[dict[str, Any]] = []
    for i, (ref, hyp) in enumerate(pairs):
        per_turn.append(
            {
                "turn_index": i,
                "speaker_id": hyp_turns[i].speaker_id,
                "cer": cer(ref, hyp),
                "cp_cer": cp_cer(ref, hyp),
                "ref": ref,
                "hyp": hyp,
            }
        )
    mean_cer = sum(p["cer"] for p in per_turn) / n if n else 0.0
    mean_cp = sum(p["cp_cer"] for p in per_turn) / n if n else 0.0
    return {
        "eval": "B0",
        "description": "MOSS-from-fusion baseline (Mode-C provisional vs reference)",
        "n_turns": n,
        "n_hyp_turns": len(hyp_turns),
        "n_ref_turns": len(ref_turns),
        "corpus_cer": corpus_cer(pairs),
        "mean_cer": mean_cer,
        "mean_cp_cer": mean_cp,
        "per_turn": per_turn,
    }
