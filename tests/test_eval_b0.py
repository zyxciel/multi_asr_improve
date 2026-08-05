from __future__ import annotations

import json
from pathlib import Path

from stage2_asr.eval_b0 import evaluate_b0


FIXTURES = Path(__file__).parent / "fixtures"


def test_evaluate_b0_identical_is_zero():
    hyp = FIXTURES / "mode_c.json"
    result = evaluate_b0(hyp_path=hyp, ref_path=hyp)
    assert result["eval"] == "B0"
    assert result["n_turns"] >= 1
    assert result["corpus_cer"] == 0.0
    assert result["mean_cer"] == 0.0


def test_evaluate_b0_detects_errors(tmp_path: Path):
    ref = {
        "turns": [
            {"start": 0, "end": 1, "speaker_id": "s0", "text": "你好世界"},
            {"start": 1, "end": 2, "speaker_id": "s1", "text": "单框架"},
        ]
    }
    hyp = {
        "turns": [
            {"start": 0, "end": 1, "speaker_id": "s0", "text": "你好世界"},
            {"start": 1, "end": 2, "speaker_id": "s1", "text": "单方接"},
        ]
    }
    ref_path = tmp_path / "ref.json"
    hyp_path = tmp_path / "hyp.json"
    ref_path.write_text(json.dumps(ref, ensure_ascii=False), encoding="utf-8")
    hyp_path.write_text(json.dumps(hyp, ensure_ascii=False), encoding="utf-8")
    result = evaluate_b0(hyp_path=hyp_path, ref_path=ref_path)
    assert result["corpus_cer"] > 0.0
    assert result["per_turn"][1]["cer"] > 0.0
