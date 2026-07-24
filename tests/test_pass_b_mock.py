from __future__ import annotations

from stage2_asr.pass_b import run_pass_b
from stage2_asr.types import Turn


def test_pass_b_hotword_alias():
    turns = [Turn(0, 1, "s0"), Turn(1, 2, "s1")]
    draft = {0: "需要单方接兼容", 1: "单框架已经上线"}
    out, audits = run_pass_b(turns, draft, hotwords=["单框架|单方接"])
    assert "单框架" in out[0]
    assert "单方接" not in out[0]
    assert audits
