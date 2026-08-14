from __future__ import annotations

from stage2_asr.pass_b import run_pass_b
from stage2_asr.types import PipelineConfig, Turn


class _RecordingBatchJudge:
    """Edits 奔至→蹦字; records judge_many sizes and neighbor snapshots."""

    name = "batch"

    def __init__(self) -> None:
        self.judge_many_sizes: list[int] = []
        self.judge_calls = 0
        self.neighbor_texts: dict[str, str] = {}

    def judge(self, **kwargs):
        self.judge_calls += 1
        return self._ok(kwargs["hypotheses"][0].text, kwargs.get("overlap", False))

    def judge_many(self, jobs, max_workers=1):
        self.judge_many_sizes.append(len(jobs))
        out = []
        for job in jobs:
            unit_id = str(job.get("unit_id", ""))
            joined = " ".join(str(r.get("text", "")) for r in job.get("neighbor_draft") or [])
            self.neighbor_texts[unit_id] = joined
            out.append(self._ok(job["hypotheses"][0].text, job.get("overlap", False)))
        return out

    @staticmethod
    def _ok(text: str, overlap: bool) -> dict:
        if "奔至" in text:
            return {
                "text": text.replace("奔至", "蹦字"),
                "base_model": "draft",
                "edits": [
                    {
                        "span_asr": "奔至",
                        "span_out": "蹦字",
                        "tier": "C",
                        "pinyin_asr": "benzhi",
                        "pinyin_out": "bengzi",
                        "anchor": "meeting_draft",
                    }
                ],
                "overlap": bool(overlap),
            }
        return {
            "text": text,
            "base_model": "draft",
            "edits": [],
            "overlap": bool(overlap),
        }


def _turns_and_draft():
    turns = [
        Turn(0.0, 1.0, "s0", "系统奔至了"),
        Turn(1.0, 2.0, "s1", "刚才说蹦字问题"),
        Turn(2.0, 3.0, "s0", "还有奔至"),
    ]
    draft = {0: "系统奔至了", 1: "刚才说蹦字问题", 2: "还有奔至"}
    return turns, draft


def test_pass_b_batch_uses_judge_many_and_snapshots_neighbors():
    turns, draft = _turns_and_draft()
    judge = _RecordingBatchJudge()
    out, audits = run_pass_b(
        turns,
        draft,
        hotwords=[],
        llm_judge=judge,
        config=PipelineConfig(pass_b_batch_size=8, llm_max_retries=0),
    )
    assert judge.judge_calls == 0
    assert judge.judge_many_sizes == [3]
    assert "蹦字" in out[0]
    assert "蹦字" in out[2]
    assert any(a.get("batched") for a in audits)
    # Snapshot: later turns still see the original Pass A "奔至", not the in-pass rewrite.
    assert "奔至" in judge.neighbor_texts["pass_b_t1"]
    assert "奔至" in judge.neighbor_texts["pass_b_t2"]


def test_pass_b_default_stays_sequential():
    turns, draft = _turns_and_draft()
    judge = _RecordingBatchJudge()
    run_pass_b(
        turns,
        draft,
        hotwords=[],
        llm_judge=judge,
        config=PipelineConfig(pass_b_batch_size=1),
    )
    assert judge.judge_calls == 3
    assert judge.judge_many_sizes == []
