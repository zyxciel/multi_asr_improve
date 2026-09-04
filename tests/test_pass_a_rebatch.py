from __future__ import annotations

from stage2_asr.pass_a import run_pass_a_batch
from stage2_asr.types import AsrUnit, Hypothesis, PipelineConfig, Turn


class _RebatchJudge:
    """First judge_many fails validation; second succeeds — must stay batched."""

    name = "rebatch"

    def __init__(self) -> None:
        self.judge_many_sizes: list[int] = []
        self.judge_calls = 0
        self._many_rounds = 0

    def judge(self, **kwargs):
        self.judge_calls += 1
        return self._ok(kwargs["hypotheses"], kwargs.get("overlap", False))

    def judge_many(self, jobs, max_workers=1):
        self.judge_many_sizes.append(len(jobs))
        self._many_rounds += 1
        if self._many_rounds == 1:
            return [{"text": "bad"} for _ in jobs]
        return [
            self._ok(job["hypotheses"], job.get("overlap", False)) for job in jobs
        ]

    @staticmethod
    def _ok(hyps, overlap: bool) -> dict:
        h = hyps[0]
        return {
            "text": h.text,
            "base_model": h.model,
            "edits": [],
            "overlap": bool(overlap),
        }


def test_pass_a_retries_are_rebatched_not_serial():
    turns = [
        Turn(0.0, 1.0, "s0", "甲"),
        Turn(1.0, 2.0, "s0", "乙"),
        Turn(2.0, 3.0, "s0", "丙"),
    ]
    items = []
    for i, ch in enumerate("甲乙丙"):
        items.append(
            {
                "unit": AsrUnit(f"u{i}", float(i), float(i) + 1.0, "s0", [i]),
                "hyps": [
                    Hypothesis("moss", ch),
                    Hypothesis("qwen", ch + "改"),
                ],
            }
        )
    judge = _RebatchJudge()
    results = run_pass_a_batch(
        items=items,
        turns=turns,
        draft_texts={0: "甲", 1: "乙", 2: "丙"},
        llm_judge=judge,
        hotwords=[],
        config=PipelineConfig(pass_a_batch_size=8, llm_max_retries=2),
    )
    assert len(results) == 3
    assert all(t == ch for (t, _), ch in zip(results, "甲乙丙"))
    # First attempt + one retry, both size-3 batches; no serial judge().
    assert judge.judge_many_sizes == [3, 3]
    assert judge.judge_calls == 0
    assert all(a.get("retries", 0) == 1 for _, a in results)
