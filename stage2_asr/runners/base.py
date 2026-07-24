from __future__ import annotations

from typing import Protocol

from stage2_asr.types import AsrUnit, Hypothesis, Turn


class UnsupportedRunnerError(RuntimeError):
    pass


class AsrRunner(Protocol):
    name: str

    def transcribe_unit(
        self,
        unit: AsrUnit,
        turns: list[Turn],
        audio_path: str,
        *,
        moss_exclusive: bool = False,
    ) -> list[Hypothesis]:
        ...


class LlmJudge(Protocol):
    def judge(
        self,
        *,
        hypotheses: list[Hypothesis],
        neighbor_draft: list[dict],
        hotwords: list[str],
        overlap: bool,
        heavy_overlap: bool,
        unit_id: str,
    ) -> dict:
        """Return raw JSON-able dict matching the judge schema."""
        ...
