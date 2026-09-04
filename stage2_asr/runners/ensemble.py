from __future__ import annotations

"""Combine MOSS provisional text with Qwen + FireRed runners."""

from stage2_asr.runners.compat import transcribe_unit_compat
from stage2_asr.text_map import join_turn_texts
from stage2_asr.types import AsrUnit, Hypothesis, Turn


class EnsembleAsrRunner:
    name = "ensemble"

    def __init__(self, qwen_runner, firered_runner):
        self.qwen_runner = qwen_runner
        self.firered_runner = firered_runner

    def _moss(self, unit: AsrUnit, turns: list[Turn]) -> Hypothesis | None:
        texts = []
        for i in unit.turn_indices:
            if 0 <= i < len(turns):
                t = turns[i]
                if (t.text or "").strip():
                    texts.append(t.text)
        joined = join_turn_texts(texts)
        if not joined:
            return None
        return Hypothesis(model="moss", text=joined, meta={"moss_merged": len(texts) > 1})

    def transcribe_unit(
        self,
        unit: AsrUnit,
        turns: list[Turn],
        audio_path: str,
        *,
        moss_exclusive: bool = False,
        crop_path: str | None = None,
        selected_models: set[str] | None = None,
    ) -> list[Hypothesis]:
        selected = selected_models or {"moss", "qwen", "firered"}
        hyps: list[Hypothesis] = []
        moss = self._moss(unit, turns)
        if moss and "moss" in selected:
            hyps.append(moss)
        if moss_exclusive:
            return hyps
        # Prefer calling with crop_path if runners accept it
        if "qwen" in selected:
            hyps.extend(
                transcribe_unit_compat(
                    self.qwen_runner,
                    unit,
                    turns,
                    audio_path,
                    moss_exclusive=False,
                    crop_path=crop_path,
                )
            )
        if "firered" in selected:
            hyps.extend(
                transcribe_unit_compat(
                    self.firered_runner,
                    unit,
                    turns,
                    audio_path,
                    moss_exclusive=False,
                    crop_path=crop_path,
                )
            )
        return hyps
