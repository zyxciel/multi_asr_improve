from __future__ import annotations

"""Combine MOSS provisional text with Qwen + FireRed runners."""

from stage2_asr.text_map import join_turn_texts
from stage2_asr.types import AsrStatus, AsrUnit, Hypothesis, Turn


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
        kwargs = {"crop_path": crop_path} if crop_path is not None else {}
        # Prefer calling with crop_path if runners accept it
        if "qwen" in selected:
            try:
                hyps.extend(
                    self.qwen_runner.transcribe_unit(
                        unit, turns, audio_path, moss_exclusive=False, crop_path=crop_path
                    )
                )
            except TypeError:
                hyps.extend(
                    self.qwen_runner.transcribe_unit(unit, turns, audio_path, moss_exclusive=False)
                )
        if "firered" in selected:
            try:
                hyps.extend(
                    self.firered_runner.transcribe_unit(
                        unit, turns, audio_path, moss_exclusive=False, crop_path=crop_path
                    )
                )
            except TypeError:
                hyps.extend(
                    self.firered_runner.transcribe_unit(unit, turns, audio_path, moss_exclusive=False)
                )
        return hyps
