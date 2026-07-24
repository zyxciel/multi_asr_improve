from __future__ import annotations

"""
Qwen3-ASR adapter.

Upstream: https://github.com/QwenLM/Qwen3-ASR
API shape (transformers backend):

    from qwen_asr import Qwen3ASRModel
    model = Qwen3ASRModel.from_pretrained("Qwen/Qwen3-ASR-1.7B", ...)
    results = model.transcribe(audio=[path_or_tuple], language="Chinese")

Weights are NOT downloaded by this package. Pass a preloaded model or enable=True
with qwen_asr installed and model_id resolvable locally.
"""

from pathlib import Path

from stage2_asr.runners.base import UnsupportedRunnerError
from stage2_asr.types import AsrUnit, Hypothesis, Turn


class Qwen3AsrRunner:
    name = "qwen3_asr"

    def __init__(
        self,
        *,
        enabled: bool = False,
        model=None,
        model_id: str = "Qwen/Qwen3-ASR-1.7B",
        language: str | None = "Chinese",
        work_dir: Path | None = None,
    ):
        self.enabled = enabled
        self.model = model
        self.model_id = model_id
        self.language = language
        self.work_dir = work_dir

    def _ensure_model(self):
        if self.model is not None:
            return self.model
        if not self.enabled:
            raise UnsupportedRunnerError(
                "Qwen3AsrRunner disabled. Use MockAsrRunner or pass model=/enabled=True with qwen-asr installed."
            )
        try:
            from qwen_asr import Qwen3ASRModel  # type: ignore
        except ImportError as e:
            raise UnsupportedRunnerError(
                "qwen_asr not installed. See https://github.com/QwenLM/Qwen3-ASR"
            ) from e
        # from_pretrained may download weights — only when explicitly enabled
        self.model = Qwen3ASRModel.from_pretrained(self.model_id)
        return self.model

    def transcribe_unit(
        self,
        unit: AsrUnit,
        turns: list[Turn],
        audio_path: str,
        *,
        moss_exclusive: bool = False,
        crop_path: str | None = None,
    ) -> list[Hypothesis]:
        if moss_exclusive:
            return []
        model = self._ensure_model()
        path = crop_path or audio_path
        # Official API: transcribe returns list[ASRTranscription] with .text
        results = model.transcribe(audio=path, language=self.language)
        text = ""
        if results:
            r0 = results[0]
            text = getattr(r0, "text", None) or (r0.get("text") if isinstance(r0, dict) else str(r0))
        return [Hypothesis(model="qwen", text=str(text or ""), meta={"backend": "qwen3_asr", "model_id": self.model_id})]
