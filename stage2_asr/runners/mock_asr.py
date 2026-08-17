from __future__ import annotations

import json
from pathlib import Path

from stage2_asr.types import AsrStatus, AsrUnit, Hypothesis, Turn


class MockAsrRunner:
    """Fixture-backed ASR. Emits moss/qwen/firered hyps; FireRed includes lid + punc meta."""

    name = "mock_asr"

    def __init__(self, fixture_path: Path | None = None, fixture: dict | None = None):
        if fixture is not None:
            self._data = fixture
        elif fixture_path is not None and fixture_path.exists():
            self._data = json.loads(fixture_path.read_text(encoding="utf-8"))
        else:
            self._data = {"by_unit_id": {}, "default": {}}

    def _moss_text(self, unit: AsrUnit, turns: list[Turn]) -> tuple[str, bool]:
        texts = []
        for i in unit.turn_indices:
            if 0 <= i < len(turns):
                t = turns[i]
                if (t.text or "").strip():
                    texts.append(t.text)
        merged = len(texts) > 1
        if not texts:
            return "", False
        return "。".join(texts), merged

    def _default_qwen_firered(self, moss_text: str) -> tuple[str, str]:
        """Inject controlled disagreements for mock repair demos."""
        if "产用" in moss_text:
            return moss_text, moss_text.replace("产用", "采用")
        if "单方接" in moss_text:
            # Keep disagreement without using 大话机 (pinyin distance > 2 to 单框架)
            return moss_text, moss_text.replace("单方接", "单方接") + "啊"
        if "帐号" in moss_text:
            return moss_text, moss_text.replace("帐号", "账号")
        return moss_text, moss_text

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
        _ = crop_path  # mock ignores audio; real runners use crop_path
        entry = self._data.get("by_unit_id", {}).get(unit.unit_id) or dict(self._data.get("default", {}))
        moss_text, moss_merged = self._moss_text(unit, turns)
        if "moss" in entry:
            moss_text = entry["moss"]

        hyps: list[Hypothesis] = []
        if moss_text and "moss" in selected:
            hyps.append(Hypothesis(model="moss", text=moss_text, meta={"moss_merged": moss_merged}))
        if moss_exclusive:
            return hyps

        qwen_default, firered_default = self._default_qwen_firered(moss_text)
        qwen = entry.get("qwen", qwen_default)
        firered = entry.get("firered", firered_default)
        lid = entry.get("lid", "zh")
        if qwen and "qwen" in selected:
            hyps.append(Hypothesis(model="qwen", text=qwen))
        if firered and "firered" in selected:
            hyps.append(
                Hypothesis(
                    model="firered",
                    text=firered,
                    lid=lid,
                    meta={"vad": False, "punc": True, "lid": True},
                )
            )
        return hyps
