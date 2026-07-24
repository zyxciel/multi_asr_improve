from __future__ import annotations

from stage2_asr.pinyin_util import pinyin_edit_distance, to_pinyin
from stage2_asr.types import Hypothesis


class MockLlmJudge:
    """Deterministic mock judge for Tier A/B/C + retry demos."""

    def __init__(self):
        self._calls: dict[str, int] = {}

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
        self._calls[unit_id] = self._calls.get(unit_id, 0) + 1
        if unit_id.endswith("_invalid") and self._calls[unit_id] == 1:
            return {"text": "x"}

        by_model = {h.model: h for h in hypotheses}
        if (overlap or heavy_overlap) and "moss" in by_model:
            base_model = "moss"
        elif "moss" in by_model:
            base_model = "moss"
        else:
            base_model = next(iter(by_model), "moss")
        base = by_model.get(base_model) or (hypotheses[0] if hypotheses else None)
        text = base.text if base else ""
        edits: list[dict] = []

        # Prefer selecting a hyp that already has the corrected form (Tier A)
        for h in hypotheses:
            if "采用" in h.text and any("产用" in x.text for x in hypotheses):
                return {
                    "text": h.text,
                    "base_model": h.model,
                    "edits": [],
                    "overlap": bool(overlap or heavy_overlap),
                }

        for h in hypotheses:
            if "产用" in h.text:
                text = h.text.replace("产用", "采用")
                edits.append(
                    {
                        "span_asr": "产用",
                        "span_out": "采用",
                        "tier": "C",
                        "pinyin_asr": to_pinyin("产用"),
                        "pinyin_out": to_pinyin("采用"),
                        "anchor": "hyp",
                    }
                )
                base_model = h.model
                break
            if "单方接" in h.text:
                text = h.text.replace("单方接", "单框架")
                if pinyin_edit_distance("单方接", "单框架") <= 2:
                    edits.append(
                        {
                            "span_asr": "单方接",
                            "span_out": "单框架",
                            "tier": "C",
                            "pinyin_asr": to_pinyin("单方接"),
                            "pinyin_out": to_pinyin("单框架"),
                            "anchor": "hotword" if hotwords else "neighbor_draft",
                        }
                    )
                    base_model = h.model
                break
            if "帐号" in h.text:
                text = h.text.replace("帐号", "账号")
                edits.append(
                    {
                        "span_asr": "帐号",
                        "span_out": "账号",
                        "tier": "B",
                        "pinyin_asr": to_pinyin("帐号"),
                        "pinyin_out": to_pinyin("账号"),
                        "anchor": "hyp",
                    }
                )
                base_model = h.model
                break

        return {
            "text": text,
            "base_model": base_model,
            "edits": edits,
            "overlap": bool(overlap or heavy_overlap),
        }
