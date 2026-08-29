from __future__ import annotations

import re

from stage2_asr.pinyin_util import pinyin_edit_distance, to_pinyin
from stage2_asr.polish import apply_polish_edits
from stage2_asr.types import Hypothesis


def _blob_from_hyps(hypotheses) -> str:
    parts: list[str] = []
    for h in hypotheses or []:
        if h is None:
            continue
        if isinstance(h, dict):
            parts.append(str(h.get("text") or ""))
            meta = h.get("meta") if isinstance(h.get("meta"), dict) else {}
            parts.append(str(meta.get("unit_text") or ""))
        else:
            parts.append(str(getattr(h, "text", "") or ""))
            meta = getattr(h, "meta", None) or {}
            if isinstance(meta, dict):
                parts.append(str(meta.get("unit_text") or ""))
    return " ".join(parts)


def _blob_from_neighbors(neighbor_draft) -> str:
    if not neighbor_draft:
        return ""
    if isinstance(neighbor_draft, str):
        return neighbor_draft
    parts: list[str] = []
    for row in neighbor_draft:
        if isinstance(row, dict):
            parts.append(str(row.get("text") or ""))
        else:
            parts.append(str(row))
    return " ".join(parts)


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

    def polish(
        self,
        *,
        text: str,
        neighbor_draft: list[dict],
        hotwords: list[str],
        turn_index: int,
        unit_id: str,
        hypotheses: list | None = None,
        **_kwargs,
    ) -> dict:
        """Deterministic polish: evidenced entity/codeswitch only. No ITN or trailing 。."""
        _ = (hotwords, turn_index, unit_id)
        original = text or ""
        edits: list[dict] = []
        hyp_blob = _blob_from_hyps(hypotheses)
        neighbor_blob = _blob_from_neighbors(neighbor_draft)
        if "温度的问题" in original and "windows产品" in hyp_blob.lower():
            edits.append(
                {
                    "span_asr": "温度的问题",
                    "span_out": "Windows产品",
                    "kind": "codeswitch",
                    "anchor": "hyp",
                    "evidence": "qwen hyp contains windows产品",
                }
            )
        elif "温度" in original and "Windows" in hyp_blob:
            edits.append(
                {
                    "span_asr": "温度",
                    "span_out": "Windows",
                    "kind": "codeswitch",
                    "anchor": "hyp",
                    "evidence": "qwen hyp contains Windows",
                }
            )
        if "张三风" in original and "张三丰" in neighbor_blob:
            edits.append(
                {
                    "span_asr": "张三风",
                    "span_out": "张三丰",
                    "kind": "entity",
                    "anchor": "neighbor_draft",
                    "evidence": "neighbor contains 张三丰",
                }
            )
        if "爱情" in original and "娃娃亲" in neighbor_blob:
            edits.append(
                {
                    "span_asr": "爱情",
                    "span_out": "娃娃亲",
                    "kind": "entity",
                    "anchor": "neighbor_draft",
                    "evidence": "neighbor contains 娃娃亲",
                }
            )
        for src, dst in (
            ("wifi", "Wi-Fi"),
            ("WIFI", "Wi-Fi"),
            ("gpu", "GPU"),
            ("Gpu", "GPU"),
        ):
            if src in original and src != dst:
                edits.append(
                    {
                        "span_asr": src,
                        "span_out": dst,
                        "kind": "codeswitch",
                        "evidence": "latin token already in the phonetic final",
                    }
                )
        new_text, _located = apply_polish_edits(original, edits)
        return {"text": new_text, "edits": edits}

    def polish_many(self, jobs: list[dict], *, max_workers: int = 8) -> list[dict]:
        _ = max_workers
        return [self.polish(**job) for job in jobs]

    def publish(
        self,
        *,
        meeting: str,
        hotwords: list | None = None,
        glossary: dict | None = None,
        unit_id: str = "",
        **_kwargs,
    ) -> dict:
        from stage2_asr.publish import FILLERS, MARK_RE

        _ = (hotwords, glossary, unit_id)
        edits: list[dict] = []
        skip = set(MARK_RE.findall(meeting or ""))
        _ = skip
        for tok in ("嗯", "啊", "那个", "就是说", "呃", "um", "uh", "ah"):
            if tok in (meeting or ""):
                edits.append({"span_asr": tok, "span_out": "", "kind": "filler"})
        if "伍柒叁" in (meeting or ""):
            edits.append({"span_asr": "伍柒叁", "span_out": "573", "kind": "itn"})
        if "x平方" in (meeting or ""):
            edits.append({"span_asr": "x平方", "span_out": "$x^{2}$", "kind": "latex"})
        if "周二不周三" in (meeting or ""):
            edits.append({"span_asr": "周二不周三", "span_out": "周三", "kind": "repair"})
        if "Tuesday no Wednesday" in (meeting or ""):
            edits.append(
                {
                    "span_asr": "Tuesday no Wednesday",
                    "span_out": "Wednesday",
                    "kind": "repair",
                }
            )
        _ = FILLERS
        return {"edits": edits}

    def publish_many(self, jobs: list[dict], *, max_workers: int = 8) -> list[dict]:
        _ = max_workers
        return [self.publish(**job) for job in jobs]

    def extract_terms(
        self,
        *,
        meeting: str,
        glossary: dict | None = None,
        unit_id: str = "",
        **_kwargs,
    ) -> dict:
        _ = (glossary, unit_id)
        keywords = []
        if re.search(r"GPU|gpu", meeting or ""):
            keywords.append({"surface": "GPU", "score": 1.0})
        new_terms = []
        if "Windows产品" in (meeting or ""):
            new_terms.append(
                {
                    "surface": "Windows产品",
                    "aliases": [],
                    "kind": "product",
                    "latex": None,
                }
            )
        return {"keywords": keywords, "rare_words": [], "new_terms": new_terms}

    def eval_publish(
        self,
        *,
        original: str,
        published: str,
        unit_id: str = "",
        enable_thinking: bool = True,
        **_kwargs,
    ) -> dict:
        from stage2_asr.publish import latin_runs

        _ = (unit_id, enable_thinking)
        faithful = True
        if "伍柒叁" in (original or "") and "五百三十七" in (published or ""):
            faithful = False
        lost = latin_runs(original) - latin_runs(published)
        if lost and not any(ch.isdigit() for ch in (published or "")):
            if lost - {"um", "uh", "ah"}:
                faithful = False
        return {
            "faithful": faithful,
            "clearer": True,
            "more_concise": True,
            "easier": True,
            "scores": {
                "faithfulness": 1.0 if faithful else 0.0,
                "clarity": 1.0,
                "concision": 1.0,
                "ease": 1.0,
            },
            "issues": [] if faithful else ["unfaithful"],
        }
