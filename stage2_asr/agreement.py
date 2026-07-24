from __future__ import annotations

import re

from stage2_asr.types import Hypothesis


_PUNCT_RE = re.compile(r"[\s\u3000，。！？、；：,.!?;:\"'“”‘’（）()【】\[\]《》<>…—\-]+")


def normalize_for_cer(text: str) -> str:
    return _PUNCT_RE.sub("", text)


def char_error_rate(ref: str, hyp: str) -> float:
    r = list(normalize_for_cer(ref))
    h = list(normalize_for_cer(hyp))
    if not r and not h:
        return 0.0
    if not r:
        return 1.0
    # Levenshtein
    prev = list(range(len(h) + 1))
    for i, rc in enumerate(r, 1):
        cur = [i]
        for j, hc in enumerate(h, 1):
            cost = 0 if rc == hc else 1
            cur.append(min(cur[j - 1] + 1, prev[j] + 1, prev[j - 1] + cost))
        prev = cur
    return prev[-1] / max(1, len(r))


def all_hyps_agree(hyps: list[Hypothesis]) -> bool:
    texts = [normalize_for_cer(h.text) for h in hyps if h.text]
    if len(texts) <= 1:
        return True
    base = texts[0]
    return all(char_error_rate(base, t) == 0.0 for t in texts[1:])


def pick_best_hyp(hyps: list[Hypothesis], prefer_moss_on_overlap: bool) -> Hypothesis | None:
    if not hyps:
        return None
    if prefer_moss_on_overlap:
        for h in hyps:
            if h.model == "moss" and h.text:
                return h
    nonempty = [h for h in hyps if h.text]
    return nonempty[0] if nonempty else hyps[0]
