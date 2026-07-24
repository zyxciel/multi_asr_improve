from __future__ import annotations

from pypinyin import Style, lazy_pinyin


def to_pinyin(text: str, tone: bool = False) -> str:
    style = Style.TONE3 if tone else Style.NORMAL
    parts = lazy_pinyin(text, style=style, errors=lambda x: [c for c in x if c.strip()])
    return " ".join(p for p in parts if p)


def pinyin_equal(a: str, b: str) -> bool:
    return to_pinyin(a) == to_pinyin(b)


def _levenshtein(a: list[str], b: list[str]) -> int:
    if not a:
        return len(b)
    if not b:
        return len(a)
    prev = list(range(len(b) + 1))
    for i, ca in enumerate(a, 1):
        cur = [i]
        for j, cb in enumerate(b, 1):
            cost = 0 if ca == cb else 1
            cur.append(min(cur[j - 1] + 1, prev[j] + 1, prev[j - 1] + cost))
        prev = cur
    return prev[-1]


def pinyin_edit_distance(a: str, b: str) -> int:
    ta = to_pinyin(a).split()
    tb = to_pinyin(b).split()
    return _levenshtein(ta, tb)
