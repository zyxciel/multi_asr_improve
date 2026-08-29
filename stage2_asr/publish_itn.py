from __future__ import annotations

import re

_CN_ONES = {
    "零": 0,
    "〇": 0,
    "一": 1,
    "二": 2,
    "两": 2,
    "兩": 2,
    "三": 3,
    "四": 4,
    "五": 5,
    "六": 6,
    "七": 7,
    "八": 8,
    "九": 9,
    "壹": 1,
    "贰": 2,
    "貳": 2,
    "叁": 3,
    "參": 3,
    "肆": 4,
    "伍": 5,
    "陆": 6,
    "陸": 6,
    "柒": 7,
    "捌": 8,
    "玖": 9,
}
_CN_UNITS = {
    "十": 10,
    "拾": 10,
    "百": 100,
    "佰": 100,
    "千": 1000,
    "仟": 1000,
}
_CN_BIG = {"万": 10000, "萬": 10000, "亿": 100000000, "億": 100000000}
_CN_PLACE_CHARS = set(_CN_UNITS) | set(_CN_BIG) | {"点"}
_CN_PERCENT_PREFIXES = ("百分之",)

_EN_ONES = {
    "zero": 0,
    "oh": 0,
    "one": 1,
    "two": 2,
    "three": 3,
    "four": 4,
    "five": 5,
    "six": 6,
    "seven": 7,
    "eight": 8,
    "nine": 9,
}
_EN_TEENS = {
    "ten": 10,
    "eleven": 11,
    "twelve": 12,
    "thirteen": 13,
    "fourteen": 14,
    "fifteen": 15,
    "sixteen": 16,
    "seventeen": 17,
    "eighteen": 18,
    "nineteen": 19,
}
_EN_TENS = {
    "twenty": 20,
    "thirty": 30,
    "forty": 40,
    "fifty": 50,
    "sixty": 60,
    "seventy": 70,
    "eighty": 80,
    "ninety": 90,
}
_EN_SCALES = {"hundred": 100, "thousand": 1000, "million": 1_000_000}
_EN_PLACE_WORDS = set(_EN_SCALES) | {"percent"}

_ARABIC_RE = re.compile(r"^\d+(?:\.\d+)?%?$")


def _norm(s: str) -> str:
    return (s or "").strip().replace("％", "%")


def _en_tokens(s: str) -> list[str]:
    return [t.lower() for t in re.split(r"[-\s]+", _norm(s)) if t]


def _serial_cn_digits(s: str) -> str | None:
    chars = [ch for ch in _norm(s) if not ch.isspace()]
    if len(chars) < 2:
        return None
    if any(ch in _CN_PLACE_CHARS for ch in chars):
        return None
    digits: list[str] = []
    for ch in chars:
        if ch not in _CN_ONES:
            return None
        digits.append(str(_CN_ONES[ch]))
    return "".join(digits)


def _parse_cn_place(s: str) -> tuple[int, bool] | None:
    """Return (value, percent) for a Chinese place-value span, or None."""
    text = _norm(s)
    percent = False
    for prefix in _CN_PERCENT_PREFIXES:
        if text.startswith(prefix):
            percent = True
            text = text[len(prefix) :]
            break
    if "点" in text:
        return None
    if not any(ch in _CN_PLACE_CHARS or ch in _CN_BIG for ch in text):
        return None
    total = 0
    current = 0
    saw_unit = False
    for ch in text:
        if ch in _CN_ONES:
            current = _CN_ONES[ch]
        elif ch in _CN_UNITS:
            saw_unit = True
            unit = _CN_UNITS[ch]
            if current == 0:
                current = 1
            total += current * unit
            current = 0
        elif ch in _CN_BIG:
            saw_unit = True
            unit = _CN_BIG[ch]
            total = (total + current) * unit
            current = 0
        elif ch.isspace():
            continue
        else:
            return None
    total += current
    if not saw_unit:
        return None
    return total, percent


def _serial_en_digits(s: str) -> str | None:
    tokens = _en_tokens(s)
    if len(tokens) < 2:
        return None
    if any(t in _EN_PLACE_WORDS or t in _EN_TEENS or t in _EN_TENS for t in tokens):
        return None
    digits: list[str] = []
    for t in tokens:
        if t not in _EN_ONES:
            return None
        digits.append(str(_EN_ONES[t]))
    return "".join(digits)


def _parse_en_place(s: str) -> tuple[int, bool] | None:
    tokens = _en_tokens(s)
    if not tokens:
        return None
    percent = False
    if tokens[-1] == "percent":
        percent = True
        tokens = tokens[:-1]
    if not any(t in _EN_SCALES for t in tokens):
        return None
    total = 0
    current = 0
    for t in tokens:
        if t in _EN_ONES:
            current += _EN_ONES[t]
        elif t in _EN_TEENS:
            current += _EN_TEENS[t]
        elif t in _EN_TENS:
            current += _EN_TENS[t]
        elif t == "hundred":
            if current == 0:
                current = 1
            current *= 100
        elif t in {"thousand", "million"}:
            if current == 0:
                current = 1
            total += current * _EN_SCALES[t]
            current = 0
        else:
            return None
    total += current
    return total, percent


def _compact_value(s: str) -> tuple[str, int | None, bool] | None:
    """Arabic compact form: digit string, optional int value, percent flag."""
    text = _norm(s).replace(",", "")
    if not _ARABIC_RE.match(text):
        return None
    percent = text.endswith("%")
    core = text[:-1] if percent else text
    if "." in core:
        return core, None, percent
    return core, int(core), percent


def itn_edit_allowed(span_asr: str, span_out: str) -> tuple[bool, str | None]:
    """Allow compact ITN only when serial/place-value meaning is unchanged."""
    src = _norm(span_asr)
    dst = _norm(span_out)
    if not src or not dst:
        return False, "empty itn span"
    if src == dst:
        return True, None

    compact = _compact_value(dst)
    if compact is None:
        return False, "itn output must be compact arabic"

    dst_digits, dst_int, dst_pct = compact
    src_arabic = _compact_value(src)
    if src_arabic is not None:
        return False, "tts-style tn forbidden"

    serial = _serial_cn_digits(src) or _serial_en_digits(src)
    if serial is not None:
        if dst_pct or dst_int is None:
            return False, "serial itn must be the same digit string"
        if dst_digits != serial:
            return False, "serial digit order changed"
        return True, None

    place = _parse_cn_place(src) or _parse_en_place(src)
    if place is not None:
        val, pct = place
        if dst_int is None:
            return False, "place-value itn must be an integer compact form"
        if val != dst_int or pct != dst_pct:
            return False, "place-value numeric meaning changed"
        return True, None

    return False, "ambiguous or unsupported number span"
