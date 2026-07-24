from __future__ import annotations

"""CER / cpCER evaluation helpers (no model weights)."""

from stage2_asr.agreement import char_error_rate, normalize_for_cer
from stage2_asr.pinyin_util import to_pinyin


def cer(ref: str, hyp: str) -> float:
    return char_error_rate(ref, hyp)


def cp_cer(ref: str, hyp: str) -> float:
    """Character/pinyin CER: average of char CER and pinyin-token CER."""
    c = char_error_rate(ref, hyp)
    pr = to_pinyin(normalize_for_cer(ref)).split()
    ph = to_pinyin(normalize_for_cer(hyp)).split()
    if not pr and not ph:
        p = 0.0
    elif not pr:
        p = 1.0
    else:
        # reuse char_error_rate on joined tokens with spaces stripped via list levenshtein
        from stage2_asr.pinyin_util import _levenshtein

        p = _levenshtein(pr, ph) / max(1, len(pr))
    return 0.5 * (c + p)


def corpus_cer(pairs: list[tuple[str, str]]) -> float:
    if not pairs:
        return 0.0
    total_err = 0.0
    total_len = 0
    for ref, hyp in pairs:
        r = list(normalize_for_cer(ref))
        if not r:
            continue
        total_err += char_error_rate(ref, hyp) * len(r)
        total_len += len(r)
    return total_err / total_len if total_len else 0.0
