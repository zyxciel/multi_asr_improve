"""Backoff helpers for LLM judge retries."""

from __future__ import annotations

import time


def sleep_before_retry(retry_index: int, backoff_s: float) -> None:
    """Sleep before a retry. retry_index 0 is the first attempt (no sleep)."""
    if retry_index <= 0 or backoff_s <= 0:
        return
    time.sleep(float(backoff_s) * (2 ** (retry_index - 1)))
