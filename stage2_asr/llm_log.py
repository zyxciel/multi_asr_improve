"""Append-only JSONL logger for LLM inference (Pass A/B and polish)."""

from __future__ import annotations

import json
import threading
import time
from contextvars import ContextVar
from pathlib import Path
from typing import Any, Callable

LogFn = Callable[[dict[str, Any]], None]
_BODY_KEYS = ("user", "response", "reasoning")
_active_log_fn: ContextVar[LogFn | None] = ContextVar("stage2_llm_log_fn", default=None)


def activate_llm_log(fn: LogFn | None):
    return _active_log_fn.set(fn)


def reset_llm_log(token) -> None:
    _active_log_fn.reset(token)


def current_llm_log() -> LogFn | None:
    return _active_log_fn.get()


class LlmInferLogger:
    """Thread-safe JSONL writer for LLM request/response traces."""

    def __init__(self, path: Path, mode: str = "meta"):
        self.path = Path(path)
        mode = str(mode or "meta").lower()
        if mode not in {"full", "meta", "off"}:
            mode = "meta"
        self.mode = mode
        self._lock = threading.Lock()
        self._fh = None
        if self.mode != "off":
            self.path.parent.mkdir(parents=True, exist_ok=True)
            self._fh = self.path.open("a", encoding="utf-8")

    def log(self, event: dict[str, Any]) -> None:
        if self.mode == "off" or self._fh is None:
            return
        row = {"ts": time.time(), **event}
        if self.mode == "meta":
            row = {k: v for k, v in row.items() if k not in _BODY_KEYS}
        line = json.dumps(row, ensure_ascii=False)
        with self._lock:
            self._fh.write(line + "\n")
            self._fh.flush()

    def close(self) -> None:
        with self._lock:
            if self._fh is not None and not self._fh.closed:
                self._fh.close()

    def __enter__(self) -> "LlmInferLogger":
        return self

    def __exit__(self, *args) -> None:
        self.close()
