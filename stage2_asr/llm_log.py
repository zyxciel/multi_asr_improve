"""Append-only JSONL logger for LLM inference (Pass A/B)."""

from __future__ import annotations

import json
import threading
import time
from pathlib import Path
from typing import Any


class LlmInferLogger:
    """Thread-safe JSONL writer for LLM request/response traces."""

    def __init__(self, path: Path):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()
        self._fh = self.path.open("a", encoding="utf-8")

    def log(self, event: dict[str, Any]) -> None:
        row = {"ts": time.time(), **event}
        line = json.dumps(row, ensure_ascii=False)
        with self._lock:
            self._fh.write(line + "\n")
            self._fh.flush()

    def close(self) -> None:
        with self._lock:
            if not self._fh.closed:
                self._fh.close()

    def __enter__(self) -> "LlmInferLogger":
        return self

    def __exit__(self, *args) -> None:
        self.close()
