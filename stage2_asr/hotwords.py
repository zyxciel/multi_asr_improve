"""Load hotword lists for Pass A/B (JSON array or plaintext one-per-line)."""

from __future__ import annotations

import json
from pathlib import Path


def load_hotwords(path: str | Path | None) -> list[str]:
    """
    Load hotwords from:
      - JSON array: ["单框架", "账号|帐号"]
      - JSON object: {"hotwords": [...]}
      - Plaintext: one term per line (docs/hotwords.txt)

    Alias form ``canon|alt1|alt2`` is preserved as a single string for Pass B.
    Blank lines and exact duplicates are dropped (order preserved).
    """
    if not path:
        return []
    p = Path(path)
    text = p.read_text(encoding="utf-8")
    stripped = text.strip()
    if not stripped:
        return []

    items: list[str]
    if stripped[0] in "[{":
        payload = json.loads(stripped)
        if isinstance(payload, list):
            items = [str(x).strip() for x in payload]
        elif isinstance(payload, dict):
            raw = payload.get("hotwords", payload.get("words", []))
            if not isinstance(raw, list):
                raise ValueError(f"hotwords JSON object must contain a list field: {p}")
            items = [str(x).strip() for x in raw]
        else:
            raise ValueError(f"unsupported hotwords JSON type in {p}")
    else:
        items = [line.strip() for line in text.splitlines()]

    out: list[str] = []
    seen: set[str] = set()
    for item in items:
        if not item or item.startswith("#"):
            continue
        if item in seen:
            continue
        seen.add(item)
        out.append(item)
    return out
