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


def cap_hotword_list(hotwords: list[str] | None, max_chars: int) -> list[str]:
    """Keep whole terms whose JSON list stays within ``max_chars`` (prompt budget)."""
    if not hotwords or int(max_chars) <= 0:
        return []
    out: list[str] = []
    for item in hotwords:
        candidate = out + [str(item)]
        if len(json.dumps(candidate, ensure_ascii=False)) > int(max_chars):
            break
        out = candidate
    return out


def prompt_hotwords(hotwords: list[str] | None, cfg) -> list[str]:
    n = int(getattr(cfg, "hotword_prompt_chars", 4000) or 0)
    return cap_hotword_list(hotwords, n)
