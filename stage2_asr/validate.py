from __future__ import annotations

import math
from typing import Any

from stage2_asr.types import PipelineConfig, Turn


def _is_bad_number(x: Any) -> bool:
    try:
        v = float(x)
    except (TypeError, ValueError):
        return True
    return not math.isfinite(v)


def validate_turns(
    turns: list[Turn],
    config: PipelineConfig | None = None,
) -> tuple[list[Turn], list[dict[str, Any]]]:
    """Keep valid turns; log skipped invalid timestamps."""
    cfg = config or PipelineConfig()
    kept: list[Turn] = []
    skipped: list[dict[str, Any]] = []
    for i, t in enumerate(turns):
        reason = None
        if _is_bad_number(t.start) or _is_bad_number(t.end):
            reason = "nan_or_inf"
        elif float(t.end) <= float(t.start):
            reason = "end_le_start"
        elif float(t.end) - float(t.start) < cfg.min_valid_seconds:
            reason = "too_short_invalid"
        if reason is not None:
            skipped.append(
                {
                    "index": i,
                    "start": t.start,
                    "end": t.end,
                    "speaker_id": t.speaker_id,
                    "reason": reason,
                    "tag": "skipped_invalid_ts",
                }
            )
            continue
        kept.append(t)
    return kept, skipped
