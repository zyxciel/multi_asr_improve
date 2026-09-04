"""Extract JSON judgment payloads from LLM text that may include reasoning/CoT."""

from __future__ import annotations

import json
import re
from dataclasses import dataclass

from stage2_asr.runners.base import UnsupportedRunnerError

# Common thinking / reasoning wrappers (Qwen3, etc.)
_THINK_PATTERNS = [
    re.compile(r"<think>(.*?)</think>", re.IGNORECASE | re.DOTALL),
    re.compile(r"<thinking>(.*?)</thinking>", re.IGNORECASE | re.DOTALL),
    re.compile(r"<reason>(.*?)</reason>", re.IGNORECASE | re.DOTALL),
    re.compile(r"<reasoning>(.*?)</reasoning>", re.IGNORECASE | re.DOTALL),
    # Some builds use special unicode delimiters
    re.compile(r"◁think▷(.*?)◁/think▷", re.DOTALL),
]


@dataclass
class ParsedLlmOutput:
    json_text: str
    reasoning: str | None
    answer_text: str
    raw: str


def _extract_balanced_object(text: str) -> str | None:
    """Return the first top-level JSON object substring, or None."""
    start = text.find("{")
    if start < 0:
        return None
    depth = 0
    in_str = False
    escape = False
    for i in range(start, len(text)):
        ch = text[i]
        if in_str:
            if escape:
                escape = False
            elif ch == "\\":
                escape = True
            elif ch == '"':
                in_str = False
            continue
        if ch == '"':
            in_str = True
        elif ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0:
                return text[start : i + 1]
    return None


def extract_json_and_reasoning(text: str) -> ParsedLlmOutput:
    """
    Split optional reasoning from the JSON answer.

    Preference:
    1) Content after closed </think> (or equivalent) tags
    2) Strip all think-blocks and parse remaining text
    3) Balanced {...} extraction anywhere in the raw string
    """
    raw = text or ""
    reasoning_parts: list[str] = []
    remainder = raw

    for pat in _THINK_PATTERNS:
        for m in pat.finditer(remainder):
            chunk = (m.group(1) or "").strip()
            if chunk:
                reasoning_parts.append(chunk)
        remainder = pat.sub("", remainder)

    remainder = remainder.strip()
    # Drop common lead-ins before JSON
    for prefix in ("```json", "```JSON", "```"):
        if remainder.startswith(prefix):
            remainder = remainder[len(prefix) :].strip()
    if remainder.endswith("```"):
        remainder = remainder[:-3].strip()

    candidate = remainder
    obj = _extract_balanced_object(candidate)
    if obj is None:
        obj = _extract_balanced_object(raw)
        candidate = raw

    if obj is None:
        raise UnsupportedRunnerError(
            f"LLM did not return JSON object: {raw[:240]!r}"
        )

    reasoning = "\n\n".join(reasoning_parts).strip() or None
    # If no tagged reasoning but there is preamble before '{', keep it as reasoning.
    if reasoning is None:
        pre = candidate[: candidate.find("{")].strip() if "{" in candidate else ""
        if pre and not pre.startswith("{"):
            reasoning = pre

    return ParsedLlmOutput(
        json_text=obj,
        reasoning=reasoning,
        answer_text=remainder or obj,
        raw=raw,
    )


def parse_judgment_json(text: str) -> tuple[dict, str | None]:
    """Parse judge JSON; return (payload, optional_reasoning)."""
    parsed = extract_json_and_reasoning(text)
    try:
        payload = json.loads(parsed.json_text)
    except json.JSONDecodeError as e:
        raise UnsupportedRunnerError(
            f"LLM JSON parse failed: {e}; snippet={parsed.json_text[:200]!r}"
        ) from e
    if not isinstance(payload, dict):
        raise UnsupportedRunnerError(
            f"LLM JSON root must be object, got {type(payload).__name__}"
        )
    return payload, parsed.reasoning
