from __future__ import annotations

from dataclasses import asdict, dataclass, field
from enum import Enum
from typing import Any


class AsrStatus(str, Enum):
    PROVISIONAL = "provisional"
    FINAL = "final"
    EMPTY = "empty"


@dataclass
class Turn:
    start: float
    end: float
    speaker_id: str
    text: str = ""
    asr_status: AsrStatus = AsrStatus.EMPTY
    source: str = "fused"
    confidence: float = 1.0

    @property
    def duration(self) -> float:
        return float(self.end) - float(self.start)

    def to_dict(self) -> dict[str, Any]:
        d = asdict(self)
        d["asr_status"] = self.asr_status.value
        return d

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "Turn":
        status = d.get("asr_status")
        if status is None:
            asr_status = (
                AsrStatus.PROVISIONAL if str(d.get("text", "")).strip() else AsrStatus.EMPTY
            )
        elif isinstance(status, AsrStatus):
            asr_status = status
        else:
            asr_status = AsrStatus(str(status))
        return cls(
            start=float(d["start"]),
            end=float(d["end"]),
            speaker_id=str(d["speaker_id"]),
            text=str(d.get("text", "")),
            asr_status=asr_status,
            source=str(d.get("source", "fused")),
            confidence=float(d.get("confidence", 1.0)),
        )


@dataclass
class Hypothesis:
    model: str
    text: str
    lid: str | None = None
    meta: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "model": self.model,
            "text": self.text,
            "lid": self.lid,
            "meta": self.meta,
        }


@dataclass
class Edit:
    span_asr: str
    span_out: str
    tier: str
    pinyin_asr: str = ""
    pinyin_out: str = ""
    anchor: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class AsrUnit:
    unit_id: str
    start: float
    end: float
    speaker_id: str
    turn_indices: list[int]
    overlap_ratio: float = 0.0
    contains_overlap: bool = False
    heavy_overlap: bool = False
    skip_asr: bool = False
    skip_reason: str | None = None
    moss_merged: bool = False

    @property
    def duration(self) -> float:
        return float(self.end) - float(self.start)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "AsrUnit":
        return cls(
            unit_id=str(d["unit_id"]),
            start=float(d["start"]),
            end=float(d["end"]),
            speaker_id=str(d["speaker_id"]),
            turn_indices=[int(i) for i in d.get("turn_indices", [])],
            overlap_ratio=float(d.get("overlap_ratio", 0.0)),
            contains_overlap=bool(d.get("contains_overlap", False)),
            heavy_overlap=bool(d.get("heavy_overlap", False)),
            skip_asr=bool(d.get("skip_asr", False)),
            skip_reason=d.get("skip_reason"),
            moss_merged=bool(d.get("moss_merged", False)),
        )


@dataclass
class PipelineConfig:
    max_asr_seconds: float = 30.0
    max_gap_seconds: float = 5.0
    min_asr_seconds: float = 0.35
    min_valid_seconds: float = 0.01
    heavy_overlap_threshold: float = 0.30
    neighbor_max_turns: int = 20
    neighbor_window_seconds: float = 600.0
    llm_max_retries: int = 2
    llm_temperature: float = 0.1
    sample_rate: int = 16000
    energy_window_ms: float = 25.0
    energy_hop_ms: float = 10.0
    edge_zone_ratio: float = 0.10
    # Concurrent Pass A HTTP calls against vLLM/vLLM-Ascend (1 = sequential).
    pass_a_batch_size: int = 1
    # Pass B: 1 = sequential (later turns see earlier Pass B edits).
    # >1 = snapshot meeting draft + judge_many (faster; no in-pass cascade).
    pass_b_batch_size: int = 1
    # Polish: 1 = sequential (later turns see earlier polish edits).
    # >1 = snapshot neighbors + polish_many. Independent of Pass A/B.
    polish_batch_size: int = 1
    publish_batch_size: int = 1
    publish_eval: bool = True
    publish_eval_thinking: bool = True
    glossary: dict | None = None


@dataclass
class LlmJudgment:
    text: str
    base_model: str
    edits: list[Edit] = field(default_factory=list)
    overlap: bool = False
    raw: dict[str, Any] = field(default_factory=dict)
