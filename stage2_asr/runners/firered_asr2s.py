from __future__ import annotations

"""
FireRedASR2S adapter.

Upstream: https://github.com/FireRedTeam/FireRedASR2S
Uses FireRedAsr2System with:
  enable_vad=False  (Stage-1/Stage-2 crops only)
  enable_lid=True
  enable_punc=True

    from fireredasr2s import FireRedAsr2System, FireRedAsr2SystemConfig
    cfg = FireRedAsr2SystemConfig(enable_vad=False, enable_lid=True, enable_punc=True)
    system = FireRedAsr2System(cfg)
    out = system.process(wav_path, uttid=...)
"""

from dataclasses import dataclass

from stage2_asr.runners.base import UnsupportedRunnerError
from stage2_asr.types import AsrUnit, Hypothesis, Turn


@dataclass
class FireRedAsr2sConfig:
    vad: bool = False
    lid: bool = True
    punc: bool = True
    asr: bool = True
    asr_type: str = "aed"
    asr_model_dir: str = "pretrained_models/FireRedASR2-AED"
    lid_model_dir: str = "pretrained_models/FireRedLID"
    punc_model_dir: str = "pretrained_models/FireRedPunc"


class FireRedAsr2sRunner:
    name = "firered_asr2s"

    def __init__(
        self,
        *,
        enabled: bool = False,
        system=None,
        config: FireRedAsr2sConfig | None = None,
    ):
        self.enabled = enabled
        self.system = system
        self.config = config or FireRedAsr2sConfig()

    def _ensure_system(self):
        if self.system is not None:
            return self.system
        if not self.enabled:
            raise UnsupportedRunnerError(
                "FireRedAsr2sRunner disabled. Use MockAsrRunner or enabled=True with fireredasr2s + local weights."
            )
        try:
            from fireredasr2s import FireRedAsr2System, FireRedAsr2SystemConfig  # type: ignore
        except ImportError as e:
            raise UnsupportedRunnerError(
                "fireredasr2s not installed. See https://github.com/FireRedTeam/FireRedASR2S"
            ) from e
        c = self.config
        sys_cfg = FireRedAsr2SystemConfig(
            enable_vad=bool(c.vad),
            enable_lid=bool(c.lid),
            enable_punc=bool(c.punc),
            asr_type=c.asr_type,
            asr_model_dir=c.asr_model_dir,
            lid_model_dir=c.lid_model_dir,
            punc_model_dir=c.punc_model_dir,
        )
        self.system = FireRedAsr2System(sys_cfg)
        return self.system

    def transcribe_unit(
        self,
        unit: AsrUnit,
        turns: list[Turn],
        audio_path: str,
        *,
        moss_exclusive: bool = False,
        crop_path: str | None = None,
    ) -> list[Hypothesis]:
        if moss_exclusive or not self.config.asr:
            return []
        system = self._ensure_system()
        path = crop_path or audio_path
        result = system.process(path, uttid=unit.unit_id)
        # Official system returns dict with sentences / text variants
        text = ""
        lid = None
        if isinstance(result, dict):
            if "text" in result:
                text = str(result["text"])
            elif "sentences" in result:
                parts = []
                lids = []
                for s in result["sentences"]:
                    parts.append(str(s.get("text", "")))
                    if s.get("lang"):
                        lids.append(str(s["lang"]))
                text = "".join(parts)
                lid = lids[0] if lids else None
            elif "punc_text" in result:
                text = str(result["punc_text"])
        else:
            text = str(result)
        return [
            Hypothesis(
                model="firered",
                text=text,
                lid=lid,
                meta={
                    "vad": self.config.vad,
                    "lid": self.config.lid,
                    "punc": self.config.punc,
                    "backend": "firered_asr2s",
                },
            )
        ]
