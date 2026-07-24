from __future__ import annotations

"""
Qwen3.6-27B text LLM judge adapter.

Upstream weights: https://huggingface.co/Qwen/Qwen3.6-27B
Uses stage2_asr.prompt templates. Supports:
  - injected `generate_fn(system, user) -> str` for tests
  - transformers AutoModelForCausalLM when enabled (may download if model_id remote)
"""

import json
import re

from stage2_asr.pinyin_util import to_pinyin
from stage2_asr.prompt import SYSTEM_PROMPT, render_user_prompt
from stage2_asr.runners.base import UnsupportedRunnerError
from stage2_asr.types import Hypothesis


class Qwen36LlmJudge:
    name = "qwen36"

    def __init__(
        self,
        *,
        enabled: bool = False,
        generate_fn=None,
        model_id: str = "Qwen/Qwen3.6-27B",
        temperature: float = 0.1,
    ):
        self.enabled = enabled
        self.generate_fn = generate_fn
        self.model_id = model_id
        self.temperature = temperature
        self._pipe = None

    def _format_hyps(self, hypotheses: list[Hypothesis]) -> str:
        lines = []
        for h in hypotheses:
            lines.append(f"- {h.model}: {h.text} (pinyin: {to_pinyin(h.text)})")
        return "\n".join(lines) if lines else "(none)"

    def judge(
        self,
        *,
        hypotheses: list[Hypothesis],
        neighbor_draft: list[dict],
        hotwords: list[str],
        overlap: bool,
        heavy_overlap: bool,
        unit_id: str,
    ) -> dict:
        user = render_user_prompt(
            hypotheses_with_pinyin=self._format_hyps(hypotheses),
            hotwords=json.dumps(hotwords, ensure_ascii=False),
            neighbor_draft=json.dumps(neighbor_draft, ensure_ascii=False),
            overlap_flag=overlap,
            heavy_overlap_flag=heavy_overlap,
        )
        raw_text = self._generate(SYSTEM_PROMPT, user)
        return self._parse_json(raw_text)

    def _generate(self, system: str, user: str) -> str:
        if self.generate_fn is not None:
            return self.generate_fn(system, user)
        if not self.enabled:
            raise UnsupportedRunnerError(
                "Qwen36LlmJudge disabled. Use MockLlmJudge or enabled=True / generate_fn=..."
            )
        if self._pipe is None:
            try:
                from transformers import AutoModelForCausalLM, AutoTokenizer  # type: ignore
                import torch  # type: ignore
            except ImportError as e:
                raise UnsupportedRunnerError("transformers/torch required for Qwen36LlmJudge") from e
            tok = AutoTokenizer.from_pretrained(self.model_id, trust_remote_code=True)
            model = AutoModelForCausalLM.from_pretrained(
                self.model_id, trust_remote_code=True, torch_dtype="auto", device_map="auto"
            )
            self._pipe = (tok, model)
        tok, model = self._pipe
        messages = [{"role": "system", "content": system}, {"role": "user", "content": user}]
        prompt = tok.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        inputs = tok([prompt], return_tensors="pt").to(model.device)
        out = model.generate(**inputs, max_new_tokens=1024, do_sample=self.temperature > 0, temperature=max(self.temperature, 1e-5))
        gen = out[0][inputs["input_ids"].shape[-1] :]
        return tok.decode(gen, skip_special_tokens=True)

    @staticmethod
    def _parse_json(text: str) -> dict:
        text = text.strip()
        try:
            return json.loads(text)
        except json.JSONDecodeError:
            m = re.search(r"\{.*\}", text, flags=re.S)
            if not m:
                raise UnsupportedRunnerError(f"LLM did not return JSON: {text[:200]}")
            return json.loads(m.group(0))
