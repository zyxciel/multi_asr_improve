from __future__ import annotations

"""
DeepSeek text LLM judge adapter (Pass A fallback after Qwen retries).

Same backends as Qwen36LlmJudge: transformers | vllm (OpenAI-compat HTTP).
"""

import json
import re
import time
from typing import Any, Callable

from stage2_asr.pinyin_util import to_pinyin
from stage2_asr.prompt import SYSTEM_PROMPT, render_user_prompt
from stage2_asr.runners.base import UnsupportedRunnerError
from stage2_asr.runners.openai_compat import chat_completion
from stage2_asr.types import Hypothesis

LogFn = Callable[[dict[str, Any]], None]


class DeepSeekLlmJudge:
    name = "deepseek"

    def __init__(
        self,
        *,
        enabled: bool = False,
        generate_fn=None,
        model_id: str = "deepseek-ai/DeepSeek-V2.5",
        temperature: float = 0.1,
        backend: str = "transformers",
        base_url: str | None = None,
        api_key: str | None = None,
        timeout_s: float = 300.0,
        max_tokens: int = 1024,
        log_fn: LogFn | None = None,
    ):
        self.enabled = enabled
        self.generate_fn = generate_fn
        self.model_id = model_id
        self.temperature = temperature
        self.backend = str(backend).lower()
        self.base_url = base_url
        self.api_key = api_key
        self.timeout_s = timeout_s
        self.max_tokens = max_tokens
        self.log_fn = log_fn
        self._pipe = None
        if self.backend not in {"transformers", "vllm"}:
            raise ValueError(f"unsupported llm backend {backend!r}; expected transformers|vllm")

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
        raw_text = self._generate(SYSTEM_PROMPT, user, unit_id=unit_id)
        return self._parse_json(raw_text)

    def _emit_log(self, event: dict[str, Any]) -> None:
        if self.log_fn is not None:
            self.log_fn(event)

    def _generate(self, system: str, user: str, *, unit_id: str = "") -> str:
        t0 = time.time()
        err: str | None = None
        text = ""
        try:
            if self.generate_fn is not None:
                text = self.generate_fn(system, user)
            elif not self.enabled:
                raise UnsupportedRunnerError(
                    "DeepSeekLlmJudge disabled. Use MockLlmJudge, generate_fn=..., or enabled=True."
                )
            elif self.backend == "vllm":
                if not self.base_url:
                    raise UnsupportedRunnerError(
                        "vllm backend requires base_url (OpenAI-compatible server)"
                    )
                text = chat_completion(
                    base_url=self.base_url,
                    model=self.model_id,
                    messages=[
                        {"role": "system", "content": system},
                        {"role": "user", "content": user},
                    ],
                    temperature=self.temperature,
                    max_tokens=self.max_tokens,
                    api_key=self.api_key,
                    timeout_s=self.timeout_s,
                )
            else:
                text = self._generate_transformers(system, user)
            return text
        except Exception as exc:
            err = str(exc)
            raise
        finally:
            self._emit_log(
                {
                    "judge": self.name,
                    "backend": self.backend if self.generate_fn is None else "generate_fn",
                    "pass": "judge",
                    "unit_id": unit_id,
                    "ok": err is None,
                    "latency_s": time.time() - t0,
                    "user_chars": len(user),
                    "response_chars": len(text),
                    "error": err,
                    "response": text[:4000] if text else None,
                }
            )

    def _generate_transformers(self, system: str, user: str) -> str:
        if self._pipe is None:
            try:
                from transformers import AutoModelForCausalLM, AutoTokenizer  # type: ignore
            except ImportError as e:
                raise UnsupportedRunnerError(
                    "transformers/torch required for DeepSeekLlmJudge"
                ) from e
            tok = AutoTokenizer.from_pretrained(self.model_id, trust_remote_code=True)
            model = AutoModelForCausalLM.from_pretrained(
                self.model_id,
                trust_remote_code=True,
                torch_dtype="auto",
                device_map="auto",
            )
            self._pipe = (tok, model)
        tok, model = self._pipe
        messages = [
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ]
        prompt = tok.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        inputs = tok([prompt], return_tensors="pt").to(model.device)
        out = model.generate(
            **inputs,
            max_new_tokens=self.max_tokens,
            do_sample=self.temperature > 0,
            temperature=max(self.temperature, 1e-5),
        )
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
