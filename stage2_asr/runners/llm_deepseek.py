from __future__ import annotations

"""
DeepSeek text LLM judge adapter (Pass A fallback after Qwen retries).

Same backends as Qwen36LlmJudge: transformers | vllm (OpenAI-compat HTTP).
"""

import json
import time
from typing import Any, Callable

from stage2_asr.pinyin_util import to_pinyin
from stage2_asr.prompt import SYSTEM_PROMPT, render_user_prompt
from stage2_asr.llm_parse import parse_judgment_json
from stage2_asr.runners.base import UnsupportedRunnerError
from stage2_asr.runners.openai_compat import chat_completion
from stage2_asr.runners.vllm_engine import (
    format_chat_prompt,
    load_vllm_engine,
    vllm_generate_texts,
)
from stage2_asr.types import Hypothesis

LogFn = Callable[[dict[str, Any]], None]
_BACKENDS = {"transformers", "vllm", "vllm_engine"}
_LOG_TEXT_MAX = 16000


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
        engine=None,
        tensor_parallel_size: int = 1,
        gpu_memory_utilization: float = 0.90,
        max_model_len: int | None = None,
        dtype: str = "auto",
        enforce_eager: bool = True,
        use_v1: bool | None = False,
        enable_thinking: bool = False,
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
        self._engine = engine
        self.tensor_parallel_size = tensor_parallel_size
        self.gpu_memory_utilization = gpu_memory_utilization
        self.max_model_len = max_model_len
        self.dtype = dtype
        self.enforce_eager = enforce_eager
        self.use_v1 = use_v1
        self.enable_thinking = bool(enable_thinking)
        if self.backend not in _BACKENDS:
            raise ValueError(
                f"unsupported llm backend {backend!r}; expected {sorted(_BACKENDS)}"
            )
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
            elif self.backend == "vllm_engine":
                if self._engine is None:
                    self._engine = load_vllm_engine(
                        self.model_id,
                        tensor_parallel_size=self.tensor_parallel_size,
                        gpu_memory_utilization=self.gpu_memory_utilization,
                        max_model_len=self.max_model_len,
                        dtype=self.dtype,
                        enforce_eager=self.enforce_eager,
                        use_v1=self.use_v1,
                    )
                tokenizer = (
                    self._engine.get_tokenizer()
                    if hasattr(self._engine, "get_tokenizer")
                    else None
                )
                if tokenizer is not None:
                    prompt = format_chat_prompt(
                        tokenizer,
                        system,
                        user,
                        enable_thinking=self.enable_thinking,
                    )
                else:
                    prompt = f"System: {system}\n\nUser: {user}\n\nAssistant:"
                text = vllm_generate_texts(
                    self._engine,
                    [prompt],
                    temperature=self.temperature,
                    max_tokens=self.max_tokens,
                )[0]
            elif self.backend == "vllm":
                if not self.base_url:
                    raise UnsupportedRunnerError(
                        "vllm (HTTP) backend requires base_url; "
                        "or use --llm-backend vllm_engine"
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
                    "user": user[:_LOG_TEXT_MAX],
                    "response": text[:_LOG_TEXT_MAX] if text else None,
                    "enable_thinking": self.enable_thinking,
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
        try:
            prompt = tok.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
                enable_thinking=self.enable_thinking,
            )
        except TypeError:
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

    def _parse_json(self, text: str) -> dict:
        payload, reasoning = parse_judgment_json(text)
        if reasoning and self.log_fn is not None:
            self._emit_log(
                {
                    "judge": self.name,
                    "backend": self.backend,
                    "pass": "parse_reasoning",
                    "ok": True,
                    "reasoning": reasoning[:_LOG_TEXT_MAX],
                    "enable_thinking": self.enable_thinking,
                }
            )
        return payload
