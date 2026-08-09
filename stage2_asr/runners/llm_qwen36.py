from __future__ import annotations

"""
Qwen3.6-27B text LLM judge adapter.

Upstream weights: https://huggingface.co/Qwen/Qwen3.6-27B
Backends:
  - transformers: local AutoModelForCausalLM (device_map=auto) — slow
  - vllm: OpenAI-compatible HTTP server
  - vllm_engine: in-process vllm.LLM (recommended on Ascend 910B / vLLM-Ascend)
  - generate_fn: injected for tests
"""

import json
import time
from typing import Any, Callable

from stage2_asr.pinyin_util import to_pinyin
from stage2_asr.prompt import SYSTEM_PROMPT, render_user_prompt
from stage2_asr.llm_parse import parse_judgment_json
from stage2_asr.runners.base import UnsupportedRunnerError
from stage2_asr.runners.openai_compat import chat_completion, chat_completion_many
from stage2_asr.runners.vllm_engine import (
    format_chat_prompt,
    load_vllm_engine,
    vllm_generate_texts,
)
from stage2_asr.types import Hypothesis

LogFn = Callable[[dict[str, Any]], None]
_BACKENDS = {"transformers", "vllm", "vllm_engine"}


class Qwen36LlmJudge:
    name = "qwen36"

    def __init__(
        self,
        *,
        enabled: bool = False,
        generate_fn=None,
        model_id: str = "Qwen/Qwen3.6-27B",
        temperature: float = 0.1,
        backend: str = "transformers",
        base_url: str | None = None,
        api_key: str | None = None,
        timeout_s: float = 300.0,
        max_tokens: int = 1024,
        log_fn: LogFn | None = None,
        # vllm_engine options
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

    def build_user_prompt(
        self,
        *,
        hypotheses: list[Hypothesis],
        neighbor_draft: list[dict],
        hotwords: list[str],
        overlap: bool,
        heavy_overlap: bool,
    ) -> str:
        return render_user_prompt(
            hypotheses_with_pinyin=self._format_hyps(hypotheses),
            hotwords=json.dumps(hotwords, ensure_ascii=False),
            neighbor_draft=json.dumps(neighbor_draft, ensure_ascii=False),
            overlap_flag=overlap,
            heavy_overlap_flag=heavy_overlap,
        )

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
        user = self.build_user_prompt(
            hypotheses=hypotheses,
            neighbor_draft=neighbor_draft,
            hotwords=hotwords,
            overlap=overlap,
            heavy_overlap=heavy_overlap,
        )
        raw_text = self._generate(SYSTEM_PROMPT, user, unit_id=unit_id, pass_name="judge")
        return self._parse_json(raw_text)

    def judge_many(
        self,
        jobs: list[dict[str, Any]],
        *,
        max_workers: int = 8,
    ) -> list[dict | BaseException]:
        """
        Batched judges for Pass A.

        - vllm_engine: one ``LLM.generate`` call (true continuous batching)
        - vllm (HTTP): concurrent OpenAI calls
        - transformers: sequential
        """
        if not jobs:
            return []

        prompts_meta: list[tuple[str, str]] = []
        for job in jobs:
            user = self.build_user_prompt(
                hypotheses=job["hypotheses"],
                neighbor_draft=job["neighbor_draft"],
                hotwords=job["hotwords"],
                overlap=job["overlap"],
                heavy_overlap=job["heavy_overlap"],
            )
            prompts_meta.append((str(job.get("unit_id", "")), user))

        if self.generate_fn is not None:
            texts: list[str | BaseException] = []
            for unit_id, user in prompts_meta:
                try:
                    texts.append(
                        self._generate(
                            SYSTEM_PROMPT, user, unit_id=unit_id, pass_name="judge_many"
                        )
                    )
                except BaseException as exc:  # noqa: BLE001
                    texts.append(exc)
            return self._parse_many(texts)

        if self.backend == "vllm_engine":
            return self._judge_many_engine(prompts_meta)

        if self.backend == "vllm":
            return self._judge_many_http(prompts_meta, max_workers=max_workers)

        # transformers / other: sequential
        out: list[dict | BaseException] = []
        for job in jobs:
            try:
                out.append(self.judge(**job))
            except BaseException as exc:  # noqa: BLE001
                out.append(exc)
        return out

    def _judge_many_engine(
        self, prompts_meta: list[tuple[str, str]]
    ) -> list[dict | BaseException]:
        t0 = time.time()
        engine = self._ensure_engine()
        tokenizer = engine.get_tokenizer() if hasattr(engine, "get_tokenizer") else None
        rendered: list[str] = []
        for _, user in prompts_meta:
            if tokenizer is not None:
                rendered.append(
                    format_chat_prompt(
                        tokenizer,
                        SYSTEM_PROMPT,
                        user,
                        enable_thinking=self.enable_thinking,
                    )
                )
            else:
                rendered.append(f"System: {SYSTEM_PROMPT}\n\nUser: {user}\n\nAssistant:")
        try:
            texts = vllm_generate_texts(
                engine,
                rendered,
                temperature=self.temperature,
                max_tokens=self.max_tokens,
            )
        except Exception as exc:  # noqa: BLE001
            return [exc] * len(prompts_meta)

        for (unit_id, user), text in zip(prompts_meta, texts):
            reasoning = None
            try:
                _, reasoning = parse_judgment_json(text)
            except Exception:  # noqa: BLE001
                pass
            self._emit_log(
                {
                    "judge": self.name,
                    "backend": self.backend,
                    "pass": "judge_many",
                    "unit_id": unit_id,
                    "ok": True,
                    "latency_s": None,
                    "user_chars": len(user),
                    "response_chars": len(text),
                    "error": None,
                    "response": text[:4000],
                    "reasoning": (reasoning[:4000] if reasoning else None),
                    "enable_thinking": self.enable_thinking,
                }
            )
        self._emit_log(
            {
                "judge": self.name,
                "backend": self.backend,
                "pass": "judge_many_batch",
                "n": len(prompts_meta),
                "latency_s": time.time() - t0,
            }
        )
        return self._parse_many(list(texts))

    def _judge_many_http(
        self, prompts_meta: list[tuple[str, str]], *, max_workers: int
    ) -> list[dict | BaseException]:
        http_jobs = [
            {
                "base_url": self.base_url or "",
                "model": self.model_id,
                "messages": [
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": user},
                ],
                "temperature": self.temperature,
                "max_tokens": self.max_tokens,
                "api_key": self.api_key,
                "timeout_s": self.timeout_s,
            }
            for _, user in prompts_meta
        ]
        t0 = time.time()
        raw_list = chat_completion_many(http_jobs, max_workers=max_workers)
        for (unit_id, user), raw in zip(prompts_meta, raw_list):
            self._emit_log(
                {
                    "judge": self.name,
                    "backend": self.backend,
                    "pass": "judge_many",
                    "unit_id": unit_id,
                    "ok": not isinstance(raw, BaseException),
                    "latency_s": None,
                    "user_chars": len(user),
                    "response_chars": 0 if isinstance(raw, BaseException) else len(str(raw)),
                    "error": str(raw) if isinstance(raw, BaseException) else None,
                    "response": None if isinstance(raw, BaseException) else str(raw)[:4000],
                }
            )
        self._emit_log(
            {
                "judge": self.name,
                "backend": self.backend,
                "pass": "judge_many_batch",
                "n": len(prompts_meta),
                "latency_s": time.time() - t0,
                "max_workers": max_workers,
            }
        )
        return self._parse_many(raw_list)

    def _parse_many(
        self, texts: list[str | BaseException]
    ) -> list[dict | BaseException]:
        out: list[dict | BaseException] = []
        for raw in texts:
            if isinstance(raw, BaseException):
                out.append(raw)
                continue
            try:
                out.append(self._parse_json(raw))
            except BaseException as exc:  # noqa: BLE001
                out.append(exc)
        return out

    def _emit_log(self, event: dict[str, Any]) -> None:
        if self.log_fn is not None:
            self.log_fn(event)

    def _ensure_engine(self):
        if self._engine is None:
            if not self.enabled:
                raise UnsupportedRunnerError(
                    "Qwen36LlmJudge disabled. Use enabled=True for vllm_engine."
                )
            self._engine = load_vllm_engine(
                self.model_id,
                tensor_parallel_size=self.tensor_parallel_size,
                gpu_memory_utilization=self.gpu_memory_utilization,
                max_model_len=self.max_model_len,
                dtype=self.dtype,
                enforce_eager=self.enforce_eager,
                use_v1=self.use_v1,
            )
        return self._engine

    def _generate(self, system: str, user: str, *, unit_id: str = "", pass_name: str = "generate") -> str:
        t0 = time.time()
        err: str | None = None
        text = ""
        try:
            if self.generate_fn is not None:
                text = self.generate_fn(system, user)
            elif not self.enabled:
                raise UnsupportedRunnerError(
                    "Qwen36LlmJudge disabled. Use MockLlmJudge or enabled=True / generate_fn=..."
                )
            elif self.backend == "vllm_engine":
                engine = self._ensure_engine()
                tokenizer = engine.get_tokenizer() if hasattr(engine, "get_tokenizer") else None
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
                    engine,
                    [prompt],
                    temperature=self.temperature,
                    max_tokens=self.max_tokens,
                )[0]
            elif self.backend == "vllm":
                if not self.base_url:
                    raise UnsupportedRunnerError(
                        "vllm (HTTP) backend requires base_url; "
                        "or use --llm-backend vllm_engine for in-process vllm.LLM"
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
                    "pass": pass_name,
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
                raise UnsupportedRunnerError("transformers/torch required for Qwen36LlmJudge") from e
            tok = AutoTokenizer.from_pretrained(self.model_id, trust_remote_code=True)
            model = AutoModelForCausalLM.from_pretrained(
                self.model_id, trust_remote_code=True, torch_dtype="auto", device_map="auto"
            )
            self._pipe = (tok, model)
        tok, model = self._pipe
        messages = [{"role": "system", "content": system}, {"role": "user", "content": user}]
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
                    "reasoning": reasoning[:4000],
                    "enable_thinking": self.enable_thinking,
                }
            )
        return payload
