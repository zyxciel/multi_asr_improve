from __future__ import annotations

"""
Qwen3.8-27B text LLM judge adapter (also works with Qwen3.6-27B via --llm-model-id).

Upstream weights: https://huggingface.co/Qwen/Qwen3.8-27B
Backends:
  - transformers: local AutoModelForCausalLM (device_map=auto) — slow
  - vllm: OpenAI-compatible HTTP server
  - vllm_engine: in-process vllm.LLM (recommended on Ascend 910B / vLLM-Ascend)
  - generate_fn: injected for tests
"""

import concurrent.futures
import json
import time
from typing import Any, Callable

from stage2_asr.pinyin_util import to_pinyin
from stage2_asr.prompt import SYSTEM_PROMPT, render_user_prompt
from stage2_asr.polish_prompt import (
    POLISH_SYSTEM_PROMPT,
    format_polish_hypotheses,
    render_polish_user_prompt,
)
from stage2_asr.polish_cluster import HomophoneCluster
from stage2_asr.polish_cluster_prompt import (
    PARTITION_SYSTEM_PROMPT,
    render_partition_user_prompt,
)
from stage2_asr.publish_prompt import (
    EXTRACT_SYSTEM_PROMPT,
    EVAL_SYSTEM_PROMPT,
    PUBLISH_SYSTEM_PROMPT,
    render_extract_user_prompt,
    render_eval_user_prompt,
    render_publish_user_prompt,
)
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
# Cap stored prompt/response text in llm_infer.jsonl (full length still in *_chars).
_LOG_TEXT_MAX = 16000


def call_with_timeout(fn, timeout_s: float):
    """Run fn with a wall-clock timeout. Cannot kill a stuck generate thread."""
    if timeout_s is None or float(timeout_s) <= 0:
        return fn()
    ex = concurrent.futures.ThreadPoolExecutor(max_workers=1)
    fut = ex.submit(fn)
    try:
        return fut.result(timeout=float(timeout_s))
    except concurrent.futures.TimeoutError as exc:
        raise TimeoutError(f"LLM generate exceeded timeout_s={timeout_s}") from exc
    finally:
        ex.shutdown(wait=False, cancel_futures=True)


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

    def _thinking_template_kwargs(self, enable_thinking: bool | None = None) -> dict[str, bool]:
        think = self.enable_thinking if enable_thinking is None else bool(enable_thinking)
        kwargs = {"enable_thinking": think}
        if not think:
            kwargs["preserve_thinking"] = False
        return kwargs

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

    def polish(
        self,
        *,
        text: str,
        neighbor_draft: list[dict],
        hotwords: list[str],
        turn_index: int,
        unit_id: str,
        hypotheses: list | None = None,
        meeting_hyps: str | None = None,
        cluster_mappings: str = "(none)",
        **_kwargs,
    ) -> dict:
        user = render_polish_user_prompt(
            text=text,
            neighbor_draft=json.dumps(neighbor_draft, ensure_ascii=False),
            hotwords=json.dumps(hotwords, ensure_ascii=False),
            turn_index=int(turn_index),
            hypotheses=format_polish_hypotheses(hypotheses),
            meeting_hyps=str(meeting_hyps or "(none)"),
            cluster_mappings=str(cluster_mappings or "(none)"),
        )
        raw_text = self._generate(
            POLISH_SYSTEM_PROMPT,
            user,
            unit_id=unit_id,
            pass_name="polish",
            max_tokens=2048,
        )
        return self._parse_json(raw_text)

    def partition_cluster(
        self,
        *,
        cluster: "HomophoneCluster | dict",
        unit_id: str = "",
    ) -> dict:
        """Partition one homophone cluster into entity subsets (spec §2).

        One LLM call per cluster. Thinking is ON for this pass only
        (`pass_name="polish_cluster"`, `enable_thinking=True`,
        `max_tokens=2048`), independent of polish span-edit generation which
        keeps thinking off.

        Callers should pass a `HomophoneCluster` (preferred) so occurrences
        and snippets can be rendered. A dict is accepted minimally: it must
        expose the same attributes (`cluster_id`, `surfaces`, `hits`,
        `tone_mismatch_pairs`) and is wrapped into a `HomophoneCluster`.
        """
        if isinstance(cluster, dict):
            cluster = HomophoneCluster(
                cluster_id=str(cluster.get("cluster_id", "")),
                surfaces=tuple(cluster.get("surfaces", ())),
                hits=list(cluster.get("hits", [])),
                tone_mismatch_pairs=list(cluster.get("tone_mismatch_pairs", [])),
            )
        user = render_partition_user_prompt(cluster=cluster)
        raw_text = self._generate(
            PARTITION_SYSTEM_PROMPT,
            user,
            unit_id=unit_id,
            pass_name="polish_cluster",
            enable_thinking=True,
            max_tokens=2048,
        )
        return self._parse_json(raw_text)

    def publish(
        self,
        *,
        meeting: str,
        hotwords: list | None = None,
        glossary: dict | None = None,
        unit_id: str = "",
        **_kwargs,
    ) -> dict:
        user = render_publish_user_prompt(
            meeting=str(meeting or ""),
            hotwords=json.dumps(hotwords or [], ensure_ascii=False),
            glossary=json.dumps(glossary or {}, ensure_ascii=False),
        )
        raw_text = self._generate(
            PUBLISH_SYSTEM_PROMPT,
            user,
            unit_id=unit_id,
            pass_name="publish",
            enable_thinking=False,
            max_tokens=4096,
        )
        return self._parse_json(raw_text)

    def extract_terms(
        self,
        *,
        meeting: str,
        glossary: dict | None = None,
        unit_id: str = "",
        **_kwargs,
    ) -> dict:
        user = render_extract_user_prompt(
            meeting=str(meeting or ""),
            glossary=json.dumps(glossary or {}, ensure_ascii=False),
        )
        raw_text = self._generate(
            EXTRACT_SYSTEM_PROMPT,
            user,
            unit_id=unit_id,
            pass_name="extract",
            enable_thinking=False,
            max_tokens=2048,
        )
        return self._parse_json(raw_text)

    def eval_publish(
        self,
        *,
        original: str,
        published: str,
        unit_id: str = "",
        enable_thinking: bool = True,
        **_kwargs,
    ) -> dict:
        user = render_eval_user_prompt(
            original=str(original or ""),
            published=str(published or ""),
        )
        raw_text = self._generate(
            EVAL_SYSTEM_PROMPT,
            user,
            unit_id=unit_id,
            pass_name="publish_eval",
            enable_thinking=bool(enable_thinking),
            max_tokens=8192,
        )
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
        return self._complete_many(
            prompts_meta,
            system=SYSTEM_PROMPT,
            pass_name="judge_many",
            batch_pass_name="judge_many_batch",
            max_workers=max_workers,
        )

    def polish_many(
        self,
        jobs: list[dict[str, Any]],
        *,
        max_workers: int = 8,
    ) -> list[dict | BaseException]:
        if not jobs:
            return []
        prompts_meta: list[tuple[str, str]] = []
        for job in jobs:
            user = render_polish_user_prompt(
                text=str(job.get("text", "")),
                neighbor_draft=json.dumps(job.get("neighbor_draft") or [], ensure_ascii=False),
                hotwords=json.dumps(job.get("hotwords") or [], ensure_ascii=False),
                turn_index=int(job.get("turn_index") or 0),
                hypotheses=format_polish_hypotheses(job.get("hypotheses")),
                meeting_hyps=str(job.get("meeting_hyps") or "(none)"),
            )
            prompts_meta.append((str(job.get("unit_id", "")), user))
        return self._complete_many(
            prompts_meta,
            system=POLISH_SYSTEM_PROMPT,
            pass_name="polish",
            batch_pass_name="polish_batch",
            max_workers=max_workers,
        )

    def _complete_many(
        self,
        prompts_meta: list[tuple[str, str]],
        *,
        system: str,
        pass_name: str,
        batch_pass_name: str,
        max_workers: int,
    ) -> list[dict | BaseException]:
        if self.generate_fn is not None:
            texts: list[str | BaseException] = []
            for unit_id, user in prompts_meta:
                try:
                    texts.append(
                        self._generate(system, user, unit_id=unit_id, pass_name=pass_name)
                    )
                except BaseException as exc:  # noqa: BLE001
                    texts.append(exc)
            return self._parse_many(texts)

        if self.backend == "vllm_engine":
            return self._judge_many_engine(
                prompts_meta,
                system=system,
                pass_name=pass_name,
                batch_pass_name=batch_pass_name,
            )

        if self.backend == "vllm":
            return self._judge_many_http(
                prompts_meta,
                max_workers=max_workers,
                system=system,
                pass_name=pass_name,
                batch_pass_name=batch_pass_name,
            )

        out: list[dict | BaseException] = []
        for unit_id, user in prompts_meta:
            try:
                out.append(
                    self._parse_json(
                        self._generate(system, user, unit_id=unit_id, pass_name=pass_name)
                    )
                )
            except BaseException as exc:  # noqa: BLE001
                out.append(exc)
        return out

    def _judge_many_engine(
        self,
        prompts_meta: list[tuple[str, str]],
        *,
        system: str = SYSTEM_PROMPT,
        pass_name: str = "judge_many",
        batch_pass_name: str = "judge_many_batch",
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
                        system,
                        user,
                        enable_thinking=self.enable_thinking,
                    )
                )
            else:
                rendered.append(f"System: {system}\n\nUser: {user}\n\nAssistant:")
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
                    "pass": pass_name,
                    "unit_id": unit_id,
                    "ok": True,
                    "latency_s": None,
                    "user_chars": len(user),
                    "response_chars": len(text),
                    "error": None,
                    "user": user[:_LOG_TEXT_MAX],
                    "response": text[:_LOG_TEXT_MAX],
                    "reasoning": (reasoning[:_LOG_TEXT_MAX] if reasoning else None),
                    "enable_thinking": self.enable_thinking,
                }
            )
        self._emit_log(
            {
                "judge": self.name,
                "backend": self.backend,
                "pass": batch_pass_name,
                "n": len(prompts_meta),
                "latency_s": time.time() - t0,
            }
        )
        return self._parse_many(list(texts))

    def _judge_many_http(
        self,
        prompts_meta: list[tuple[str, str]],
        *,
        max_workers: int,
        system: str = SYSTEM_PROMPT,
        pass_name: str = "judge_many",
        batch_pass_name: str = "judge_many_batch",
    ) -> list[dict | BaseException]:
        http_jobs = [
            {
                "base_url": self.base_url or "",
                "model": self.model_id,
                "messages": [
                    {"role": "system", "content": system},
                    {"role": "user", "content": user},
                ],
                "temperature": self.temperature,
                "max_tokens": self.max_tokens,
                "api_key": self.api_key,
                "timeout_s": self.timeout_s,
                "chat_template_kwargs": self._thinking_template_kwargs(),
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
                    "pass": pass_name,
                    "unit_id": unit_id,
                    "ok": not isinstance(raw, BaseException),
                    "latency_s": None,
                    "user_chars": len(user),
                    "response_chars": 0 if isinstance(raw, BaseException) else len(str(raw)),
                    "error": str(raw) if isinstance(raw, BaseException) else None,
                    "user": user[:_LOG_TEXT_MAX],
                    "response": None
                    if isinstance(raw, BaseException)
                    else str(raw)[:_LOG_TEXT_MAX],
                }
            )
        self._emit_log(
            {
                "judge": self.name,
                "backend": self.backend,
                "pass": batch_pass_name,
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

    def _generate(
        self,
        system: str,
        user: str,
        *,
        unit_id: str = "",
        pass_name: str = "generate",
        enable_thinking: bool | None = None,
        max_tokens: int | None = None,
    ) -> str:
        t0 = time.time()
        err: str | None = None
        text = ""
        think = self.enable_thinking if enable_thinking is None else bool(enable_thinking)
        tokens = self.max_tokens if max_tokens is None else int(max_tokens)
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
                        enable_thinking=think,
                    )
                else:
                    prompt = f"System: {system}\n\nUser: {user}\n\nAssistant:"
                text = vllm_generate_texts(
                    engine,
                    [prompt],
                    temperature=self.temperature,
                    max_tokens=tokens,
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
                    max_tokens=tokens,
                    api_key=self.api_key,
                    timeout_s=self.timeout_s,
                    chat_template_kwargs=self._thinking_template_kwargs(think),
                )
            else:
                text = self._generate_transformers(system, user, enable_thinking=think, max_tokens=tokens)
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
                    "user": user[:_LOG_TEXT_MAX],
                    "response": text[:_LOG_TEXT_MAX] if text else None,
                    "enable_thinking": think,
                    "max_tokens": tokens,
                }
            )

    def _generate_transformers(
        self,
        system: str,
        user: str,
        *,
        enable_thinking: bool | None = None,
        max_tokens: int | None = None,
    ) -> str:
        think = self.enable_thinking if enable_thinking is None else bool(enable_thinking)
        tokens = self.max_tokens if max_tokens is None else int(max_tokens)
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
                enable_thinking=think,
            )
        except TypeError:
            prompt = tok.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )
        inputs = tok([prompt], return_tensors="pt").to(model.device)

        def _run() -> str:
            out = model.generate(
                **inputs,
                max_new_tokens=tokens,
                do_sample=self.temperature > 0,
                temperature=max(self.temperature, 1e-5),
            )
            gen = out[0][inputs["input_ids"].shape[-1] :]
            return tok.decode(gen, skip_special_tokens=True)

        return call_with_timeout(_run, self.timeout_s)

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
