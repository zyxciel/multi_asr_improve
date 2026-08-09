"""In-process vLLM engine helper (vLLM / vLLM-Ascend).

Uses ``vllm.LLM.generate`` for offline/batched inference — preferred on Ascend 910B
when you do not want a separate OpenAI HTTP server.
"""

from __future__ import annotations

import os
from typing import Any

from stage2_asr.runners.base import UnsupportedRunnerError


def prepare_vllm_process_env(*, use_v1: bool | None = None) -> None:
    """
    Mitigate PyTorch OpenMP + vLLM V1 multiprocess crashes.

    Root cause seen on Ascend / vLLM 0.18:
      c10::Error Invalid thread pool! at ParallelOpenMP.cpp
    during EngineCore SyncMPClient startup (fork/thread conflict).

    Must run *before* ``from vllm import LLM``.
    """
    # Prefer spawn over fork after torch/OpenMP has been initialized.
    os.environ.setdefault("VLLM_WORKER_MULTIPROC_METHOD", "spawn")
    # Avoid nested OpenMP pools in engine worker processes.
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    os.environ.setdefault("MKL_NUM_THREADS", "1")
    os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
    os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")
    # HOST_IP is ignored; vLLM wants VLLM_HOST_IP for multi-proc.
    if "HOST_IP" in os.environ and "VLLM_HOST_IP" not in os.environ:
        os.environ["VLLM_HOST_IP"] = os.environ["HOST_IP"]
    os.environ.setdefault("VLLM_HOST_IP", "127.0.0.1")
    if use_v1 is False:
        os.environ["VLLM_USE_V1"] = "0"
    elif use_v1 is True:
        os.environ["VLLM_USE_V1"] = "1"


def format_chat_prompt(
    tokenizer,
    system: str,
    user: str,
    *,
    enable_thinking: bool = False,
) -> str:
    """Apply chat template; disable Qwen3-style thinking by default."""
    messages = [
        {"role": "system", "content": system},
        {"role": "user", "content": user},
    ]
    if hasattr(tokenizer, "apply_chat_template"):
        # Qwen3 / Qwen3.6: enable_thinking=False keeps CoT out of the decode path.
        kwargs = {
            "tokenize": False,
            "add_generation_prompt": True,
        }
        try:
            return tokenizer.apply_chat_template(
                messages,
                enable_thinking=bool(enable_thinking),
                **kwargs,
            )
        except TypeError:
            try:
                return tokenizer.apply_chat_template(
                    messages,
                    chat_template_kwargs={"enable_thinking": bool(enable_thinking)},
                    **kwargs,
                )
            except TypeError:
                return tokenizer.apply_chat_template(messages, **kwargs)
    return f"System: {system}\n\nUser: {user}\n\nAssistant:"



def normalize_vllm_dtype(dtype: str | None) -> str:
    """Map user aliases to vLLM dtype strings."""
    if not dtype:
        return "auto"
    d = str(dtype).strip().lower()
    aliases = {
        "bf16": "bfloat16",
        "bfloat16": "bfloat16",
        "fp16": "float16",
        "float16": "float16",
        "half": "float16",
        "fp32": "float32",
        "float32": "float32",
        "auto": "auto",
    }
    if d not in aliases:
        raise ValueError(
            f"unsupported vllm dtype {dtype!r}; expected one of {sorted(set(aliases))}"
        )
    return aliases[d]


def load_vllm_engine(
    model_id: str,
    *,
    tensor_parallel_size: int = 1,
    gpu_memory_utilization: float = 0.90,
    max_model_len: int | None = None,
    trust_remote_code: bool = True,
    dtype: str = "auto",
    enforce_eager: bool = True,
    use_v1: bool | None = False,
    engine: Any | None = None,
) -> Any:
    """Return a vllm.LLM instance (or injected test double).

    Defaults tuned for stability on Ascend + vLLM 0.18:
    - prepare_vllm_process_env(use_v1=False) → avoid V1 SyncMPClient OpenMP crash
    - enforce_eager=True → skip CUDA/NPU graph capture issues during bring-up

    For 2 NPUs: tensor_parallel_size=2 and ASCEND_RT_VISIBLE_DEVICES=0,1
    """
    if engine is not None:
        return engine

    prepare_vllm_process_env(use_v1=use_v1)

    try:
        from vllm import LLM  # type: ignore
    except ImportError as e:
        raise UnsupportedRunnerError(
            "vllm not installed. On Ascend 910B install/use vLLM-Ascend, then "
            "--llm-backend vllm_engine."
        ) from e

    dtype_norm = normalize_vllm_dtype(dtype)
    kwargs: dict[str, Any] = {
        "model": model_id,
        "tensor_parallel_size": max(1, int(tensor_parallel_size)),
        "gpu_memory_utilization": float(gpu_memory_utilization),
        "trust_remote_code": bool(trust_remote_code),
        "dtype": dtype_norm,
    }
    if max_model_len is not None:
        kwargs["max_model_len"] = int(max_model_len)
    if enforce_eager:
        kwargs["enforce_eager"] = True

    try:
        return LLM(**kwargs)
    except TypeError:
        # Older vLLM without enforce_eager
        kwargs.pop("enforce_eager", None)
        return LLM(**kwargs)


def vllm_generate_texts(
    engine: Any,
    prompts: list[str],
    *,
    temperature: float = 0.1,
    max_tokens: int = 1024,
) -> list[str]:
    """Batch generate; returns one string per prompt."""
    if not prompts:
        return []
    temp = float(temperature)
    try:
        from vllm import SamplingParams  # type: ignore

        if temp <= 1e-5:
            params = SamplingParams(temperature=0.0, max_tokens=int(max_tokens))
        else:
            params = SamplingParams(
                temperature=max(temp, 0.0),
                max_tokens=int(max_tokens),
            )
    except ImportError:
        # Test doubles / environments without vllm installed.
        params = {"temperature": temp, "max_tokens": int(max_tokens)}

    outputs = engine.generate(prompts, params)
    texts: list[str] = []
    for out in outputs:
        if not getattr(out, "outputs", None):
            texts.append("")
            continue
        texts.append(str(out.outputs[0].text or ""))
    return texts
