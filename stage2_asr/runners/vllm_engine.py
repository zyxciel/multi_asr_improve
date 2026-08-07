"""In-process vLLM engine helper (vLLM / vLLM-Ascend).

Uses ``vllm.LLM.generate`` for offline/batched inference — preferred on Ascend 910B
when you do not want a separate OpenAI HTTP server.
"""

from __future__ import annotations

from typing import Any

from stage2_asr.runners.base import UnsupportedRunnerError


def format_chat_prompt(tokenizer, system: str, user: str) -> str:
    messages = [
        {"role": "system", "content": system},
        {"role": "user", "content": user},
    ]
    if hasattr(tokenizer, "apply_chat_template"):
        return tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
    return f"System: {system}\n\nUser: {user}\n\nAssistant:"


def load_vllm_engine(
    model_id: str,
    *,
    tensor_parallel_size: int = 1,
    gpu_memory_utilization: float = 0.90,
    max_model_len: int | None = None,
    trust_remote_code: bool = True,
    dtype: str = "auto",
    engine: Any | None = None,
) -> Any:
    """Return a vllm.LLM instance (or injected test double)."""
    if engine is not None:
        return engine
    try:
        from vllm import LLM  # type: ignore
    except ImportError as e:
        raise UnsupportedRunnerError(
            "vllm not installed. On Ascend 910B install/use vLLM-Ascend, then "
            "--llm-backend vllm_engine."
        ) from e
    kwargs: dict[str, Any] = {
        "model": model_id,
        "tensor_parallel_size": max(1, int(tensor_parallel_size)),
        "gpu_memory_utilization": float(gpu_memory_utilization),
        "trust_remote_code": bool(trust_remote_code),
        "dtype": dtype,
    }
    if max_model_len is not None:
        kwargs["max_model_len"] = int(max_model_len)
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
