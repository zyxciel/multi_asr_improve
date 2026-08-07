from __future__ import annotations

import json
from types import SimpleNamespace

from stage2_asr.runners.llm_qwen36 import Qwen36LlmJudge
from stage2_asr.types import Hypothesis


class _FakeTokenizer:
    def apply_chat_template(self, messages, tokenize=False, add_generation_prompt=True):
        parts = [f"{m['role']}:{m['content']}" for m in messages]
        return "\n".join(parts) + ("\nassistant:" if add_generation_prompt else "")


class _FakeEngine:
    def __init__(self):
        self.calls: list[list[str]] = []

    def get_tokenizer(self):
        return _FakeTokenizer()

    def generate(self, prompts, params):
        self.calls.append(list(prompts))
        outs = []
        for _ in prompts:
            payload = json.dumps(
                {
                    "text": "采用",
                    "base_model": "qwen",
                    "edits": [],
                    "overlap": False,
                },
                ensure_ascii=False,
            )
            outs.append(SimpleNamespace(outputs=[SimpleNamespace(text=payload)]))
        return outs


def test_normalize_vllm_dtype():
    from stage2_asr.runners.vllm_engine import normalize_vllm_dtype

    assert normalize_vllm_dtype("bf16") == "bfloat16"
    assert normalize_vllm_dtype("bfloat16") == "bfloat16"
    assert normalize_vllm_dtype("fp16") == "float16"


def test_prepare_vllm_process_env_sets_defaults(monkeypatch):
    import os

    from stage2_asr.runners.vllm_engine import prepare_vllm_process_env

    for key in (
        "VLLM_WORKER_MULTIPROC_METHOD",
        "OMP_NUM_THREADS",
        "MKL_NUM_THREADS",
        "VLLM_HOST_IP",
        "VLLM_USE_V1",
        "HOST_IP",
    ):
        monkeypatch.delenv(key, raising=False)

    prepare_vllm_process_env(use_v1=False)
    assert os.environ["VLLM_WORKER_MULTIPROC_METHOD"] == "spawn"
    assert os.environ["OMP_NUM_THREADS"] == "1"
    assert os.environ["VLLM_USE_V1"] == "0"
    assert os.environ["VLLM_HOST_IP"] == "127.0.0.1"


def test_vllm_engine_single_and_batch():
    engine = _FakeEngine()
    logs: list[dict] = []
    judge = Qwen36LlmJudge(
        enabled=True,
        backend="vllm_engine",
        model_id="dummy",
        engine=engine,
        log_fn=logs.append,
    )
    out = judge.judge(
        hypotheses=[Hypothesis("qwen", "产用")],
        neighbor_draft=[],
        hotwords=[],
        overlap=False,
        heavy_overlap=False,
        unit_id="u0",
    )
    assert out["text"] == "采用"
    assert engine.calls and len(engine.calls[0]) == 1

    jobs = [
        {
            "hypotheses": [Hypothesis("qwen", "产用")],
            "neighbor_draft": [],
            "hotwords": [],
            "overlap": False,
            "heavy_overlap": False,
            "unit_id": f"u{i}",
        }
        for i in range(3)
    ]
    many = judge.judge_many(jobs, max_workers=3)
    assert len(many) == 3
    assert all(isinstance(x, dict) and x["text"] == "采用" for x in many)
    assert any(len(c) == 3 for c in engine.calls)
    assert any(e.get("pass") == "judge_many_batch" for e in logs)
