from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from stage2_asr.batch import build_runners, run_batch
from stage2_asr.hotwords import load_hotwords
from stage2_asr.pipeline import run_pipeline
from stage2_asr.publish import load_glossary
from stage2_asr.types import PipelineConfig


def _add_common_run_args(p: argparse.ArgumentParser) -> None:
    p.add_argument("--mock", action="store_true", help="Use mock ASR/LLM (no model weights)")
    p.add_argument(
        "--backend",
        choices=["mock", "real"],
        default=None,
        help="Backend selection (default: mock if --mock else real)",
    )
    p.add_argument("--mock-hyps", default=None, help="Optional mock hypothesis fixture JSON")
    p.add_argument(
        "--hotwords",
        default=None,
        help="Hotword list path: JSON array/object or plaintext one-term-per-line (e.g. docs/hotwords.txt)",
    )
    p.add_argument(
        "--stage",
        default="all",
        choices=["all", "asr", "pass_a", "pass_b", "llm", "polish", "publish"],
        help="Execution stage: all | asr | pass_a | pass_b | llm | polish | publish",
    )
    p.add_argument(
        "--asr-models",
        default="moss,qwen,firered",
        help="Comma-separated ASR models for ASR stage/cache: moss,qwen,firered",
    )
    p.add_argument("--max-asr-seconds", type=float, default=30.0)
    p.add_argument("--qwen-model-id", default="Qwen/Qwen3-ASR-1.7B")
    p.add_argument(
        "--llm-model-id",
        default="Qwen/Qwen3.6-27B",
        help="Judge weights: Qwen/Qwen3.8-27B if the vLLM build can load it; else keep Qwen3.6-27B",
    )
    p.add_argument("--deepseek-model-id", default="deepseek-ai/DeepSeek-V2.5")
    p.add_argument(
        "--no-deepseek-fallback",
        action="store_true",
        help="Disable DeepSeek judge fallback after Qwen retries",
    )
    p.add_argument("--enable-real", action="store_true", help="Allow real runners to load models")
    p.add_argument(
        "--llm-backend",
        choices=["transformers", "vllm", "vllm_engine"],
        default="transformers",
        help=(
            "LLM backend: transformers (slow HF generate); "
            "vllm (OpenAI HTTP server); "
            "vllm_engine (in-process vllm.LLM — recommended on Ascend 910B)"
        ),
    )
    p.add_argument(
        "--llm-base-url",
        default=None,
        help="Required for --llm-backend vllm (HTTP). Unused for vllm_engine.",
    )
    p.add_argument(
        "--llm-api-key",
        default=None,
        help="Optional Bearer token for OpenAI-compat server",
    )
    p.add_argument(
        "--llm-timeout-s",
        type=float,
        default=300.0,
        help="HTTP timeout seconds for --llm-backend vllm",
    )
    p.add_argument(
        "--deepseek-base-url",
        default=None,
        help="Optional separate OpenAI-compat URL for DeepSeek fallback (else reuse --llm-base-url)",
    )
    p.add_argument(
        "--pass-a-batch-size",
        type=int,
        default=1,
        help="Pass A micro-batch size (>1 enables batched vllm_engine.generate / HTTP concurrency)",
    )
    p.add_argument(
        "--pass-b-batch-size",
        type=int,
        default=1,
        help=(
            "Pass B micro-batch size (1 = sequential, later turns see earlier Pass B edits; "
            ">1 = snapshot meeting_draft + judge_many for A/B speed vs quality)"
        ),
    )
    p.add_argument(
        "--polish-batch-size",
        type=int,
        default=1,
        help=(
            "Polish micro-batch size (1 = sequential, later turns see earlier polish edits; "
            ">1 = snapshot neighbors + polish_many). Independent of Pass A/B."
        ),
    )
    p.add_argument(
        "--glossary",
        default=None,
        help="Seed glossary JSON for --stage publish (terms + optional latex)",
    )
    p.add_argument(
        "--publish-batch-size",
        type=int,
        default=1,
        help="Publish meeting pack size (1 = one meeting at a time)",
    )
    p.add_argument(
        "--no-publish-eval",
        action="store_true",
        help="Skip the publish faithfulness LLM judge",
    )
    p.add_argument(
        "--no-publish-eval-thinking",
        action="store_true",
        help="Run the publish quality judge with thinking off",
    )
    p.add_argument(
        "--vllm-tp-size",
        type=int,
        default=1,
        help="tensor_parallel_size for vllm_engine (use 2 for two NPUs)",
    )
    p.add_argument(
        "--vllm-gpu-memory-utilization",
        type=float,
        default=0.90,
        help="gpu_memory_utilization for vllm_engine (lower if KV cache OOM, e.g. 0.85)",
    )
    p.add_argument(
        "--vllm-max-model-len",
        type=int,
        default=None,
        help="Optional max_model_len (lower e.g. 4096/8192 frees KV cache memory)",
    )
    p.add_argument(
        "--vllm-dtype",
        default="auto",
        help="vllm_engine dtype: auto|bf16|bfloat16|fp16|float16|fp32 (bf16 recommended on 910B)",
    )
    p.add_argument(
        "--vllm-enforce-eager",
        action="store_true",
        default=True,
        help="Pass enforce_eager=True to vllm.LLM (default on; stabilizes Ascend bring-up)",
    )
    p.add_argument(
        "--no-vllm-enforce-eager",
        action="store_true",
        help="Disable enforce_eager for vllm_engine",
    )
    p.add_argument(
        "--vllm-use-v1",
        action="store_true",
        help="Use vLLM V1 engine (default off: VLLM_USE_V1=0 avoids OpenMP Invalid thread pool crash)",
    )
    p.add_argument(
        "--llm-enable-thinking",
        action="store_true",
        help="Allow Qwen3-style thinking/CoT (default: off — JSON-only for ASR judge speed/validity)",
    )


def _resolve_backend(args: argparse.Namespace) -> str | None:
    backend = args.backend or ("mock" if args.mock or not args.enable_real else "real")
    if backend == "real" and not args.enable_real:
        print(
            "Real backend requires --enable-real (may download/load weights). "
            "Use --mock for offline tests.",
            file=sys.stderr,
        )
        return None
    if getattr(args, "llm_backend", "transformers") == "vllm" and not getattr(args, "llm_base_url", None):
        if backend == "real" and str(args.stage).lower() in {
            "all",
            "pass_a",
            "pass_b",
            "llm",
            "polish",
            "publish",
        }:
            print(
                "--llm-backend vllm (HTTP) requires --llm-base-url. "
                "For in-process vllm.LLM on Ascend, use --llm-backend vllm_engine instead.",
                file=sys.stderr,
            )
            return None
    return backend


def _vllm_flags(args: argparse.Namespace) -> dict:
    enforce = True
    if getattr(args, "no_vllm_enforce_eager", False):
        enforce = False
    use_v1: bool | None = False
    if getattr(args, "vllm_use_v1", False):
        use_v1 = True
    return {
        "vllm_tp_size": int(args.vllm_tp_size),
        "vllm_gpu_memory_utilization": float(args.vllm_gpu_memory_utilization),
        "vllm_max_model_len": args.vllm_max_model_len,
        "vllm_dtype": str(args.vllm_dtype),
        "vllm_enforce_eager": enforce,
        "vllm_use_v1": use_v1,
        "llm_enable_thinking": bool(getattr(args, "llm_enable_thinking", False)),
    }


def _pipeline_config(args: argparse.Namespace) -> PipelineConfig:
    glossary = None
    raw_glossary = getattr(args, "glossary", None)
    if raw_glossary:
        glossary = load_glossary(Path(raw_glossary))
    return PipelineConfig(
        max_asr_seconds=float(args.max_asr_seconds),
        pass_a_batch_size=max(1, int(args.pass_a_batch_size)),
        pass_b_batch_size=max(1, int(getattr(args, "pass_b_batch_size", 1))),
        polish_batch_size=max(1, int(getattr(args, "polish_batch_size", 1))),
        publish_batch_size=max(1, int(getattr(args, "publish_batch_size", 1))),
        publish_eval=not bool(getattr(args, "no_publish_eval", False)),
        publish_eval_thinking=not bool(getattr(args, "no_publish_eval_thinking", False)),
        glossary=glossary,
    )


def _cmd_run(args: argparse.Namespace) -> int:
    backend = _resolve_backend(args)
    if backend is None:
        return 2

    work_dir = Path(args.work_dir)
    work_dir.mkdir(parents=True, exist_ok=True)
    cfg = _pipeline_config(args)
    stage = str(args.stage).lower()
    asr_models = [m.strip().lower() for m in str(args.asr_models).split(",") if m.strip()]
    mock_hyps = Path(args.mock_hyps) if args.mock_hyps else None

    asr, llm, fallback_judge = build_runners(
        backend=backend,
        stage=stage,
        work_dir=work_dir,
        enable_real=bool(args.enable_real),
        mock_hyps=mock_hyps,
        qwen_model_id=args.qwen_model_id,
        llm_model_id=args.llm_model_id,
        deepseek_model_id=args.deepseek_model_id,
        no_deepseek_fallback=bool(args.no_deepseek_fallback),
        llm_backend=args.llm_backend,
        llm_base_url=args.llm_base_url,
        llm_api_key=args.llm_api_key,
        llm_timeout_s=float(args.llm_timeout_s),
        deepseek_base_url=args.deepseek_base_url,
        **_vllm_flags(args),
    )

    result = run_pipeline(
        input_json=Path(args.input),
        audio_path=Path(args.audio),
        work_dir=work_dir,
        asr_runner=asr,
        llm_judge=llm,
        config=cfg,
        hotwords=load_hotwords(args.hotwords),
        fallback_judge=fallback_judge,
        stage=stage,
        asr_models=asr_models,
    )
    payload = {
        "ok": True,
        "backend": backend,
        "stage": stage,
        "llm_backend": args.llm_backend,
        "n_turns": result.get("n_turns"),
        "n_units": result.get("n_units"),
    }
    if result.get("final_path") is not None:
        payload["final"] = str(result["final_path"])
    if result.get("draft_path") is not None:
        payload["draft"] = str(result["draft_path"])
    if result.get("draft_merged_path") is not None:
        payload["draft_merged"] = str(result["draft_merged_path"])
    if result.get("final_merged_path") is not None:
        payload["final_merged"] = str(result["final_merged_path"])
    if result.get("stats_path") is not None:
        payload["pass_stats"] = str(result["stats_path"])
    if result.get("llm_log_path") is not None:
        payload["llm_log"] = str(result["llm_log_path"])
    if result.get("polished_path") is not None:
        payload["polished"] = str(result["polished_path"])
    if result.get("published_path") is not None:
        payload["published"] = str(result["published_path"])
    if result.get("transcript_path") is not None:
        payload["transcript"] = str(result["transcript_path"])
    if result.get("glossary_path") is not None:
        payload["glossary"] = str(result["glossary_path"])
    if result.get("asr_hypotheses_path") is not None:
        payload["asr_hypotheses"] = str(result["asr_hypotheses_path"])
    if result.get("asr_models") is not None:
        payload["asr_models"] = result["asr_models"]
    print(json.dumps(payload, ensure_ascii=False))
    return 0


def _cmd_run_batch(args: argparse.Namespace) -> int:
    backend = _resolve_backend(args)
    if backend is None:
        return 2

    datasets = None
    if args.datasets:
        datasets = [d.strip() for d in str(args.datasets).split(",") if d.strip()]
    asr_models = [m.strip().lower() for m in str(args.asr_models).split(",") if m.strip()]
    cfg = _pipeline_config(args)
    mock_hyps = Path(args.mock_hyps) if args.mock_hyps else None

    summary = run_batch(
        wav_benchmark=Path(args.wav_benchmark),
        mode_c_benchmark=Path(args.mode_c_benchmark),
        work_root=Path(args.work_root),
        backend=backend,
        stage=str(args.stage).lower(),
        asr_models=asr_models,
        hotwords=load_hotwords(args.hotwords),
        datasets=datasets,
        limit=args.limit,
        dry_run=bool(args.dry_run),
        enable_real=bool(args.enable_real),
        mock_hyps=mock_hyps,
        config=cfg,
        qwen_model_id=args.qwen_model_id,
        llm_model_id=args.llm_model_id,
        deepseek_model_id=args.deepseek_model_id,
        no_deepseek_fallback=bool(args.no_deepseek_fallback),
        continue_on_error=not bool(args.fail_fast),
        llm_backend=args.llm_backend,
        llm_base_url=args.llm_base_url,
        llm_api_key=args.llm_api_key,
        llm_timeout_s=float(args.llm_timeout_s),
        deepseek_base_url=args.deepseek_base_url,
        **_vllm_flags(args),
    )
    print(
        json.dumps(
            {
                "ok": summary["n_error"] == 0,
                "backend": summary["backend"],
                "stage": summary["stage"],
                "llm_backend": summary.get("llm_backend"),
                "n_paired": summary["n_paired"],
                "n_ok": summary["n_ok"],
                "n_skip": summary["n_skip"],
                "n_error": summary["n_error"],
                "summary": str(Path(args.work_root) / "batch_summary.json"),
            },
            ensure_ascii=False,
        )
    )
    return 1 if summary["n_error"] else 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="stage2-asr", description="Stage-2 multi-ASR + LLM fusion")
    sub = parser.add_subparsers(dest="cmd", required=True)

    run_p = sub.add_parser("run", help="Run Stage-2 on a single Mode-C + wav pair")
    run_p.add_argument("--input", required=True, help="Path to mode_c.json")
    run_p.add_argument("--audio", required=True, help="Path to prepared wav")
    run_p.add_argument("--work-dir", required=True, help="Output / cache directory")
    _add_common_run_args(run_p)

    batch_p = sub.add_parser(
        "run-batch",
        help="Run Stage-2 over benchmark/*/Audio wavs paired with Mode-C JSONs",
    )
    batch_p.add_argument(
        "--wav-benchmark",
        required=True,
        help="Root .../benchmark containing {dataset}/Audio/*.wav",
    )
    batch_p.add_argument(
        "--mode-c-benchmark",
        required=True,
        help="Root .../benchmark containing {dataset}/Audio/{stem}/mode_c.json",
    )
    batch_p.add_argument(
        "--work-root",
        required=True,
        help="Output root; writes work-root/{dataset}/{stem}/ plus batch_summary.json",
    )
    batch_p.add_argument(
        "--datasets",
        default=None,
        help="Optional comma-separated dataset names under benchmark (default: all)",
    )
    batch_p.add_argument("--limit", type=int, default=None, help="Optional max number of paired samples")
    batch_p.add_argument(
        "--dry-run",
        action="store_true",
        help="Only discover pairs and write batch_summary.json (no inference)",
    )
    batch_p.add_argument(
        "--fail-fast",
        action="store_true",
        help="Stop on first sample error (default: continue and record errors)",
    )
    _add_common_run_args(batch_p)

    args = parser.parse_args(argv)
    if args.cmd == "run":
        return _cmd_run(args)
    if args.cmd == "run-batch":
        return _cmd_run_batch(args)
    parser.error(f"unknown command {args.cmd}")
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
