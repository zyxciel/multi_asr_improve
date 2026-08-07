from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from stage2_asr.batch import build_runners, run_batch
from stage2_asr.pipeline import run_pipeline
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
    p.add_argument("--hotwords", default=None, help="Optional hotword list JSON")
    p.add_argument(
        "--stage",
        default="all",
        choices=["all", "asr", "pass_a", "pass_b", "llm"],
        help="Execution stage: all | asr | pass_a | pass_b | llm",
    )
    p.add_argument(
        "--asr-models",
        default="moss,qwen,firered",
        help="Comma-separated ASR models for ASR stage/cache: moss,qwen,firered",
    )
    p.add_argument("--max-asr-seconds", type=float, default=30.0)
    p.add_argument("--qwen-model-id", default="Qwen/Qwen3-ASR-1.7B")
    p.add_argument("--llm-model-id", default="Qwen/Qwen3.6-27B")
    p.add_argument("--deepseek-model-id", default="deepseek-ai/DeepSeek-V2.5")
    p.add_argument(
        "--no-deepseek-fallback",
        action="store_true",
        help="Disable DeepSeek judge fallback after Qwen retries",
    )
    p.add_argument("--enable-real", action="store_true", help="Allow real runners to load models")
    p.add_argument(
        "--llm-backend",
        choices=["transformers", "vllm"],
        default="transformers",
        help="LLM serving backend: local transformers or OpenAI-compat vLLM/vLLM-Ascend HTTP",
    )
    p.add_argument(
        "--llm-base-url",
        default=None,
        help="OpenAI-compat base URL for --llm-backend vllm (e.g. http://127.0.0.1:8000 or .../v1)",
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
        help="HTTP timeout seconds for vLLM chat completions",
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
        help="Concurrent Pass A LLM calls (use >1 with --llm-backend vllm; Ascend continuous batching)",
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
        if backend == "real" and str(args.stage).lower() in {"all", "pass_a", "pass_b", "llm"}:
            print(
                "--llm-backend vllm requires --llm-base-url "
                "(OpenAI-compatible server, e.g. vLLM-Ascend on Ascend 910B).",
                file=sys.stderr,
            )
            return None
    return backend


def _load_hotwords(path: str | None) -> list[str]:
    if not path:
        return []
    return json.loads(Path(path).read_text(encoding="utf-8"))


def _pipeline_config(args: argparse.Namespace) -> PipelineConfig:
    return PipelineConfig(
        max_asr_seconds=float(args.max_asr_seconds),
        pass_a_batch_size=max(1, int(args.pass_a_batch_size)),
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
    )

    result = run_pipeline(
        input_json=Path(args.input),
        audio_path=Path(args.audio),
        work_dir=work_dir,
        asr_runner=asr,
        llm_judge=llm,
        config=cfg,
        hotwords=_load_hotwords(args.hotwords),
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
    if result.get("stats_path") is not None:
        payload["pass_stats"] = str(result["stats_path"])
    if result.get("llm_log_path") is not None:
        payload["llm_log"] = str(result["llm_log_path"])
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
        hotwords=_load_hotwords(args.hotwords),
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
