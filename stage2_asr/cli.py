from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

from stage2_asr.batch import build_runners, launch_npu_shards, run_batch
from stage2_asr.glossary_union import union_corpus_glossary, write_corpus_glossary
from stage2_asr.hotwords import load_hotwords
from stage2_asr.model_paths import (
    DEFAULT_FIRERED_ASR_MODEL_DIR,
    DEFAULT_FIRERED_LID_MODEL_DIR,
    DEFAULT_FIRERED_PUNC_MODEL_DIR,
    DEFAULT_QWEN_MODEL_ID,
    resolve_firered_model_dirs,
    resolve_qwen_model_id,
)
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
        help="Execution stage: all | asr | pass_a | pass_b | llm (pass_a+pass_b+polish+publish) | polish | publish",
    )
    p.add_argument(
        "--asr-models",
        default="moss,qwen,firered",
        help="Comma-separated ASR models for ASR stage/cache: moss,qwen,firered",
    )
    p.add_argument("--max-asr-seconds", type=float, default=30.0)
    p.add_argument(
        "--qwen-model-id",
        default=None,
        help=(
            "Qwen3-ASR weights (local dir or HuggingFace id). "
            f"Else STAGE2_QWEN_MODEL_ID, then QWEN_MODEL_ID, then {DEFAULT_QWEN_MODEL_ID}"
        ),
    )
    p.add_argument(
        "--firered-asr-model-dir",
        default=None,
        help=(
            "FireRed ASR weight dir. Else STAGE2_FIRERED_ASR_MODEL_DIR / FIRERED_ASR_MODEL_DIR, "
            f"then {DEFAULT_FIRERED_ASR_MODEL_DIR}"
        ),
    )
    p.add_argument(
        "--firered-lid-model-dir",
        default=None,
        help=(
            "FireRed LID weight dir. Else STAGE2_FIRERED_LID_MODEL_DIR / FIRERED_LID_MODEL_DIR, "
            f"then {DEFAULT_FIRERED_LID_MODEL_DIR}"
        ),
    )
    p.add_argument(
        "--firered-punc-model-dir",
        default=None,
        help=(
            "FireRed Punc weight dir. Else STAGE2_FIRERED_PUNC_MODEL_DIR / FIRERED_PUNC_MODEL_DIR, "
            f"then {DEFAULT_FIRERED_PUNC_MODEL_DIR}"
        ),
    )
    p.add_argument(
        "--llm-model-id",
        default="Qwen/Qwen3.6-27B",
        help="Judge weights: Qwen/Qwen3.8-27B if the vLLM build can load it; else keep Qwen3.6-27B",
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
        help="Optional Bearer token (else STAGE2_LLM_API_KEY, then OPENAI_API_KEY)",
    )
    p.add_argument(
        "--llm-timeout-s",
        type=float,
        default=300.0,
        help="HTTP timeout seconds for --llm-backend vllm; also caps transformers generate",
    )
    p.add_argument(
        "--llm-retry-backoff-s",
        type=float,
        default=0.25,
        help="Sleep seconds before LLM retries (exponential: backoff, 2x, 4x, ...). 0 disables",
    )
    p.add_argument(
        "--force-refresh",
        action="store_true",
        help=(
            "ASR: rebuild units and skip asr_cache. "
            "pass_a/pass_b/llm: allow stale asr_units.json after an input fingerprint mismatch"
        ),
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
        help=(
            "Seed glossary JSON for --stage publish; unioned with work_dir/glossary.json "
            "(CLI covers the same surface)"
        ),
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
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Pass enforce_eager to vllm.LLM (default on; --no-vllm-enforce-eager to disable)",
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
    p.add_argument(
        "--llm-log-mode",
        choices=["full", "meta", "off"],
        default="meta",
        help="llm_infer.jsonl: meta (default, no prompt/response bodies), full, or off",
    )
    p.add_argument("--neighbor-max-turns", type=int, default=20)
    p.add_argument("--neighbor-window-seconds", type=float, default=600.0)
    p.add_argument(
        "--neighbor-char-budget",
        type=int,
        default=8192,
        help="Max neighbor-draft characters in LLM prompts (approx 0.5 token/char)",
    )
    p.add_argument(
        "--hotword-prompt-chars",
        type=int,
        default=4000,
        help="Max JSON characters of hotwords sent to the LLM (aliases still use the full list)",
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
    enforce = bool(getattr(args, "vllm_enforce_eager", True))
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


def resolve_llm_api_key(cli_value: str | None) -> str | None:
    """CLI flag wins, then STAGE2_LLM_API_KEY, then OPENAI_API_KEY. Never log the value."""
    if cli_value:
        return str(cli_value)
    return os.environ.get("STAGE2_LLM_API_KEY") or os.environ.get("OPENAI_API_KEY") or None


def _resolved_asr_kwargs(args: argparse.Namespace) -> dict:
    asr_dir, lid_dir, punc_dir = resolve_firered_model_dirs(
        asr_model_dir=getattr(args, "firered_asr_model_dir", None),
        lid_model_dir=getattr(args, "firered_lid_model_dir", None),
        punc_model_dir=getattr(args, "firered_punc_model_dir", None),
    )
    return {
        "qwen_model_id": resolve_qwen_model_id(getattr(args, "qwen_model_id", None)),
        "firered_asr_model_dir": asr_dir,
        "firered_lid_model_dir": lid_dir,
        "firered_punc_model_dir": punc_dir,
    }


def _pipeline_config(args: argparse.Namespace) -> PipelineConfig:
    glossary = None
    raw_glossary = getattr(args, "glossary", None)
    if raw_glossary:
        glossary = load_glossary(Path(raw_glossary))
    neighbor_char_budget = int(getattr(args, "neighbor_char_budget", 8192))
    max_len = getattr(args, "vllm_max_model_len", None)
    if max_len:
        derived = max(512, (int(max_len) - 1024) * 2)
        neighbor_char_budget = min(neighbor_char_budget, derived)
    return PipelineConfig(
        max_asr_seconds=float(args.max_asr_seconds),
        pass_a_batch_size=max(1, int(args.pass_a_batch_size)),
        pass_b_batch_size=max(1, int(getattr(args, "pass_b_batch_size", 1))),
        polish_batch_size=max(1, int(getattr(args, "polish_batch_size", 1))),
        publish_batch_size=max(1, int(getattr(args, "publish_batch_size", 1))),
        publish_eval=not bool(getattr(args, "no_publish_eval", False)),
        publish_eval_thinking=not bool(getattr(args, "no_publish_eval_thinking", False)),
        glossary=glossary,
        llm_retry_backoff_s=float(getattr(args, "llm_retry_backoff_s", 0.0)),
        force_refresh=bool(getattr(args, "force_refresh", False)),
        neighbor_max_turns=max(0, int(getattr(args, "neighbor_max_turns", 20))),
        neighbor_window_seconds=float(getattr(args, "neighbor_window_seconds", 600.0)),
        neighbor_char_budget=max(0, neighbor_char_budget),
        hotword_prompt_chars=max(0, int(getattr(args, "hotword_prompt_chars", 4000))),
        llm_log_mode=str(getattr(args, "llm_log_mode", "meta") or "meta"),
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
        llm_model_id=args.llm_model_id,
        llm_backend=args.llm_backend,
        llm_base_url=args.llm_base_url,
        llm_api_key=resolve_llm_api_key(args.llm_api_key),
        llm_timeout_s=float(args.llm_timeout_s),
        **_resolved_asr_kwargs(args),
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


_LAUNCHER_VALUE_FLAGS = {"--devices", "--npu-per-job", "--shard"}


def _parse_devices(raw: str | None) -> list[int]:
    if raw is None or str(raw).strip() == "":
        return []
    return [int(part.strip()) for part in str(raw).split(",") if part.strip()]


def _child_batch_argv(argv: list[str], *, npu_per_job: int) -> list[str]:
    """Drop launcher-only flags so children do not re-spawn; default TP to npu_per_job."""
    out: list[str] = []
    i = 0
    saw_tp = False
    while i < len(argv):
        tok = argv[i]
        key = tok.split("=", 1)[0]
        if key in _LAUNCHER_VALUE_FLAGS:
            i += 1 if "=" in tok else 2
            continue
        if key == "--vllm-tp-size":
            saw_tp = True
        out.append(tok)
        i += 1
    if not saw_tp:
        out.extend(["--vllm-tp-size", str(npu_per_job)])
    return out


def _emit_batch_stdout(summary: dict, work_root: Path) -> int:
    payload = {
        "ok": int(summary.get("n_error") or 0) == 0 and int(summary.get("launcher_exit") or 0) == 0,
        "backend": summary.get("backend"),
        "stage": summary.get("stage"),
        "llm_backend": summary.get("llm_backend"),
        "n_paired": summary.get("n_paired"),
        "n_ok": summary.get("n_ok"),
        "n_cached": summary.get("n_cached", 0),
        "n_skip": summary.get("n_skip"),
        "n_error": summary.get("n_error"),
        "summary": str(work_root / "batch_summary.json"),
    }
    if "n_shards" in summary:
        payload["n_shards"] = summary["n_shards"]
    print(json.dumps(payload, ensure_ascii=False))
    if int(summary.get("launcher_exit") or 0) or int(summary.get("n_error") or 0):
        return 1
    return 0


def _cmd_run_batch(args: argparse.Namespace, argv: list[str] | None = None) -> int:
    backend = _resolve_backend(args)
    if backend is None:
        return 2

    devices = _parse_devices(getattr(args, "devices", None))
    npu_per_job = max(1, int(getattr(args, "npu_per_job", 2) or 2))
    shard = getattr(args, "shard", None)
    work_root = Path(args.work_root)

    if devices and not shard:
        n_jobs = len(devices) // npu_per_job
        if n_jobs < 1:
            print(
                f"need at least {npu_per_job} devices for --npu-per-job {npu_per_job}, got {devices}",
                file=sys.stderr,
            )
            return 2
        if n_jobs >= 2:
            leftover = devices[n_jobs * npu_per_job :]
            if leftover:
                print(f"[batch] ignoring leftover devices {leftover}", file=sys.stderr)
            merged = launch_npu_shards(
                devices=devices,
                npu_per_job=npu_per_job,
                work_root=work_root,
                child_argv=_child_batch_argv(list(argv or []), npu_per_job=npu_per_job),
            )
            return _emit_batch_stdout(merged, work_root)
        os.environ["ASCEND_RT_VISIBLE_DEVICES"] = ",".join(
            str(d) for d in devices[:npu_per_job]
        )

    datasets = None
    if args.datasets:
        datasets = [d.strip() for d in str(args.datasets).split(",") if d.strip()]
    asr_models = [m.strip().lower() for m in str(args.asr_models).split(",") if m.strip()]
    cfg = _pipeline_config(args)
    mock_hyps = Path(args.mock_hyps) if args.mock_hyps else None

    summary = run_batch(
        wav_benchmark=Path(args.wav_benchmark),
        mode_c_benchmark=Path(args.mode_c_benchmark),
        work_root=work_root,
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
        llm_model_id=args.llm_model_id,
        continue_on_error=not bool(args.fail_fast),
        llm_backend=args.llm_backend,
        llm_base_url=args.llm_base_url,
        llm_api_key=resolve_llm_api_key(args.llm_api_key),
        llm_timeout_s=float(args.llm_timeout_s),
        skip_existing=bool(getattr(args, "skip_existing", True)),
        sample_workers=max(1, int(getattr(args, "sample_workers", 1))),
        shard=shard,
        **_resolved_asr_kwargs(args),
        **_vllm_flags(args),
    )
    return _emit_batch_stdout(summary, work_root)


def _cmd_union_glossary(args: argparse.Namespace) -> int:
    work_root = Path(args.work_root)
    if not work_root.is_dir():
        print(f"work-root is not a directory: {work_root}", file=sys.stderr)
        return 2
    out = Path(args.out) if args.out else work_root / "corpus_glossary.json"
    corpus = union_corpus_glossary(
        work_root, context_chars=max(0, int(getattr(args, "context_chars", 80)))
    )
    paths = write_corpus_glossary(corpus, out)
    print(
        json.dumps(
            {
                "ok": True,
                "n_samples": (corpus.get("meta") or {}).get("n_samples"),
                "n_terms": len(corpus.get("terms") or []),
                "n_keywords": len(corpus.get("keywords") or []),
                "n_rare_words": len(corpus.get("rare_words") or []),
                "corpus": str(paths["corpus"]),
                "seed": str(paths["seed"]),
            },
            ensure_ascii=False,
        )
    )
    return 0


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
        help="Run Stage-2 over audio files recursively paired with Mode-C JSONs",
    )
    batch_p.add_argument(
        "--wav-benchmark",
        required=True,
        help="Wav file or directory; directories are scanned recursively for *.wav",
    )
    batch_p.add_argument(
        "--mode-c-benchmark",
        required=True,
        help="mode_c.json file or directory; directories are scanned recursively for mode_c.json",
    )
    batch_p.add_argument(
        "--work-root",
        required=True,
        help="Output root; writes work-root/<relative-audio-path>/ plus batch_summary.json",
    )
    batch_p.add_argument(
        "--datasets",
        default=None,
        help="Optional comma-separated top-level directory names under the audio root (default: all)",
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
    batch_p.add_argument(
        "--skip-existing",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Skip samples whose stage artifacts already exist (default: on; --no-skip-existing to rerun)",
    )
    batch_p.add_argument(
        "--sample-workers",
        type=int,
        default=1,
        help="Process this many samples in parallel (I/O overlap; vllm_engine generate is serialized)",
    )
    batch_p.add_argument(
        "--devices",
        default=None,
        help="Comma-separated NPU ids (e.g. 0,1,2,3,4,5,6,7). Split into concurrent jobs of --npu-per-job cards",
    )
    batch_p.add_argument(
        "--npu-per-job",
        type=int,
        default=2,
        help="NPUs per concurrent shard (default 2). 8 cards → 4 jobs that all run until every wav finishes",
    )
    batch_p.add_argument(
        "--shard",
        default=None,
        help="Process slice i/n of paired samples (set automatically by --devices)",
    )
    _add_common_run_args(batch_p)

    union_p = sub.add_parser(
        "union-glossary",
        help="Merge per-sample glossary.json under a work-root into corpus terms/keywords/rare_words",
    )
    union_p.add_argument(
        "--work-root",
        required=True,
        help="Batch work root that contains sample dirs with glossary.json",
    )
    union_p.add_argument(
        "--out",
        default=None,
        help="Corpus JSON path (default: work-root/corpus_glossary.json). Also writes <stem>.seed.json",
    )
    union_p.add_argument(
        "--context-chars",
        type=int,
        default=80,
        help="Left/right characters of published text stored on each rare_word occurrence",
    )

    argv = list(sys.argv[1:] if argv is None else argv)
    args = parser.parse_args(argv)
    if args.cmd == "run":
        return _cmd_run(args)
    if args.cmd == "run-batch":
        return _cmd_run_batch(args, argv)
    if args.cmd == "union-glossary":
        return _cmd_union_glossary(args)
    parser.error(f"unknown command {args.cmd}")


if __name__ == "__main__":
    raise SystemExit(main())
