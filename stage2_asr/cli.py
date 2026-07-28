from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from stage2_asr.pipeline import run_pipeline
from stage2_asr.runners.ensemble import EnsembleAsrRunner
from stage2_asr.runners.firered_asr2s import FireRedAsr2sConfig, FireRedAsr2sRunner
from stage2_asr.runners.llm_deepseek import DeepSeekLlmJudge
from stage2_asr.runners.llm_qwen36 import Qwen36LlmJudge
from stage2_asr.runners.mock_asr import MockAsrRunner
from stage2_asr.runners.mock_llm import MockLlmJudge
from stage2_asr.runners.qwen3_asr import Qwen3AsrRunner
from stage2_asr.types import PipelineConfig


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="stage2-asr", description="Stage-2 multi-ASR + LLM fusion")
    sub = parser.add_subparsers(dest="cmd", required=True)

    run_p = sub.add_parser("run", help="Run Stage-2 on a Mode-C fusion JSON")
    run_p.add_argument("--input", required=True, help="Path to mode_c.json")
    run_p.add_argument("--audio", required=True, help="Path to prepared wav")
    run_p.add_argument("--work-dir", required=True, help="Output / cache directory")
    run_p.add_argument("--mock", action="store_true", help="Use mock ASR/LLM (no model weights)")
    run_p.add_argument(
        "--backend",
        choices=["mock", "real"],
        default=None,
        help="Backend selection (default: mock if --mock else real)",
    )
    run_p.add_argument("--mock-hyps", default=None, help="Optional mock hypothesis fixture JSON")
    run_p.add_argument("--hotwords", default=None, help="Optional hotword list JSON")
    run_p.add_argument(
        "--stage",
        default="all",
        choices=["all", "asr", "pass_a", "pass_b", "llm"],
        help="Execution stage: all | asr | pass_a | pass_b | llm",
    )
    run_p.add_argument(
        "--asr-models",
        default="moss,qwen,firered",
        help="Comma-separated ASR models for ASR stage/cache: moss,qwen,firered",
    )
    run_p.add_argument("--max-asr-seconds", type=float, default=30.0)
    run_p.add_argument("--qwen-model-id", default="Qwen/Qwen3-ASR-1.7B")
    run_p.add_argument("--llm-model-id", default="Qwen/Qwen3.6-27B")
    run_p.add_argument("--deepseek-model-id", default="deepseek-ai/DeepSeek-V2.5")
    run_p.add_argument(
        "--no-deepseek-fallback",
        action="store_true",
        help="Disable DeepSeek judge fallback after Qwen retries",
    )
    run_p.add_argument("--enable-real", action="store_true", help="Allow real runners to load models")

    args = parser.parse_args(argv)
    if args.cmd != "run":
        parser.error(f"unknown command {args.cmd}")

    backend = args.backend or ("mock" if args.mock or not args.enable_real else "real")
    if backend == "real" and not args.enable_real:
        print(
            "Real backend requires --enable-real (may download/load weights). "
            "Use --mock for offline tests.",
            file=sys.stderr,
        )
        return 2

    work_dir = Path(args.work_dir)
    work_dir.mkdir(parents=True, exist_ok=True)
    cfg = PipelineConfig(max_asr_seconds=float(args.max_asr_seconds))

    stage = str(args.stage).lower()
    asr_models = [m.strip().lower() for m in str(args.asr_models).split(",") if m.strip()]
    needs_asr = stage in {"all", "asr"}
    needs_llm = stage in {"all", "pass_a", "pass_b", "llm"}

    fallback_judge = None
    asr = MockAsrRunner()  # lightweight default when stage does not need ASR.
    llm = MockLlmJudge()   # lightweight default when stage does not need LLM.
    if backend == "mock":
        if needs_asr:
            hyp_path = Path(args.mock_hyps) if args.mock_hyps else None
            asr = MockAsrRunner(fixture_path=hyp_path)
        if needs_llm:
            llm = MockLlmJudge()
    else:
        if needs_asr:
            asr = EnsembleAsrRunner(
                Qwen3AsrRunner(enabled=True, model_id=args.qwen_model_id, work_dir=work_dir),
                FireRedAsr2sRunner(enabled=True, config=FireRedAsr2sConfig(vad=False, lid=True, punc=True)),
            )
        if needs_llm:
            llm = Qwen36LlmJudge(enabled=True, model_id=args.llm_model_id, temperature=0.1)
            if not args.no_deepseek_fallback:
                fallback_judge = DeepSeekLlmJudge(
                    enabled=True,
                    model_id=args.deepseek_model_id,
                    temperature=0.1,
                )

    hotwords: list[str] = []
    if args.hotwords:
        hotwords = json.loads(Path(args.hotwords).read_text(encoding="utf-8"))

    result = run_pipeline(
        input_json=Path(args.input),
        audio_path=Path(args.audio),
        work_dir=work_dir,
        asr_runner=asr,
        llm_judge=llm,
        config=cfg,
        hotwords=hotwords,
        fallback_judge=fallback_judge,
        stage=stage,
        asr_models=asr_models,
    )
    payload = {
        "ok": True,
        "backend": backend,
        "stage": stage,
        "n_turns": result.get("n_turns"),
        "n_units": result.get("n_units"),
    }
    if result.get("final_path") is not None:
        payload["final"] = str(result["final_path"])
    if result.get("draft_path") is not None:
        payload["draft"] = str(result["draft_path"])
    if result.get("stats_path") is not None:
        payload["pass_stats"] = str(result["stats_path"])
    if result.get("asr_hypotheses_path") is not None:
        payload["asr_hypotheses"] = str(result["asr_hypotheses_path"])
    if result.get("asr_models") is not None:
        payload["asr_models"] = result["asr_models"]
    print(json.dumps(payload, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
