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

    fallback_judge = None
    if backend == "mock":
        hyp_path = Path(args.mock_hyps) if args.mock_hyps else None
        asr = MockAsrRunner(fixture_path=hyp_path)
        llm = MockLlmJudge()
    else:
        asr = EnsembleAsrRunner(
            Qwen3AsrRunner(enabled=True, model_id=args.qwen_model_id, work_dir=work_dir),
            FireRedAsr2sRunner(enabled=True, config=FireRedAsr2sConfig(vad=False, lid=True, punc=True)),
        )
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
    )
    print(
        json.dumps(
            {
                "ok": True,
                "backend": backend,
                "final": str(result["final_path"]),
                "n_turns": result["n_turns"],
                "pass_stats": str(result.get("stats_path")),
            },
            ensure_ascii=False,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
