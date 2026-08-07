"""Batch discovery and execution over benchmark/*/Audio layouts."""

from __future__ import annotations

import json
import sys
import traceback
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

from stage2_asr.pipeline import run_pipeline
from stage2_asr.types import PipelineConfig


def _log(msg: str) -> None:
    print(msg, file=sys.stderr, flush=True)


@dataclass(frozen=True)
class SamplePair:
    dataset: str
    stem: str
    wav: Path
    mode_c: Path

    @property
    def sample_id(self) -> str:
        return f"{self.dataset}/{self.stem}"

    def work_dir(self, work_root: Path) -> Path:
        return work_root / self.dataset / self.stem

    def to_dict(self) -> dict[str, str]:
        return {
            "dataset": self.dataset,
            "stem": self.stem,
            "sample_id": self.sample_id,
            "wav": str(self.wav),
            "mode_c": str(self.mode_c),
        }


def discover_benchmark_pairs(
    wav_benchmark: Path,
    mode_c_benchmark: Path,
    *,
    datasets: Iterable[str] | None = None,
) -> tuple[list[SamplePair], list[dict[str, Any]]]:
    """
    Pair wavs and Mode-C JSONs for layouts:

      {wav_benchmark}/{dataset}/Audio/{stem}.wav
      {mode_c_benchmark}/{dataset}/Audio/{stem}/mode_c.json
    """
    wav_benchmark = Path(wav_benchmark)
    mode_c_benchmark = Path(mode_c_benchmark)
    allow = {d.strip() for d in datasets} if datasets else None
    if allow is not None:
        allow = {d for d in allow if d}

    pairs: list[SamplePair] = []
    skips: list[dict[str, Any]] = []

    if not wav_benchmark.is_dir():
        raise FileNotFoundError(f"wav benchmark root not found: {wav_benchmark}")

    for dataset_dir in sorted(p for p in wav_benchmark.iterdir() if p.is_dir()):
        dataset = dataset_dir.name
        if allow is not None and dataset not in allow:
            continue
        audio_dir = dataset_dir / "Audio"
        if not audio_dir.is_dir():
            continue
        for wav in sorted(audio_dir.glob("*.wav")):
            stem = wav.stem
            mode_c = mode_c_benchmark / dataset / "Audio" / stem / "mode_c.json"
            if not mode_c.is_file():
                skips.append(
                    {
                        "dataset": dataset,
                        "stem": stem,
                        "wav": str(wav),
                        "mode_c": str(mode_c),
                        "reason": "missing_mode_c",
                    }
                )
                continue
            pairs.append(
                SamplePair(
                    dataset=dataset,
                    stem=stem,
                    wav=wav.resolve(),
                    mode_c=mode_c.resolve(),
                )
            )
    return pairs, skips


def build_runners(
    *,
    backend: str,
    stage: str,
    work_dir: Path,
    enable_real: bool,
    mock_hyps: Path | None,
    qwen_model_id: str,
    llm_model_id: str,
    deepseek_model_id: str,
    no_deepseek_fallback: bool,
):
    """Construct ASR/LLM runners once for a batch (reuse across samples)."""
    from stage2_asr.runners.ensemble import EnsembleAsrRunner
    from stage2_asr.runners.firered_asr2s import FireRedAsr2sConfig, FireRedAsr2sRunner
    from stage2_asr.runners.llm_deepseek import DeepSeekLlmJudge
    from stage2_asr.runners.llm_qwen36 import Qwen36LlmJudge
    from stage2_asr.runners.mock_asr import MockAsrRunner
    from stage2_asr.runners.mock_llm import MockLlmJudge
    from stage2_asr.runners.qwen3_asr import Qwen3AsrRunner

    needs_asr = stage in {"all", "asr"}
    needs_llm = stage in {"all", "pass_a", "pass_b", "llm"}
    fallback_judge = None
    asr = MockAsrRunner()
    llm = MockLlmJudge()

    if backend == "mock":
        if needs_asr:
            asr = MockAsrRunner(fixture_path=mock_hyps)
        if needs_llm:
            llm = MockLlmJudge()
        return asr, llm, fallback_judge

    if not enable_real:
        raise ValueError("Real backend requires enable_real=True")

    if needs_asr:
        asr = EnsembleAsrRunner(
            Qwen3AsrRunner(enabled=True, model_id=qwen_model_id, work_dir=work_dir),
            FireRedAsr2sRunner(
                enabled=True,
                config=FireRedAsr2sConfig(vad=False, lid=True, punc=True),
            ),
        )
    if needs_llm:
        llm = Qwen36LlmJudge(enabled=True, model_id=llm_model_id, temperature=0.1)
        if not no_deepseek_fallback:
            fallback_judge = DeepSeekLlmJudge(
                enabled=True,
                model_id=deepseek_model_id,
                temperature=0.1,
            )
    return asr, llm, fallback_judge


def run_batch(
    *,
    wav_benchmark: Path,
    mode_c_benchmark: Path,
    work_root: Path,
    backend: str = "mock",
    stage: str = "all",
    asr_models: list[str] | None = None,
    hotwords: list[str] | None = None,
    datasets: list[str] | None = None,
    limit: int | None = None,
    dry_run: bool = False,
    enable_real: bool = False,
    mock_hyps: Path | None = None,
    config: PipelineConfig | None = None,
    qwen_model_id: str = "Qwen/Qwen3-ASR-1.7B",
    llm_model_id: str = "Qwen/Qwen3.6-27B",
    deepseek_model_id: str = "deepseek-ai/DeepSeek-V2.5",
    no_deepseek_fallback: bool = False,
    continue_on_error: bool = True,
) -> dict[str, Any]:
    """Discover pairs and run Stage-2 per sample under work_root/{dataset}/{stem}/."""
    cfg = config or PipelineConfig()
    hotwords = hotwords or []
    asr_models = asr_models or ["moss", "qwen", "firered"]
    work_root = Path(work_root)
    work_root.mkdir(parents=True, exist_ok=True)

    pairs, skips = discover_benchmark_pairs(
        wav_benchmark,
        mode_c_benchmark,
        datasets=datasets,
    )
    if limit is not None:
        pairs = pairs[: max(0, int(limit))]

    summary: dict[str, Any] = {
        "backend": backend,
        "stage": stage,
        "asr_models": asr_models,
        "wav_benchmark": str(wav_benchmark),
        "mode_c_benchmark": str(mode_c_benchmark),
        "work_root": str(work_root),
        "n_paired": len(pairs),
        "n_skip": len(skips),
        "n_ok": 0,
        "n_error": 0,
        "dry_run": dry_run,
        "skips": skips,
        "results": [],
    }

    if dry_run:
        summary["results"] = [
            {**p.to_dict(), "work_dir": str(p.work_dir(work_root)), "status": "dry_run"}
            for p in pairs
        ]
        _write_summary(work_root, summary)
        _log(f"[batch] dry-run: paired={len(pairs)} skipped={len(skips)}")
        return summary

    # Load runners once; work_dir on Qwen is only used for optional cache hints.
    asr, llm, fallback_judge = build_runners(
        backend=backend,
        stage=stage,
        work_dir=work_root,
        enable_real=enable_real,
        mock_hyps=mock_hyps,
        qwen_model_id=qwen_model_id,
        llm_model_id=llm_model_id,
        deepseek_model_id=deepseek_model_id,
        no_deepseek_fallback=no_deepseek_fallback,
    )

    n_pairs = len(pairs)
    _log(
        f"[batch] start stage={stage} paired={n_pairs} skipped={len(skips)} "
        f"models={asr_models} work_root={work_root}"
    )
    for bi, pair in enumerate(pairs, start=1):
        sample_work = pair.work_dir(work_root)
        sample_work.mkdir(parents=True, exist_ok=True)
        row: dict[str, Any] = {
            **pair.to_dict(),
            "work_dir": str(sample_work),
        }
        _log(f"[batch] {bi}/{n_pairs} {pair.sample_id} begin")
        try:
            result = run_pipeline(
                input_json=pair.mode_c,
                audio_path=pair.wav,
                work_dir=sample_work,
                asr_runner=asr,
                llm_judge=llm,
                config=cfg,
                hotwords=hotwords,
                fallback_judge=fallback_judge,
                stage=stage,
                asr_models=asr_models,
            )
            row["status"] = "ok"
            row["n_turns"] = result.get("n_turns")
            row["n_units"] = result.get("n_units")
            if result.get("final_path") is not None:
                row["final"] = str(result["final_path"])
            if result.get("draft_path") is not None:
                row["draft"] = str(result["draft_path"])
            if result.get("asr_hypotheses_path") is not None:
                row["asr_hypotheses"] = str(result["asr_hypotheses_path"])
            summary["n_ok"] += 1
            _log(f"[batch] {bi}/{n_pairs} {pair.sample_id} ok")
        except Exception as exc:  # noqa: BLE001
            row["status"] = "error"
            row["error"] = str(exc)
            row["traceback"] = traceback.format_exc(limit=5)
            summary["n_error"] += 1
            _log(f"[batch] {bi}/{n_pairs} {pair.sample_id} error: {exc}")
            if not continue_on_error:
                summary["results"].append(row)
                _write_summary(work_root, summary)
                raise
        summary["results"].append(row)

    _write_summary(work_root, summary)
    _log(
        f"[batch] done ok={summary['n_ok']} error={summary['n_error']} "
        f"skip={summary['n_skip']} summary={work_root / 'batch_summary.json'}"
    )
    return summary


def _write_summary(work_root: Path, summary: dict[str, Any]) -> None:
    path = work_root / "batch_summary.json"
    path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
