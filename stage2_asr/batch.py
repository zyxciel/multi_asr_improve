"""Batch discovery and execution over audio + Mode-C trees."""

from __future__ import annotations

import json
import os
import subprocess
import sys
import traceback
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

from stage2_asr.pipeline import run_pipeline
from stage2_asr.types import PipelineConfig

_AUDIO_SUFFIXES = {".wav"}


def _log(msg: str) -> None:
    print(msg, file=sys.stderr, flush=True)


def _strip_audio_parts(rel: str) -> str:
    parts = [p for p in Path(rel).parts if p and p not in {".", ".."} and p.lower() != "audio"]
    return "/".join(parts)


def _audio_rel(wav: Path, wav_base: Path) -> str:
    try:
        return wav.relative_to(wav_base).with_suffix("").as_posix()
    except ValueError:
        return wav.stem


def _mode_c_rel(mode_c: Path, mode_c_base: Path) -> str:
    try:
        parent = mode_c.parent.relative_to(mode_c_base)
    except ValueError:
        return mode_c.parent.name
    rel = parent.as_posix()
    return "" if rel == "." else rel


def _collect_audio_files(root: Path) -> tuple[Path, list[Path]]:
    root = Path(root)
    if root.is_file():
        if root.suffix.lower() not in _AUDIO_SUFFIXES:
            raise FileNotFoundError(f"audio path is not a wav file or directory: {root}")
        return root.parent.resolve(), [root.resolve()]
    if not root.is_dir():
        raise FileNotFoundError(f"wav benchmark root not found: {root}")
    files = sorted(
        p.resolve()
        for p in root.rglob("*")
        if p.is_file() and p.suffix.lower() in _AUDIO_SUFFIXES
    )
    return root.resolve(), files


def _collect_mode_c_files(root: Path) -> tuple[Path, list[Path]]:
    root = Path(root)
    if root.is_file():
        if root.suffix.lower() != ".json":
            return root.parent.resolve(), []
        return root.parent.resolve(), [root.resolve()]
    if not root.is_dir():
        return root, []
    files = sorted(p.resolve() for p in root.rglob("mode_c.json") if p.is_file())
    return root.resolve(), files


@dataclass(frozen=True)
class SamplePair:
    dataset: str
    stem: str
    wav: Path
    mode_c: Path
    rel: str = ""

    @property
    def sample_id(self) -> str:
        return self.rel or f"{self.dataset}/{self.stem}"

    def work_dir(self, work_root: Path) -> Path:
        if self.rel:
            return work_root.joinpath(*Path(self.rel).parts)
        return work_root / self.dataset / self.stem

    def to_dict(self) -> dict[str, str]:
        return {
            "dataset": self.dataset,
            "stem": self.stem,
            "sample_id": self.sample_id,
            "rel": self.rel,
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
    Recursively pair wavs under ``wav_benchmark`` with Mode-C JSONs under
    ``mode_c_benchmark``.

    A wav at ``{audio_root}/{rel}.wav`` matches (in order):

    1. ``{mode_c_root}/{rel}/mode_c.json``
    2. the unique Mode-C whose parent path equals ``{rel}`` after dropping
       ``Audio/`` path segments (so ``ds/Audio/utt.wav`` pairs with
       ``ds/utt/mode_c.json``)
    3. the other file, when each root is a single file (or each tree has
       exactly one candidate)

    Output work dirs mirror the audio relative path: ``work_root/{rel}/``.
    Either argument may be a file or a directory.
    """
    wav_base, audio_files = _collect_audio_files(Path(wav_benchmark))
    mode_c_base, mode_c_files = _collect_mode_c_files(Path(mode_c_benchmark))
    allow = {d.strip() for d in datasets} if datasets else None
    if allow is not None:
        allow = {d for d in allow if d}

    pairs: list[SamplePair] = []
    skips: list[dict[str, Any]] = []

    mode_c_by_exact: dict[str, Path] = {}
    mode_c_by_stripped: dict[str, list[Path]] = {}
    for mc in mode_c_files:
        rel = _mode_c_rel(mc, mode_c_base)
        mode_c_by_exact[rel] = mc
        mode_c_by_stripped.setdefault(_strip_audio_parts(rel), []).append(mc)

    used_mode_c: set[Path] = set()

    def _resolve_mode_c(wav_rel: str) -> Path | str | None:
        exact = mode_c_by_exact.get(wav_rel)
        if exact is not None and exact not in used_mode_c:
            return exact
        stripped = _strip_audio_parts(wav_rel)
        cands = [p for p in mode_c_by_stripped.get(stripped, []) if p not in used_mode_c]
        if len(cands) == 1:
            return cands[0]
        if len(cands) > 1:
            return "ambiguous"
        return None

    single_pair = len(audio_files) == 1 and len(mode_c_files) == 1

    for wav in audio_files:
        rel = _audio_rel(wav, wav_base)
        dataset = rel.split("/")[0] if rel else wav.stem
        if allow is not None and dataset not in allow:
            continue
        if single_pair:
            mode_c: Path | str | None = mode_c_files[0]
        else:
            mode_c = _resolve_mode_c(rel)
        stem = wav.stem
        if mode_c == "ambiguous":
            skips.append(
                {
                    "dataset": dataset,
                    "stem": stem,
                    "rel": rel,
                    "wav": str(wav),
                    "mode_c": "",
                    "reason": "ambiguous_mode_c",
                }
            )
            continue
        if mode_c is None:
            skips.append(
                {
                    "dataset": dataset,
                    "stem": stem,
                    "rel": rel,
                    "wav": str(wav),
                    "mode_c": str(mode_c_base / Path(rel) / "mode_c.json"),
                    "reason": "missing_mode_c",
                }
            )
            continue
        used_mode_c.add(mode_c)
        pairs.append(
            SamplePair(
                dataset=dataset,
                stem=stem,
                wav=wav,
                mode_c=mode_c,
                rel=rel,
            )
        )
    return pairs, skips


_STAGE_MARKERS: dict[str, tuple[str, ...]] = {
    "pass_a": ("mode_c_draft.json",),
    "pass_b": ("mode_c_asr_final.json",),
    "polish": ("mode_c_polished.json",),
    "publish": ("mode_c_published.json",),
    "llm": ("mode_c_asr_final.json", "mode_c_polished.json", "mode_c_published.json"),
    "all": (
        "asr_hypotheses.json",
        "mode_c_asr_final.json",
        "mode_c_polished.json",
        "mode_c_published.json",
    ),
}


def _hyp_models(path: Path) -> set[str]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return set()
    models: set[str] = set()
    if isinstance(payload, dict):
        for m in (payload.get("meta") or {}).get("asr_models") or []:
            models.add(str(m).strip().lower())
        records = payload.get("records", [])
    else:
        records = payload
    if not isinstance(records, list):
        return models
    for rec in records:
        if not isinstance(rec, dict):
            continue
        for h in rec.get("hyps") or []:
            if isinstance(h, dict) and str(h.get("model") or "").strip():
                models.add(str(h["model"]).strip().lower())
    return models


def stage_complete(
    work_dir: Path,
    stage: str,
    asr_models: list[str] | None = None,
) -> bool:
    """True when this work_dir already has the artifacts for ``stage``."""
    work_dir = Path(work_dir)
    stage = str(stage).lower()
    if stage == "asr":
        path = work_dir / "asr_hypotheses.json"
        if not path.is_file() or path.stat().st_size <= 0:
            return False
        needed = {m.strip().lower() for m in (asr_models or []) if m.strip()}
        if not needed:
            return True
        return needed <= _hyp_models(path)
    files = _STAGE_MARKERS.get(stage)
    if not files:
        return False
    for name in files:
        p = work_dir / name
        if not p.is_file() or p.stat().st_size <= 0:
            return False
    return True


def split_even(items: list, n_parts: int) -> list[list]:
    """Split ``items`` into ``n_parts`` contiguous slices covering every element once."""
    n_parts = int(n_parts)
    if n_parts < 1:
        raise ValueError("n_parts must be >= 1")
    seq = list(items)
    q, r = divmod(len(seq), n_parts)
    parts: list[list] = []
    start = 0
    for _k in range(n_parts):
        size = q + (1 if _k < r else 0)
        parts.append(seq[start : start + size])
        start += size
    return parts


def parse_shard(spec: str | None) -> tuple[int, int] | None:
    if spec is None or str(spec).strip() == "":
        return None
    raw = str(spec).strip()
    if "/" not in raw:
        raise ValueError(f"invalid shard {spec!r}; expected i/n (e.g. 0/4)")
    left, right = raw.split("/", 1)
    index, n_shards = int(left), int(right)
    if n_shards < 1 or index < 0 or index >= n_shards:
        raise ValueError(f"invalid shard {spec!r}; expected 0 <= i < n")
    return index, n_shards


def plan_npu_jobs(devices: list[int], npu_per_job: int) -> list[dict[str, Any]]:
    npu_per_job = int(npu_per_job)
    if npu_per_job < 1:
        raise ValueError("npu_per_job must be >= 1")
    ids = [int(d) for d in devices]
    n_jobs = len(ids) // npu_per_job
    if n_jobs < 1:
        raise ValueError(f"need at least {npu_per_job} devices for one job, got {ids}")
    return [
        {
            "shard_index": i,
            "n_shards": n_jobs,
            "devices": ids[i * npu_per_job : (i + 1) * npu_per_job],
        }
        for i in range(n_jobs)
    ]


def shard_summary_name(index: int) -> str:
    return f"batch_summary.shard{index}.json"


def merge_shard_summaries(work_root: Path, n_shards: int) -> dict[str, Any]:
    work_root = Path(work_root)
    shards: list[dict[str, Any]] = []
    missing: list[str] = []
    for i in range(n_shards):
        path = work_root / shard_summary_name(i)
        if not path.is_file():
            missing.append(str(path))
            continue
        shards.append(json.loads(path.read_text(encoding="utf-8")))
    if missing:
        raise FileNotFoundError(f"shard summaries missing: {missing}")
    merged = dict(shards[0])
    merged["n_paired"] = sum(int(s.get("n_paired") or 0) for s in shards)
    merged["n_ok"] = sum(int(s.get("n_ok") or 0) for s in shards)
    merged["n_cached"] = sum(int(s.get("n_cached") or 0) for s in shards)
    merged["n_error"] = sum(int(s.get("n_error") or 0) for s in shards)
    merged["n_skip"] = int(shards[0].get("n_skip") or 0)
    merged["skips"] = shards[0].get("skips") or []
    merged["results"] = [row for s in shards for row in (s.get("results") or [])]
    merged["n_shards"] = n_shards
    merged["shards"] = [str(s.get("shard") or f"{i}/{n_shards}") for i, s in enumerate(shards)]
    return merged


def launch_npu_shards(
    *,
    devices: list[int],
    npu_per_job: int,
    work_root: Path,
    child_argv: list[str],
    executable: str | None = None,
) -> dict[str, Any]:
    """Spawn one process per NPU group, wait for all, merge shard summaries."""
    jobs = plan_npu_jobs(devices, npu_per_job)
    work_root = Path(work_root)
    work_root.mkdir(parents=True, exist_ok=True)
    exe = executable or sys.executable
    repo_root = str(Path(__file__).resolve().parent.parent)
    procs: list[subprocess.Popen] = []
    for job in jobs:
        env = os.environ.copy()
        visible = ",".join(str(d) for d in job["devices"])
        env["ASCEND_RT_VISIBLE_DEVICES"] = visible
        existing = env.get("PYTHONPATH", "")
        env["PYTHONPATH"] = repo_root if not existing else f"{repo_root}{os.pathsep}{existing}"
        cmd = [
            exe,
            "-m",
            "stage2_asr.cli",
            *child_argv,
            "--shard",
            f"{job['shard_index']}/{job['n_shards']}",
        ]
        _log(
            f"[batch] launch shard {job['shard_index']}/{job['n_shards']} "
            f"devices={visible}"
        )
        procs.append(
            subprocess.Popen(
                cmd,
                env=env,
                stdout=subprocess.PIPE,
                stderr=None,
            )
        )
    rc = 0
    for job, proc in zip(jobs, procs):
        out, _err = proc.communicate()
        code = int(proc.returncode or 0)
        if out:
            text = out.decode("utf-8", errors="replace").strip()
            if text:
                _log(f"[batch] shard {job['shard_index']}/{job['n_shards']} stdout: {text}")
        if code:
            rc = code
            _log(f"[batch] shard {job['shard_index']}/{job['n_shards']} exit={code}")
    merged = merge_shard_summaries(work_root, n_shards=len(jobs))
    merged["launcher_exit"] = rc
    _write_summary(work_root, merged)
    _log(
        f"[batch] all shards done n_paired={merged['n_paired']} "
        f"ok={merged['n_ok']} error={merged['n_error']} "
        f"summary={work_root / 'batch_summary.json'}"
    )
    return merged


def build_runners(
    *,
    backend: str,
    stage: str,
    work_dir: Path,
    enable_real: bool,
    mock_hyps: Path | None,
    qwen_model_id: str,
    llm_model_id: str,
    llm_backend: str = "transformers",
    llm_base_url: str | None = None,
    llm_api_key: str | None = None,
    llm_timeout_s: float = 300.0,
    vllm_tp_size: int = 1,
    vllm_gpu_memory_utilization: float = 0.90,
    vllm_max_model_len: int | None = None,
    vllm_dtype: str = "auto",
    vllm_enforce_eager: bool = True,
    vllm_use_v1: bool | None = False,
    llm_enable_thinking: bool = False,
):
    """Construct ASR/LLM runners once for a batch (reuse across samples)."""
    from stage2_asr.runners.ensemble import EnsembleAsrRunner
    from stage2_asr.runners.firered_asr2s import FireRedAsr2sConfig, FireRedAsr2sRunner
    from stage2_asr.runners.llm_qwen36 import Qwen36LlmJudge
    from stage2_asr.runners.mock_asr import MockAsrRunner
    from stage2_asr.runners.mock_llm import MockLlmJudge
    from stage2_asr.runners.qwen3_asr import Qwen3AsrRunner

    needs_asr = stage in {"all", "asr"}
    needs_llm = stage in {"all", "pass_a", "pass_b", "llm", "polish", "publish"}
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
        llm = Qwen36LlmJudge(
            enabled=True,
            model_id=llm_model_id,
            temperature=0.1,
            backend=llm_backend,
            base_url=llm_base_url,
            api_key=llm_api_key,
            timeout_s=llm_timeout_s,
            tensor_parallel_size=vllm_tp_size,
            gpu_memory_utilization=vllm_gpu_memory_utilization,
            max_model_len=vllm_max_model_len,
            dtype=vllm_dtype,
            enforce_eager=vllm_enforce_eager,
            use_v1=vllm_use_v1,
            enable_thinking=llm_enable_thinking,
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
    continue_on_error: bool = True,
    llm_backend: str = "transformers",
    llm_base_url: str | None = None,
    llm_api_key: str | None = None,
    llm_timeout_s: float = 300.0,
    vllm_tp_size: int = 1,
    vllm_gpu_memory_utilization: float = 0.90,
    vllm_max_model_len: int | None = None,
    vllm_dtype: str = "auto",
    vllm_enforce_eager: bool = True,
    vllm_use_v1: bool | None = False,
    llm_enable_thinking: bool = False,
    skip_existing: bool = True,
    sample_workers: int = 1,
    shard: str | None = None,
) -> dict[str, Any]:
    """Discover pairs and run Stage-2 per sample under work_root/{audio-relative-path}/."""
    cfg = config or PipelineConfig()
    hotwords = hotwords or []
    asr_models = asr_models or ["moss", "qwen", "firered"]
    work_root = Path(work_root)
    work_root.mkdir(parents=True, exist_ok=True)
    do_skip = bool(skip_existing) and not bool(getattr(cfg, "force_refresh", False))
    workers = max(1, int(sample_workers or 1))
    if not continue_on_error:
        workers = 1
    parsed_shard = parse_shard(shard)
    summary_name = (
        shard_summary_name(parsed_shard[0]) if parsed_shard is not None else "batch_summary.json"
    )

    pairs, skips = discover_benchmark_pairs(
        wav_benchmark,
        mode_c_benchmark,
        datasets=datasets,
    )
    if limit is not None:
        pairs = pairs[: max(0, int(limit))]
    n_paired_all = len(pairs)
    if parsed_shard is not None:
        index, n_shards = parsed_shard
        pairs = split_even(pairs, n_shards)[index]

    summary: dict[str, Any] = {
        "backend": backend,
        "stage": stage,
        "asr_models": asr_models,
        "llm_backend": llm_backend,
        "llm_base_url": llm_base_url,
        "wav_benchmark": str(wav_benchmark),
        "mode_c_benchmark": str(mode_c_benchmark),
        "work_root": str(work_root),
        "n_paired": len(pairs),
        "n_paired_all": n_paired_all,
        "n_skip": len(skips),
        "n_ok": 0,
        "n_cached": 0,
        "n_error": 0,
        "dry_run": dry_run,
        "skip_existing": do_skip,
        "sample_workers": workers,
        "skips": skips,
        "results": [],
    }
    if parsed_shard is not None:
        summary["shard"] = f"{parsed_shard[0]}/{parsed_shard[1]}"

    if dry_run:
        summary["results"] = [
            {**p.to_dict(), "work_dir": str(p.work_dir(work_root)), "status": "dry_run"}
            for p in pairs
        ]
        _write_summary(work_root, summary, filename=summary_name)
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
        llm_backend=llm_backend,
        llm_base_url=llm_base_url,
        llm_api_key=llm_api_key,
        llm_timeout_s=llm_timeout_s,
        vllm_tp_size=vllm_tp_size,
        vllm_gpu_memory_utilization=vllm_gpu_memory_utilization,
        vllm_max_model_len=vllm_max_model_len,
        vllm_dtype=vllm_dtype,
        vllm_enforce_eager=vllm_enforce_eager,
        vllm_use_v1=vllm_use_v1,
        llm_enable_thinking=llm_enable_thinking,
    )

    n_pairs = len(pairs)
    _log(
        f"[batch] start stage={stage} paired={n_pairs} skipped={len(skips)} "
        f"workers={workers} skip_existing={do_skip} "
        f"models={asr_models} work_root={work_root}"
    )

    def _process(pair: SamplePair) -> dict[str, Any]:
        sample_work = pair.work_dir(work_root)
        sample_work.mkdir(parents=True, exist_ok=True)
        row: dict[str, Any] = {
            **pair.to_dict(),
            "work_dir": str(sample_work),
        }
        if do_skip and stage_complete(sample_work, stage, asr_models):
            row["status"] = "skipped_existing"
            return row
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
            if result.get("draft_merged_path") is not None:
                row["draft_merged"] = str(result["draft_merged_path"])
            if result.get("final_merged_path") is not None:
                row["final_merged"] = str(result["final_merged_path"])
            if result.get("polished_path") is not None:
                row["polished"] = str(result["polished_path"])
            if result.get("published_path") is not None:
                row["published"] = str(result["published_path"])
            if result.get("transcript_path") is not None:
                row["transcript"] = str(result["transcript_path"])
            if result.get("glossary_path") is not None:
                row["glossary"] = str(result["glossary_path"])
            if result.get("asr_hypotheses_path") is not None:
                row["asr_hypotheses"] = str(result["asr_hypotheses_path"])
            return row
        except Exception as exc:  # noqa: BLE001
            row["status"] = "error"
            row["error"] = str(exc)
            row["traceback"] = traceback.format_exc(limit=5)
            if not continue_on_error:
                raise
            return row

    rows: list[dict[str, Any]] = []
    if workers <= 1:
        for bi, pair in enumerate(pairs, start=1):
            _log(f"[batch] {bi}/{n_pairs} {pair.sample_id} begin")
            try:
                row = _process(pair)
            except Exception as exc:  # noqa: BLE001
                row = {
                    **pair.to_dict(),
                    "work_dir": str(pair.work_dir(work_root)),
                    "status": "error",
                    "error": str(exc),
                    "traceback": traceback.format_exc(limit=5),
                }
                rows.append(row)
                summary["results"] = rows
                _tally_batch(summary, rows)
                _write_summary(work_root, summary, filename=summary_name)
                raise
            _log(f"[batch] {bi}/{n_pairs} {pair.sample_id} {row.get('status')}")
            rows.append(row)
    else:
        with ThreadPoolExecutor(max_workers=workers) as pool:
            futs = [pool.submit(_process, pair) for pair in pairs]
            for bi, (pair, fut) in enumerate(zip(pairs, futs), start=1):
                row = fut.result()
                _log(f"[batch] {bi}/{n_pairs} {pair.sample_id} {row.get('status')}")
                rows.append(row)

    summary["results"] = rows
    _tally_batch(summary, rows)
    _write_summary(work_root, summary, filename=summary_name)
    _log(
        f"[batch] done ok={summary['n_ok']} cached={summary['n_cached']} "
        f"error={summary['n_error']} skip={summary['n_skip']} "
        f"summary={work_root / summary_name}"
    )
    return summary


def _tally_batch(summary: dict[str, Any], rows: list[dict[str, Any]]) -> None:
    summary["n_ok"] = sum(1 for r in rows if r.get("status") == "ok")
    summary["n_cached"] = sum(1 for r in rows if r.get("status") == "skipped_existing")
    summary["n_error"] = sum(1 for r in rows if r.get("status") == "error")


def _write_summary(
    work_root: Path,
    summary: dict[str, Any],
    *,
    filename: str = "batch_summary.json",
) -> None:
    path = work_root / filename
    path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
