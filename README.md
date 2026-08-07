# Stage-2 Multi-ASR + LLM Fusion

Mock-first Stage-2 package on **diarizen_moss_fusion** Mode-C outputs, with optional real adapters for Qwen3-ASR, FireRedASR2S (VAD off / LID+Punc on), Qwen3.6-27B (primary judge), and DeepSeek (judge fallback).

## Docs

- [Design](docs/2026-07-24-stage2-multi-asr-llm-fusion-design.md)
- [Upstream references](docs/references.md)

## Install

```bash
python3.12 -m venv .venv312
source .venv312/bin/activate
pip install -e ".[dev]"
pytest -q
```

## Mock end-to-end (no weights)

```bash
stage2-asr run \
  --input tests/fixtures/mode_c.json \
  --audio /tmp/unused.wav \
  --work-dir /tmp/stage2_out \
  --mock
```

## Real backend (opt-in; needs local packages + weights)

```bash
# optional: clone API references
# git clone --depth 1 https://github.com/QwenLM/Qwen3-ASR third_party/Qwen3-ASR
# git clone --depth 1 https://github.com/FireRedTeam/FireRedASR2S third_party/FireRedASR2S

stage2-asr run \
  --input mode_c.json \
  --audio prepared.wav \
  --work-dir out \
  --backend real --enable-real
# optional: --no-deepseek-fallback
```

## Staged mode (low-resource / step-by-step)

You can run ASR and LLM in separate stages and reuse artifacts in `--work-dir`.

```bash
# 1) ASR only (save per-unit hyps/cache; no LLM)
stage2-asr run --input mode_c.json --audio prepared.wav --work-dir out --backend real --enable-real --stage asr --asr-models qwen
stage2-asr run --input mode_c.json --audio prepared.wav --work-dir out --backend real --enable-real --stage asr --asr-models firered

# 2) LLM only (reads out/asr_hypotheses.json; no ASR inference)
stage2-asr run --input mode_c.json --audio prepared.wav --work-dir out --backend real --enable-real --stage llm
```

Available stages: `all` (default), `asr`, `pass_a`, `pass_b`, `llm`  
ASR model subsets: `moss`, `qwen`, `firered` (comma-separated via `--asr-models`)

FireRed system config used by the adapter:

`enable_vad=False`, `enable_lid=True`, `enable_punc=True`

## Dataset batch mode

Pairs wavs and Mode-C JSONs under parallel `benchmark/` trees:

```text
{wav-benchmark}/{dataset}/Audio/{stem}.wav
{mode-c-benchmark}/{dataset}/Audio/{stem}/mode_c.json
→ work-root/{dataset}/{stem}/
```

Example (your layout):

```bash
# dry-run: list pairs / missing mode_c without loading models
stage2-asr run-batch \
  --wav-benchmark /home/ma-user/work/dataset/audio_process_ulan_obs/zyx/test_datasets/benchmark \
  --mode-c-benchmark /home/ma-user/work/dataset/audio_process_ulan_obs/zyx/DiarizenMossFusion/benchmark \
  --work-root /home/ma-user/work/dataset/audio_process_ulan_obs/zyx/stage2_out \
  --dry-run

# staged ASR then LLM on one dataset
stage2-asr run-batch \
  --wav-benchmark .../test_datasets/benchmark \
  --mode-c-benchmark .../DiarizenMossFusion/benchmark \
  --work-root .../stage2_out \
  --datasets some_dataset_name \
  --backend real --enable-real --stage asr --asr-models qwen

stage2-asr run-batch \
  --wav-benchmark .../test_datasets/benchmark \
  --mode-c-benchmark .../DiarizenMossFusion/benchmark \
  --work-root .../stage2_out \
  --datasets some_dataset_name \
  --backend real --enable-real --stage asr --asr-models firered

stage2-asr run-batch \
  --wav-benchmark .../test_datasets/benchmark \
  --mode-c-benchmark .../DiarizenMossFusion/benchmark \
  --work-root .../stage2_out \
  --datasets some_dataset_name \
  --backend real --enable-real --stage llm
```

Useful flags: `--limit N`, `--fail-fast`, `--hotwords path.json`.  
Summary + skips/errors: `work-root/batch_summary.json`.

## LLM backend (vLLM / Ascend 910B)

Three options:

| `--llm-backend` | How it runs | When to use |
|---|---|---|
| `vllm_engine` | **In-process `vllm.LLM`** (loads once, batched `generate`) | **Recommended on Ascend 910B** (vLLM-Ascend installed) |
| `vllm` | OpenAI HTTP client → external server | Separate long-lived `vllm serve` |
| `transformers` | HF `generate` one-by-one | Debug only (very slow) |

### Recommended: in-process engine (no separate server)

```bash
# Requires vLLM-Ascend (or vLLM) importable in the same Python env as Stage-2.
python -m stage2_asr.cli run-batch \
  --wav-benchmark ... --mode-c-benchmark ... --work-root ... \
  --backend real --enable-real --stage llm \
  --llm-backend vllm_engine \
  --llm-model-id /path/or/hf/id/to/Qwen3.6-27B \
  --pass-a-batch-size 16 \
  --vllm-tp-size 1 \
  --vllm-gpu-memory-utilization 0.90 \
  --no-deepseek-fallback
```

- First call loads the model into NPU; later units reuse the same engine.
- `--pass-a-batch-size N` runs Pass A as one `LLM.generate([N prompts])` (true batching).
- DeepSeek fallback is **disabled automatically** for `vllm_engine` (avoids loading a second engine / OOM). Prefer `--no-deepseek-fallback`.
- Traces: `work-dir/llm_infer.jsonl`

**If you see `Invalid thread pool` / `Engine core initialization failed` (vLLM 0.18 V1):** this is a PyTorch OpenMP + V1 multiprocess bug, not Stage-2 logic. Pull latest (defaults `VLLM_USE_V1=0`, `enforce_eager`, `spawn`) or set before running:

```bash
export VLLM_USE_V1=0
export VLLM_WORKER_MULTIPROC_METHOD=spawn
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export VLLM_HOST_IP=127.0.0.1
```

Also confirm `from vllm import LLM` is **vLLM-Ascend** (NPU), not a CUDA-only build (`Device:-1` in logs is a red flag).

### Alternative: HTTP server (`--llm-backend vllm`)

Start vLLM-Ascend yourself, then `--llm-base-url http://127.0.0.1:8000`.

## Eval B0 (MOSS-from-fusion baseline)

```bash
python scripts/eval_b0.py --hyp mode_c.json --ref ref.json --json-out out/b0.json
```

Compares Mode-C provisional turn texts to a same-order reference; reports corpus CER and mean cpCER.

## Layout

```text
stage2_asr/          # pipeline + runners
scripts/             # eval helpers (e.g. eval_b0.py)
tests/               # unit + e2e mocks (fake injected models)
docs/                # design + references
third_party/         # optional local clones (gitignored)
```

## Artifacts

`asr_units.json` (reloaded on `pass_a`/`pass_b`/`llm` so unit_ids stay stable), `asr_hypotheses.json` (hyps merge across `asr` and `all` re-runs), `mode_c_draft.json`, `mode_c_asr_final.json`, `llm_edits.jsonl` (Pass A preserved when re-running `pass_b`), `pass_stats.json` (merged across staged passes), `llm_infer.jsonl` (LLM request/response traces for Pass A/B), `asr_cache/`, `crops/` (reused across ASR model runs; not rewritten if present)

Progress logs go to **stderr** (`[asr]`, `[pass_a]`, `[pass_b]`, `[batch]`); the final JSON summary stays on **stdout**.
