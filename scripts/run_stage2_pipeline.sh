#!/usr/bin/env bash
# Stage-2 full workflow: install this package, smoke-test, then run ASR → LLM on a cluster.
#
# This repo only pip-installs stage2-asr (+ pytest). Real weights and
# qwen_asr / fireredasr2s / vLLM-Ascend must already be importable in PYTHON.
#
# Usage:
#   export STAGE2_ENV=debug    # or prod; sources scripts/env.${STAGE2_ENV}.sh
#   export WAV_BENCHMARK=/path/to/audio_root
#   export MODE_C_BENCHMARK=/path/to/mode_c_root
#   export WORK_ROOT=/path/to/stage2_out
#   export LLM_MODEL_ID=/path/to/Qwen3.8-27B
#   export DEVICES=0,1,2,3,4,5,6,7
#
#   ./scripts/run_stage2_pipeline.sh install
#   ./scripts/run_stage2_pipeline.sh check
#   ./scripts/run_stage2_pipeline.sh mock
#   ./scripts/run_stage2_pipeline.sh dry-run
#   ./scripts/run_stage2_pipeline.sh asr          # moss+qwen, then firered
#   ./scripts/run_stage2_pipeline.sh llm          # pass_a+b + polish + publish
#   ./scripts/run_stage2_pipeline.sh pipeline     # dry-run + asr + llm
#
# One stage only:  asr-qwen | asr-firered | pass_a | pass_b | polish | publish
# Skip install inside pipeline:  SKIP_INSTALL=1 ./scripts/run_stage2_pipeline.sh pipeline

set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

log() { printf '[stage2] %s\n' "$*" >&2; }
die() { printf '[stage2] ERROR: %s\n' "$*" >&2; exit 1; }

# Debug vs production ASR weight locations (copy scripts/env.<name>.sh.example).
STAGE2_ENV="${STAGE2_ENV:-}"
if [[ -n "$STAGE2_ENV" ]]; then
  env_file="$ROOT/scripts/env.${STAGE2_ENV}.sh"
  [[ -f "$env_file" ]] || die "STAGE2_ENV=$STAGE2_ENV but missing $env_file — copy scripts/env.${STAGE2_ENV}.sh.example and fill local Qwen/FireRed paths"
  # shellcheck disable=SC1090
  source "$env_file"
  log "loaded $env_file (STAGE2_ENV=$STAGE2_ENV)"
fi

# ---------------------------------------------------------------------------
# Paths / models (override with environment variables)
# ---------------------------------------------------------------------------
WAV_BENCHMARK="${WAV_BENCHMARK:-/home/ma-user/work/dataset/audio_process_ulan_obs/zyx/test_datasets/benchmark}"
MODE_C_BENCHMARK="${MODE_C_BENCHMARK:-/home/ma-user/work/dataset/audio_process_ulan_obs/zyx/DiarizenMossFusion/benchmark}"
WORK_ROOT="${WORK_ROOT:-/home/ma-user/work/dataset/audio_process_ulan_obs/zyx/stage2_out}"
DATASETS="${DATASETS:-}"                          # e.g. ds_a,ds_b  (empty = all)
LIMIT="${LIMIT:-}"                                # e.g. 8 for a smoke subset
HOTWORDS="${HOTWORDS:-$ROOT/docs/hotwords.txt}"

QWEN_MODEL_ID="${QWEN_MODEL_ID:-Qwen/Qwen3-ASR-1.7B}"
FIRERED_ASR_MODEL_DIR="${FIRERED_ASR_MODEL_DIR:-}"
FIRERED_LID_MODEL_DIR="${FIRERED_LID_MODEL_DIR:-}"
FIRERED_PUNC_MODEL_DIR="${FIRERED_PUNC_MODEL_DIR:-}"
LLM_MODEL_ID="${LLM_MODEL_ID:-Qwen/Qwen3.6-27B}"

# LLM NPU split: 8 cards → 4 jobs × 2 NPUs. Empty DEVICES = single process.
DEVICES="${DEVICES:-0,1,2,3,4,5,6,7}"
NPU_PER_JOB="${NPU_PER_JOB:-2}"
# ASR is lighter; default 1 card per shard. Empty ASR_DEVICES = no split.
ASR_DEVICES="${ASR_DEVICES:-}"
ASR_NPU_PER_JOB="${ASR_NPU_PER_JOB:-1}"

LLM_BACKEND="${LLM_BACKEND:-vllm_engine}"
VLLM_TP_SIZE="${VLLM_TP_SIZE:-$NPU_PER_JOB}"
VLLM_DTYPE="${VLLM_DTYPE:-bf16}"
VLLM_MAX_MODEL_LEN="${VLLM_MAX_MODEL_LEN:-8192}"
VLLM_GPU_MEMORY_UTILIZATION="${VLLM_GPU_MEMORY_UTILIZATION:-0.90}"
PASS_A_BATCH_SIZE="${PASS_A_BATCH_SIZE:-16}"
PASS_B_BATCH_SIZE="${PASS_B_BATCH_SIZE:-16}"
POLISH_BATCH_SIZE="${POLISH_BATCH_SIZE:-16}"
SAMPLE_WORKERS="${SAMPLE_WORKERS:-1}"

PYTHON="${PYTHON:-python3}"
VENV_DIR="${VENV_DIR:-$ROOT/.venv312}"
SKIP_INSTALL="${SKIP_INSTALL:-0}"
SKIP_TESTS="${SKIP_TESTS:-0}"

CMD="${1:-help}"
shift || true

# ---------------------------------------------------------------------------
activate_python() {
  if [[ -n "${CONDA_PREFIX:-}" && -x "${CONDA_PREFIX}/bin/python" ]]; then
    PYTHON="${CONDA_PREFIX}/bin/python"
    log "using conda python: $PYTHON"
    return
  fi
  if [[ -x "$VENV_DIR/bin/python" ]]; then
    # shellcheck disable=SC1091
    source "$VENV_DIR/bin/activate"
    PYTHON="$VENV_DIR/bin/python"
    log "using venv: $VENV_DIR"
    return
  fi
  log "using PATH python: $PYTHON"
}

cli() {
  "$PYTHON" -m stage2_asr.cli "$@"
}

need_paths() {
  [[ -e "$WAV_BENCHMARK" ]] || die "WAV_BENCHMARK not found: $WAV_BENCHMARK"
  [[ -e "$MODE_C_BENCHMARK" ]] || die "MODE_C_BENCHMARK not found: $MODE_C_BENCHMARK"
  mkdir -p "$WORK_ROOT"
}

common_batch_args() {
  local -a args=(
    run-batch
    --wav-benchmark "$WAV_BENCHMARK"
    --mode-c-benchmark "$MODE_C_BENCHMARK"
    --work-root "$WORK_ROOT"
    --hotwords "$HOTWORDS"
    --qwen-model-id "$QWEN_MODEL_ID"
    --llm-model-id "$LLM_MODEL_ID"
    --sample-workers "$SAMPLE_WORKERS"
  )
  if [[ -n "${FIRERED_ASR_MODEL_DIR:-}" ]]; then
    args+=(--firered-asr-model-dir "$FIRERED_ASR_MODEL_DIR")
  fi
  if [[ -n "${FIRERED_LID_MODEL_DIR:-}" ]]; then
    args+=(--firered-lid-model-dir "$FIRERED_LID_MODEL_DIR")
  fi
  if [[ -n "${FIRERED_PUNC_MODEL_DIR:-}" ]]; then
    args+=(--firered-punc-model-dir "$FIRERED_PUNC_MODEL_DIR")
  fi
  if [[ -n "$DATASETS" ]]; then
    args+=(--datasets "$DATASETS")
  fi
  if [[ -n "$LIMIT" ]]; then
    args+=(--limit "$LIMIT")
  fi
  printf '%s\n' "${args[@]}"
}

device_args() {
  local devices="$1"
  local npu="$2"
  if [[ -n "$devices" ]]; then
    printf '%s\n' --devices "$devices" --npu-per-job "$npu"
  fi
}

harden_vllm_env() {
  export VLLM_USE_V1="${VLLM_USE_V1:-0}"
  export VLLM_WORKER_MULTIPROC_METHOD="${VLLM_WORKER_MULTIPROC_METHOD:-spawn}"
  export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"
  export MKL_NUM_THREADS="${MKL_NUM_THREADS:-1}"
  export VLLM_HOST_IP="${VLLM_HOST_IP:-127.0.0.1}"
}

run_real_batch() {
  local stage="$1"
  shift
  local -a extra=("$@")
  local -a args
  mapfile -t args < <(common_batch_args)
  args+=(--backend real --enable-real --stage "$stage")
  args+=("${extra[@]}")
  log "cli ${args[*]}"
  cli "${args[@]}"
}

# ---------------------------------------------------------------------------
cmd_help() {
  cat <<'EOF'
Stage-2 pipeline helper

Commands:
  install       Create .venv312 (if no conda) and pip install -e ".[dev]"
  check         pytest + import probes for qwen_asr / fireredasr2s / vllm
  mock          Offline mock run on tests/fixtures/mode_c.json
  dry-run       Pair wav ↔ mode_c.json, write batch_summary.json, no models
  asr           Real ASR: moss+qwen, then firered (hyps merge in work-root)
  asr-qwen      Real ASR moss+qwen only
  asr-firered   Real ASR firered only
  llm           Pass A + Pass B + polish + publish (no ASR), multi-NPU if DEVICES set
  pass_a        Pass A only
  pass_b        Pass B only
  polish        Polish only
  publish       Publish only
  pipeline      dry-run + asr + llm  (set SKIP_INSTALL=1 to skip venv/pip)
  help          This text

Required env for real runs:
  WAV_BENCHMARK     audio root (recursive *.wav)
  MODE_C_BENCHMARK  mode_c.json root
  WORK_ROOT         output root  (work-root/{audio-rel}/)
  LLM_MODEL_ID      local Qwen3.8-27B (or 3.6) path
  DEVICES           e.g. 0,1,2,3,4,5,6,7   (empty = single process)

Optional:
  STAGE2_ENV=debug|prod   sources scripts/env.<name>.sh (Qwen/FireRed local paths)
  DATASETS  LIMIT  ASR_DEVICES  NPU_PER_JOB  QWEN_MODEL_ID
  FIRERED_ASR_MODEL_DIR  FIRERED_LID_MODEL_DIR  FIRERED_PUNC_MODEL_DIR
  PASS_A_BATCH_SIZE  PASS_B_BATCH_SIZE  POLISH_BATCH_SIZE
EOF
}

cmd_install() {
  if [[ -n "${CONDA_PREFIX:-}" ]]; then
    log "conda env already active ($CONDA_PREFIX); skip venv create"
    PYTHON="${CONDA_PREFIX}/bin/python"
  elif [[ ! -x "$VENV_DIR/bin/python" ]]; then
    log "creating $VENV_DIR"
    "$PYTHON" -m venv "$VENV_DIR"
    # shellcheck disable=SC1091
    source "$VENV_DIR/bin/activate"
    PYTHON="$VENV_DIR/bin/python"
  else
    # shellcheck disable=SC1091
    source "$VENV_DIR/bin/activate"
    PYTHON="$VENV_DIR/bin/python"
  fi
  log "pip install -e .[dev]"
  "$PYTHON" -m pip install -U pip
  "$PYTHON" -m pip install -e ".[dev]"
  log "installed stage2-asr into $PYTHON"
  log "NOTE: qwen_asr, fireredasr2s, and vLLM-Ascend are NOT installed by this step."
}

cmd_check() {
  activate_python
  if [[ "$SKIP_TESTS" != "1" ]]; then
    log "pytest"
    "$PYTHON" -m pytest -q
  fi
  log "import probes (real backends are optional)"
  "$PYTHON" - <<'PY'
import importlib, sys
print("python", sys.executable)
for name in ("stage2_asr", "numpy", "pypinyin"):
    importlib.import_module(name)
    print("ok", name)
for name, hint in (
    ("qwen_asr", "Qwen3-ASR (needed for --asr-models qwen)"),
    ("fireredasr2s", "FireRedASR2S (needed for --asr-models firered)"),
    ("vllm", "vLLM-Ascend (needed for --llm-backend vllm_engine)"),
):
    try:
        importlib.import_module(name)
        print("ok", name)
    except Exception as exc:
        print("MISSING", name, "-", hint, "-", type(exc).__name__)
PY
}

cmd_mock() {
  activate_python
  local out="${WORK_ROOT%/}_mock"
  mkdir -p "$out"
  log "mock single-sample e2e → $out"
  cli run \
    --input "$ROOT/tests/fixtures/mode_c.json" \
    --audio /tmp/unused.wav \
    --work-dir "$out" \
    --mock
}

cmd_dry_run() {
  activate_python
  need_paths
  local -a args
  mapfile -t args < <(common_batch_args)
  args+=(--dry-run --mock)
  log "dry-run pairing"
  cli "${args[@]}"
}

cmd_asr_models() {
  local models="$1"
  activate_python
  need_paths
  local -a extra
  mapfile -t extra < <(device_args "$ASR_DEVICES" "$ASR_NPU_PER_JOB")
  extra+=(--asr-models "$models")
  run_real_batch asr "${extra[@]}"
}

cmd_llm_stage() {
  local stage="$1"
  activate_python
  need_paths
  harden_vllm_env
  local -a extra
  mapfile -t extra < <(device_args "$DEVICES" "$NPU_PER_JOB")
  extra+=(
    --llm-backend "$LLM_BACKEND"
    --vllm-tp-size "$VLLM_TP_SIZE"
    --vllm-dtype "$VLLM_DTYPE"
    --vllm-max-model-len "$VLLM_MAX_MODEL_LEN"
    --vllm-gpu-memory-utilization "$VLLM_GPU_MEMORY_UTILIZATION"
    --pass-a-batch-size "$PASS_A_BATCH_SIZE"
    --pass-b-batch-size "$PASS_B_BATCH_SIZE"
    --polish-batch-size "$POLISH_BATCH_SIZE"
  )
  run_real_batch "$stage" "${extra[@]}"
}

cmd_pipeline() {
  if [[ "$SKIP_INSTALL" != "1" ]]; then
    cmd_install
  fi
  cmd_check
  cmd_dry_run
  log "=== ASR moss+qwen ==="
  cmd_asr_models moss,qwen
  log "=== ASR firered ==="
  cmd_asr_models firered
  log "=== LLM (pass_a + pass_b + polish + publish) ==="
  cmd_llm_stage llm
  log "done. summary=$WORK_ROOT/batch_summary.json"
}

# ---------------------------------------------------------------------------
case "$CMD" in
  help|-h|--help) cmd_help ;;
  install) cmd_install ;;
  check) cmd_check ;;
  mock) cmd_mock ;;
  dry-run) cmd_dry_run ;;
  asr)
    cmd_asr_models moss,qwen
    cmd_asr_models firered
    ;;
  asr-qwen) cmd_asr_models moss,qwen ;;
  asr-firered) cmd_asr_models firered ;;
  llm) cmd_llm_stage llm ;;
  pass_a|pass_b|polish|publish) cmd_llm_stage "$CMD" ;;
  pipeline) cmd_pipeline ;;
  *)
    cmd_help
    die "unknown command: $CMD"
    ;;
esac
