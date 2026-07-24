# Stage-2 Multi-ASR + LLM Fusion

Mock-first Stage-2 package on **diarizen_moss_fusion** Mode-C outputs, with optional real adapters for Qwen3-ASR, FireRedASR2S (VAD off / LID+Punc on), and Qwen3.6-27B.

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
```

FireRed system config used by the adapter:

`enable_vad=False`, `enable_lid=True`, `enable_punc=True`

## Layout

```text
stage2_asr/          # pipeline + runners
tests/               # unit + e2e mocks (fake injected models)
docs/                # design + references
third_party/         # optional local clones (gitignored)
```

## Artifacts

`asr_units.json`, `asr_hypotheses.json`, `mode_c_draft.json`, `mode_c_asr_final.json`, `llm_edits.jsonl`, `asr_cache/`, `crops/`
