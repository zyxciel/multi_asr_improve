# Stage-2 Status & Next Steps (Handoff)

**Date:** 2026-08-11  
**Repo:** `zyxciel/multi_asr_improve` (branch `master`)  
**Audience:** operators running Ascend / multi-GPU batch jobs; engineers planning post-pilot cleanup  
**Related:** [Architecture design](2026-07-24-stage2-multi-asr-llm-fusion-design.md), [README](../README.md), [hotwords](hotwords.txt)

---

## 1. Where we are

Stage-2 is **operational** for pilot / platform runs:

| Capability | Status |
|---|---|
| Mode-C input (full doc or plain turn array) | Done |
| Staged runs (`asr` / `pass_a` / `pass_b` / `llm`) + artifact reuse | Done |
| Dataset `run-batch` (wav ↔ `mode_c.json` pairing) | Done |
| Multi-ASR (MOSS / Qwen3-ASR / FireRed) + crop reuse | Done |
| Pass A/B LLM judge (Qwen3.8-27B) via `vllm_engine` on Ascend | Done |
| Thinking/CoT off by default + robust JSON extract | Done |
| Pass A true batching + **re-batched retries** (no serial `1/1` tail) | Done |
| Optional Pass B `judge_many` (`--pass-b-batch-size N`, default sequential) | Done |
| `llm_infer.jsonl` with `user` + `response` (≤16k chars each) | Done |
| Hotwords: `docs/hotwords.txt` (761 unique; plaintext loader) | Done |

**Still deferred (by design until pilot metrics land):**

- Discourse / semantic Tier (e.g. `爱情` → `娃娃亲` from topic context)
- Punctuation / fluency / term-normalization as a separate polish stage
- Token-aware neighbor truncation (ops workaround: raise `--vllm-max-model-len`)
- Built-in data-parallel launcher (use split tasks instead)

---

## 2. What the system actually optimizes

Primary goal is **phonetic fidelity**, not creative transcript rewriting.

**Allowed repairs**

- Tier A: select / merge among ASR hypotheses  
- Tier B: exact pinyin, wrong characters  
- Tier C: pinyin edit distance ≤ 2 **and** anchor ∈ `{hyp, neighbor_draft, meeting_draft, hotword}`  
- Span-local: `|len(span_out) − len(span_asr)| ≤ 1`

**Forbidden**

- Context-only / open-world fixes without a pinyin link  
- Abbreviation expansion  

**Implication:** neighbors and hotwords are **anchors for near-homophones**, not proof of “discourse understanding.” Modest CER gains vs strong multi-ASR fusion alone are expected. Cases like `爱情`→`娃娃亲` are **correctly rejected** under current policy.

---

## 3. Artifacts to inspect after a run

Per sample under `work-root/{dataset}/{stem}/`:

| File | Use |
|---|---|
| `mode_c_draft.json` | After Pass A |
| `mode_c_asr_final.json` | After Pass B (deliverable) |
| `llm_edits.jsonl` | Accepted edits (Pass A + Pass B) |
| `llm_infer.jsonl` | Full judge traces (`user`, `response`, `reasoning`, errors) |
| `pass_stats.json` | Counts: skipped LLM, retries, Pass B `n_audits`, … |
| `asr_hypotheses.json` / `asr_units.json` / `crops/` | ASR cache for staged re-runs |

Batch-level: `work-root/batch_summary.json` — **run log only** (ok/error/paths), **not** an input manifest.

**Pass B `n_audits: 0` is often normal** — audits are change/failure records, not per-turn “visited” counters. Confirm LLM ran via `pass_b_t*` lines in `llm_infer.jsonl`.

---

## 4. How to run (pilot → scale)

### 4.1 Single sample / small batch (Ascend, typical)

```bash
export ASCEND_RT_VISIBLE_DEVICES=0,1
export VLLM_USE_V1=0
export VLLM_WORKER_MULTIPROC_METHOD=spawn

stage2-asr run-batch \
  --wav-benchmark /path/to/wav/benchmark \
  --mode-c-benchmark /path/to/mode_c/benchmark \
  --work-root /path/to/stage2_out \
  --backend real --enable-real --stage llm \
  --llm-backend vllm_engine \
  --llm-model-id /path/or/hf/id/Qwen3.8-27B \
  --vllm-tp-size 2 \
  --vllm-dtype bf16 \
  --vllm-max-model-len 8192 \
  --pass-a-batch-size 64 \
  --no-deepseek-fallback \
  --hotwords docs/hotwords.txt
```

Do **not** pass `--llm-enable-thinking` unless debugging CoT.

### 4.2 Hotwords

```bash
--hotwords docs/hotwords.txt
```

Formats accepted:

- Plaintext, one term per line (current repo file)  
- JSON array: `["单框架", "账号|帐号"]`  
- Alias `正确词|错误写法` → Pass B **deterministic** rewrite when `|Δlen|≤1`

Current `docs/hotwords.txt` has **no `|` aliases** → terms are LLM Tier-C **hints** only.

### 4.3 Multi-GPU / 32-device scale-out (recommended)

**Prefer data-parallel split tasks, not one process with TP=32.**

- Code today: one in-process `vllm.LLM` per task; `--vllm-tp-size` = tensor parallel only  
- No built-in data parallel across samples  

**Recipe (example: 32 GPUs, TP=2 → 16 tasks):**

1. Partition work with `--datasets A,B,...` and/or split `Audio/` trees  
2. Per task: unique `ASCEND_RT_VISIBLE_DEVICES`, unique `--work-root`  
3. Same model flags / hotwords on every task  
4. Merge finals offline from each `work-root`

Raising `--pass-a-batch-size` alone does **not** use idle GPUs outside the TP group.

### 4.4 Prompt length / max_model_len

Pass B (and heavy Pass A neighbors) can exceed small `max_model_len` (e.g. 4096) because:

- Neighbor char budget ≈ 8k chars (rough token estimate)  
- Template + hyps/pinyin + hotwords + `max_tokens` reservation  

**Ops fix first:** `--vllm-max-model-len 8192` (or 16384) if KV allows; else lower util (`--vllm-gpu-memory-utilization 0.85`) / keep TP=2.  
**Pass B speed A/B:** keep ASR + Pass A artifacts; rerun `--stage pass_b` with `--pass-b-batch-size 1` vs `64` into **separate** `--work-dir` copies (or copy `mode_c_draft.json` / hyps). Compare wall time, `pass_stats.pass_b.n_audits` / `n_batched`, and `llm_edits.jsonl` Pass B lines.

---

## 5. Evidence checklist (before claiming “context helps”)

| Check | How |
|---|---|
| Neighbor ablation | Set `PipelineConfig.neighbor_max_turns=0` (not CLI yet) vs default 20 / 600s; compare CER |
| Tier-C-anchor rate | From `llm_edits.jsonl`: share of edits with `tier=C` and `anchor∈{neighbor_draft,meeting_draft}` |
| Hotword impact | Run with / without `--hotwords`; compare edits with `anchor=hotword` |
| Slice metrics | Near-homophone / hotword / overlap slices vs full-corpus CER (Eval B0 helpers exist) |

If ablating neighbors barely moves CER and Tier-C neighbor anchors are rare, gains are mostly **multi-ASR selection**, not discourse context.

---

## 6. Known behaviors seen in pilot

1. **`unit_0023`-style cases** (`爱情` / `玩钱` vs topic `娃娃亲`): context present in `user`, edit blocked by Tier B/C — **expected**.  
2. **Pass A `63/63` then many `1/1`:** previously serial retries; fixed by re-batch (`3840e08`). After pull, retries should stay `N/N`.  
3. **Low NPU util:** long prompts + `max_tokens=1024` KV reservation + `enforce_eager` / V0 — shrink neighbors and/or lower max tokens for better packing.  
4. **Batch sample → next sample:** no cross-utterance LLM memory; only within-meeting neighbors.  
5. **`batch_summary.json` errors:** status log; split data via filesystem / `--datasets`, not this file.

---

## 7. Suggested next steps (after this pilot’s metrics)

Ordered for a large-scale cleanup launch:

### A. Decide go / no-go from pilot

- [ ] Corpus CER / cpCER vs Eval B0 (MOSS-from-fusion) and vs Pass A-only draft  
- [ ] Retry rate, Pass B `n_audits`, hotword-anchored edit rate  
- [ ] Spot-check `llm_infer.jsonl` for overflow / empty / bad JSON  

### B. Ops scale-out (no policy change)

- [ ] Shard datasets across N tasks × TP=2 (or TP as needed for KV)  
- [ ] Standardize flags: `vllm_engine`, thinking off, hotwords path, max_model_len  
- [ ] Per-shard `work-root` + final merge script / checklist  

### C. Hotword hygiene (cheap quality win)

- [ ] Add high-value `canon|alt` aliases for frequent ASR confusions  
- [ ] Keep list deduped; avoid dumping unused jargon if prompts hit max_len  

### D. Engineering (if util / OOM still hurts)

- [ ] CLI for `--neighbor-max-turns` / `--neighbor-window-seconds`  
- [ ] Token-aware prompt truncate (reserve `max_tokens`)  
- [ ] Lower default judge `max_tokens` (256–512)  

### E. Policy (only if pilot shows discourse errors dominate)

- [ ] Optional semantic / topic-consistency pass (new claim, new eval slice)  
- [ ] Keep phonetic Pass A/B unchanged so gains remain measurable  

### F. Later polish (explicitly out of Stage-2 phonetic core)

- [ ] Punctuation / fluency / display terms  

---

## 8. Key commits (recent)

| Commit | Topic |
|---|---|
| `c495a2e` | Plaintext hotwords loader + `docs/hotwords.txt` |
| `3840e08` | Pass A retry re-batching |
| `590ddc8` | Log `user` prompts in `llm_infer.jsonl` |
| `e1235be` | Thinking off + JSON / reasoning extract |
| `2a3abe8` / `8a5b1e3` / `2f0c382` | Ascend `vllm_engine`, dtype, TP, OpenMP/V1 hardening |
| `5b67365` | HTTP vLLM + Pass A batch + infer log |
| `2d92e54` | `run-batch` dataset pairing |

---

## 9. One-page operator cheat sheet

```text
Goal:      phonetic multi-ASR fusion + constrained LLM repair
Input:     Mode-C JSON + wav (batch layout under benchmark/*/Audio)
Hotwords:  --hotwords docs/hotwords.txt
LLM:       --llm-backend vllm_engine --vllm-tp-size 2 --vllm-dtype bf16
Batch:     --pass-a-batch-size 64  (retries also batched)
Thinking:  OFF (default)
Scale-out: split data → many tasks; do not TP across all 32 as one job
Debug:     work-dir/llm_infer.jsonl  (+ user field)
Final:     work-dir/.../mode_c_asr_final.json
```

When pilot numbers are back, start from **§7.A → §7.B**; open **§7.E** only if discourse errors clearly dominate the remaining CER gap.
