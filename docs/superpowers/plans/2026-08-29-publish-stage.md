# Publish Stage Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add `--stage publish` that writes fluent bilingual display artifacts (`mode_c_published.json`, `transcript.md`, `glossary.json`, `publish_eval.json`) without changing Pass A/B, polish, or WER files.

**Architecture:** New modules beside polish. Pipeline loads `mode_c_polished.json` else `mode_c_asr_final_merged.json`, runs three LLM calls (span edits thinking-off / extract thinking-off / faithfulness judge thinking-on), validates edits in code, writes display artifacts. Existing `_persist_polish` and polish validators stay untouched.

**Tech Stack:** Python 3.10+, pytest, existing Qwen36LlmJudge / MockLlmJudge / vLLM generate path.

## Global Constraints

- Do **not** modify `stage2_asr/polish.py`, `stage2_asr/polish_prompt.py`, `stage2_asr/pass_a.py`, `stage2_asr/pass_b.py`, or polish validators/prompts.
- Do **not** overwrite `mode_c_asr_final.json` or `mode_c_polished.json`.
- Chinese and English are equal; code-switch must not be translated or deleted.
- Publish edits thinking **off**, T=0.1, `max_tokens=2048`; extract thinking **off**, T=0.1, `max_tokens=1024`; quality judge thinking **on**, T=0.1, `max_tokens=2048`.
- `--llm-enable-thinking` still controls Pass A/B / polish only.
- ITN toward compact form only; serial `伍柒叁` ≠ place-value `五百三十七`; TTS-style TN forbidden.
- `--stage llm` still skips polish **and** publish; `--stage all` runs polish then publish.

## File map

| File | Role |
|------|------|
| Create `stage2_asr/publish_itn.py` | Serial vs place-value ITN checker |
| Create `stage2_asr/publish_prompt.py` | Bilingual prompts for publish / extract / eval |
| Create `stage2_asr/publish.py` | Markers, validate/apply edits, run_publish, glossary merge, markdown |
| Create `tests/test_publish_itn.py` | ITN checker unit tests |
| Create `tests/test_publish.py` | Validator, run_publish, pipeline stage |
| Create `docs/glossary.json` | Empty seed `{ "terms": [] }` |
| Modify `stage2_asr/types.py` | `publish_batch_size`, `publish_eval`, `publish_eval_thinking` |
| Modify `stage2_asr/cli.py` | Additive flags + `publish` stage choice |
| Modify `stage2_asr/pipeline.py` | `_persist_publish` + stage `publish` / `all` tail only |
| Modify `stage2_asr/batch.py` | `needs_llm` includes publish; summary paths |
| Modify `stage2_asr/runners/mock_llm.py` | `publish` / `extract_terms` / `eval_publish` |
| Modify `stage2_asr/runners/llm_qwen36.py` | Same three methods; per-call thinking/max_tokens on `_generate` **defaults unchanged** for judge/polish |
| Modify `README.md` | Document `--stage publish` only |
| Modify `tests/test_progress_logs.py` | Assert `[publish]` on `stage=all` |

Do not add publish methods to DeepSeek (polish is Qwen-only today).

---

### Task 1: ITN checker

**Files:**
- Create: `stage2_asr/publish_itn.py`
- Test: `tests/test_publish_itn.py`

**Interfaces:**
- Produces: `itn_edit_allowed(span_asr: str, span_out: str) -> tuple[bool, str | None]`

- [ ] **Step 1: Write failing tests** for `伍柒叁`→`573` ok, `伍柒叁`→`五百三十七` fail, `五百三十七`→`537` ok, `573`→`five hundred seventy-three` fail, `三点` fail (ambiguous), `five seven three`→`573` ok, `百分之五十`→`50%` ok.

- [ ] **Step 2: Run pytest** — expect import/function missing.

- [ ] **Step 3: Implement checker** — classify serial (consecutive CN numerals / EN number-words, no 十百千万亿 / hundred|thousand|million) vs place-value vs arabic; allow compact same digit-string or same numeric value only.

- [ ] **Step 4: Tests pass.** Commit.

---

### Task 2: Publish edits + validator

**Files:**
- Create: `stage2_asr/publish.py` (validate/apply/markers/fillers/latin invariant)
- Test: `tests/test_publish.py` (edit tests first)

**Interfaces:**
- Produces: `TURN_MARK = "⟦t{i}⟧"` via `concat_meeting(texts: dict[int,str]) -> str` and `split_meeting(s: str) -> dict[int,str]`
- `FILLERS: frozenset[str]` bilingual
- `validate_publish_edits(edits, *, meeting: str, glossary_terms: list) -> tuple[bool, str | None]` — whole-payload fail on schema/marker; callers drop per-edit failures via `filter_publish_edits`
- `apply_publish_edits(meeting: str, edits: list[dict]) -> tuple[str, list[dict]]`
- `latin_runs(s: str) -> set[str]` length ≥ 2

Allowed kinds: `filler`, `repair`, `punc`, `latex`, `itn`.

- [ ] **Step 1: Failing tests** — filler `嗯`/`um`; repair CN/EN; latex `x平方`; reject marker edit; reject `GPU`→`显卡`; reject wrapping `Windows` in `$...$`; ITN via checker.

- [ ] **Step 2: Implement** filter+apply. Empty `span_out` allowed for `filler` only.

- [ ] **Step 3: Tests pass.** Commit.

---

### Task 3: Prompts + run_publish (edits, extract, eval, artifacts)

**Files:**
- Create: `stage2_asr/publish_prompt.py`
- Modify: `stage2_asr/publish.py` add `run_publish`, `merge_glossary`, `render_transcript`, `load_glossary`
- Modify: `stage2_asr/runners/mock_llm.py` add `publish`, `publish_many`, `extract_terms`, `eval_publish`
- Test: `tests/test_publish.py`

**Interfaces:**
- `run_publish(turns, texts, *, llm_judge, hotwords, glossary, config) -> tuple[dict[int,str], list[dict], dict, dict | None]`
  returns (texts, audits, glossary_out, eval_payload)
- Judge methods:
  - `publish(meeting, hotwords, glossary, unit_id) -> {edits: [...]}`
  - `extract_terms(meeting, glossary, unit_id) -> {keywords, rare_words, new_terms}`
  - `eval_publish(original, published, unit_id) -> {faithful, clearer, more_concise, easier, scores, issues}`
- Mock: drop fillers in meeting; ITN `伍柒叁`→`573` if present; extract `GPU` if present; eval `faithful: true` unless published lost a latin run.

Eval `faithful: false` reverts texts before markdown/glossary extract (extract runs on reverted text).

- [ ] **Step 1: Failing tests** for `run_publish` filler+ITN, extract merge seed-wins, eval revert, marker dump keeps original.

- [ ] **Step 2: Implement prompts + run_publish retries using `cfg.llm_max_retries`.**

- [ ] **Step 3: Tests pass.** Commit.

---

### Task 4: Wire stage without touching polish logic

**Files:**
- Modify: `types.py`, `cli.py`, `pipeline.py` (additive), `batch.py`, `llm_qwen36.py` (`_generate(..., enable_thinking=None, max_tokens=None)` defaults to `self.*` so polish/judge unchanged)
- Modify: `README.md`, `tests/test_progress_logs.py`
- Test: pipeline tests in `tests/test_publish.py`

**Interfaces:**
- `PipelineConfig.publish_batch_size: int = 1`, `publish_eval: bool = True`, `publish_eval_thinking: bool = True`, `glossary: dict | None = None`
- CLI: `--stage publish`, `--glossary`, `--publish-batch-size`, `--no-publish-eval`, `--no-publish-eval-thinking`
- `_persist_publish` reads polished else merged final; writes four artifacts; `_rewrite_edits_keep_other_passes` for `publish`/`extract`/`publish_eval`
- `stage=all` calls `_persist_publish` **after** existing `_persist_polish`
- `stage=publish` does not call polish

Qwen36: `publish`/`extract_terms`/`eval_publish` call `_generate` with spec max_tokens; eval passes `enable_thinking=True` unless `config.publish_eval_thinking` is false (pass via job kwargs / judge method arg `enable_thinking`).

- [ ] **Step 1: Failing pipeline tests** — `stage=publish` writes three files, does not clobber polished/final; `stage=all` writes published; `stage=llm` has no published; missing input raises.

- [ ] **Step 2: Wire CLI/pipeline/batch/qwen. Do not edit polish.py.**

- [ ] **Step 3: `pytest tests/test_publish.py tests/test_publish_itn.py tests/test_polish.py tests/test_progress_logs.py tests/test_pipeline_e2e_mock.py -q` all pass.**

- [ ] **Step 4: Commit.**

---

## Spec coverage

- Fluency span edits, bilingual fillers/repairs/punc/latex, ITN §2.1, glossary seed+extract, markdown, quality judge gate, inference table, code-switch invariant, batch `--stage publish`, artifacts list, polish frozen.
