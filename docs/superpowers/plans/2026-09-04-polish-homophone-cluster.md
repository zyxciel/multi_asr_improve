# Polish Homophone Clusters Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** During polish, recall ASR full-run homophone variants, partition them into entity subsets with one LLM call per cluster, and allow span edits only toward a hyp-attested canonical — without touching WER files.

**Architecture:** New `stage2_asr/polish_cluster.py` owns deterministic cluster construction, partition-payload parsing, allow-list, subset consistency check, and leftover warnings. `run_polish` calls partition (thinking on) then existing span-edit polish (thinking off). Every cluster-channel edit still goes through `validate_polish_edits`. Rollback and leftover reporting are per **entity subset**, never the whole phonetic bag.

**Tech Stack:** Python 3.10+, pytest, existing `pinyin_util`, `Qwen36LlmJudge`, `MockLlmJudge`, `PipelineConfig` / CLI.

**Spec:** `docs/superpowers/specs/2026-09-04-polish-homophone-cluster-design.md` (approved).

## Global Constraints

- Do **not** overwrite `mode_c_asr_final.json`.
- Do **not** enable thinking on polish **span-edit** generate; partition call only (`pass=polish_cluster`, `enable_thinking=True`).
- Do **not** change Pass A/B `pinyin_edit_distance` (toneless, cap 2).
- Canonical must be a **full CJK run** that appears in some hyp of that entity subset.
- Cluster = recall. Rewrite permission = approved entity subset. Rollback grain = that subset’s **cluster-channel edits** only.
- Unedited leftovers in an approved subset are a **warning**, not a revert.
- Sliding-window substrings of a longer CJK run are never cluster members.
- Default **on** for polish/`all`; `--no-polish-cluster` restores 2026-09-04 polish behavior for A/B.
- Tests that involve partition **must mock** the judge; do not call a real LLM.

## File map

| File | Role |
|------|------|
| Create `stage2_asr/polish_cluster.py` | Full-run extract, pairing, clusters, parse partition, allow-list, check, leftover |
| Create `stage2_asr/polish_cluster_prompt.py` | Partition system/user templates (neutral tone-mismatch wording) |
| Create `tests/test_polish_cluster.py` | Construction + parse + allow-list + check (no pipeline) |
| Modify `tests/test_polish.py` | Wire-in: allow-list, leftover, WER file, mock partition |
| Modify `stage2_asr/polish.py` | `run_polish` partition → allow-list → tag cluster-channel edits → check |
| Modify `stage2_asr/polish_prompt.py` | Optional `{cluster_allow}` block of **approved mappings only** |
| Modify `stage2_asr/types.py` | `polish_cluster: bool = True` |
| Modify `stage2_asr/cli.py` | `--no-polish-cluster` |
| Modify `stage2_asr/pipeline.py` | Pass `hyp_records` into polish; write `polish_cluster` jsonl rows |
| Modify `stage2_asr/runners/llm_qwen36.py` | `partition_cluster(..., enable_thinking=True)` |
| Modify `stage2_asr/runners/mock_llm.py` | `partition_cluster` default empty / injectable |
| Modify `README.md` | Flag + A/B note |

Do not add partition to deleted DeepSeek. Do not change `pinyin_util.pinyin_edit_distance` defaults.

---

### Task 1: Full-run extract + pairing + cluster construction

**Files:**
- Create: `stage2_asr/polish_cluster.py`
- Test: `tests/test_polish_cluster.py`

**Interfaces:**
- Produces:
  - `extract_full_cjk_runs(text: str) -> list[str]` — maximal Han runs, length ≥ 3, **no** sliding windows
  - `pair_surfaces(a: str, b: str) -> bool` — spec §1 pairing (1)–(4)
  - `build_homophone_clusters(hyp_records: list[dict]) -> list[HomophoneCluster]`
  - `@dataclass HomophoneCluster`: `cluster_id: str`, `surfaces: tuple[str, ...]`, `hits: list[dict]` (each hit has `surface`, `model`, `unit_id`, `turn_indices`, `hyp_text`), `tone_mismatch_pairs: list[tuple[str, str]]`

Hyp record shape (same as `asr_hypotheses.json`): `{"unit_id": "unit_0000", "turn_indices": [0], "hyps": [{"model": "qwen", "text": "..."}]}`.

- [ ] **Step 1: Write failing tests** in `tests/test_polish_cluster.py`:

```python
from stage2_asr.polish_cluster import (
    build_homophone_clusters,
    extract_full_cjk_runs,
    pair_surfaces,
)

def test_extract_full_runs_no_windows():
    runs = extract_full_cjk_runs("找张三丰签字")
    assert runs == ["找张三丰签字"]
    assert "找张三丰" not in runs
    assert "张三丰" not in runs  # proper substring of a longer run

def test_length_2_never_pairs():
    assert pair_surfaces("意图", "音图") is False

def test_flagship_zhang_pairs():
    assert pair_surfaces("张三风", "涨三丰") is True

def test_zhang_vs_zhang_chapter_pairs():
    assert pair_surfaces("张三丰", "章三丰") is True

def test_identical_hyps_no_cluster():
    recs = [
        {"unit_id": "u0", "turn_indices": [0], "hyps": [{"model": "qwen", "text": "张三丰来了"}]},
        {"unit_id": "u1", "turn_indices": [1], "hyps": [{"model": "moss", "text": "张三丰走了"}]},
    ]
    assert build_homophone_clusters(recs) == []

def test_two_writings_across_engines_cluster():
    recs = [
        {"unit_id": "u0", "turn_indices": [0], "hyps": [{"model": "qwen", "text": "张三风"}]},
        {"unit_id": "u0", "turn_indices": [0], "hyps": [{"model": "firered", "text": "涨三丰"}]},
    ]
    # If a record has one hyps list with both models, that is also valid:
    recs = [
        {
            "unit_id": "u0",
            "turn_indices": [0],
            "hyps": [
                {"model": "qwen", "text": "张三风"},
                {"model": "firered", "text": "涨三丰"},
            ],
        }
    ]
    clusters = build_homophone_clusters(recs)
    assert len(clusters) == 1
    assert set(clusters[0].surfaces) == {"张三风", "涨三丰"}

def test_windows_from_long_run_not_members():
    recs = [
        {"unit_id": "u0", "turn_indices": [0], "hyps": [{"model": "qwen", "text": "找张三丰签字"}]},
        {"unit_id": "u1", "turn_indices": [1], "hyps": [{"model": "moss", "text": "签张三丰"}]},
    ]
    clusters = build_homophone_clusters(recs)
    members = {s for c in clusters for s in c.surfaces}
    assert "找张三丰" not in members
    assert "签张三丰" in members or not any("找张三丰签字" in (c.surfaces) and "签张三丰" in c.surfaces for c in clusters)
    # full runs 找张三丰签字 vs 签张三丰 must not pair (first/last syllables differ)

def test_full_run_phrase_residual_may_cluster():
    recs = [
        {"unit_id": "u0", "turn_indices": [0], "hyps": [{"model": "qwen", "text": "找张三丰"}]},
        {"unit_id": "u1", "turn_indices": [1], "hyps": [{"model": "moss", "text": "签张三丰"}]},
    ]
    clusters = build_homophone_clusters(recs)
    assert any(set(c.surfaces) == {"找张三丰", "签张三丰"} for c in clusters)

def test_transitive_closure_keeps_distant_pair():
    # Construct A-B and B-C legal, A-C dist > 2 if possible; if hard with real names,
    # use three surfaces where union-find still yields one cluster of size 3.
    recs = [
        {"unit_id": "u0", "turn_indices": [0], "hyps": [{"model": "qwen", "text": "张三风"}]},
        {"unit_id": "u1", "turn_indices": [1], "hyps": [{"model": "moss", "text": "涨三丰"}]},
        {"unit_id": "u2", "turn_indices": [2], "hyps": [{"model": "firered", "text": "章三丰"}]},
    ]
    clusters = build_homophone_clusters(recs)
    assert len(clusters) == 1
    assert set(clusters[0].surfaces) == {"张三风", "涨三丰", "章三丰"}
```

Fix the `test_two_writings` body to a single `recs` assignment (do not leave the duplicate). `test_windows_from_long_run_not_members`: assert no cluster contains both `找张三丰签字` and `签张三丰`, and `找张三丰` is never a member.

- [ ] **Step 2: Run** `python -m pytest tests/test_polish_cluster.py -q --tb=line`  
  Expected: FAIL (import missing) or collection error.

- [ ] **Step 3: Implement** `stage2_asr/polish_cluster.py`:
  - Runs: `re.findall(r"[\u4e00-\u9fff]+", text)` then keep `len(run) >= 3`.
  - `pair_surfaces`: `pinyin_edit_distance` ≤ 2; first or last **toneless** syllable equal; `dist / max(syl, 1) ≤ 2/3`.
  - Union-find on eligible (surface, model, unit_id) hits; cluster if ≥2 distinct surfaces and at least two writings from different `model` or different `unit_id`.
  - `tone_mismatch_pairs`: pairs in the cluster where `to_pinyin(a, True) != to_pinyin(b, True)`.
  - Do **not** re-filter pairwise inside the cluster after union-find.

- [ ] **Step 4: Tests pass.** Commit: `Build polish homophone clusters from full CJK ASR runs only.`

---

### Task 2: Parse partition payload → allow-list

**Files:**
- Modify: `stage2_asr/polish_cluster.py`
- Test: `tests/test_polish_cluster.py`

**Interfaces:**
- Consumes: `HomophoneCluster`
- Produces:
  - `@dataclass EntitySubset`: `surfaces: frozenset[str]`, `canonical: str | None`, `same_entity: bool`, `reason: str`
  - `parse_partition_payload(raw: dict | None, cluster: HomophoneCluster) -> list[EntitySubset]`  
    Invalid JSON/type → `[]`. `same_entity: true` without `canonical` in that subset’s surfaces → **drop that subset**. Surfaces not listed in any subset are **not** returned (implicit no-permission singletons).
  - `cluster_allow_list(subsets: list[EntitySubset]) -> dict[str, str]`  
    For each `same_entity` subset, map every surface except canonical… actually map **all** surfaces in S including canonical (`canonical -> canonical` is a no-op). Map `span_asr -> canonical` for every surface in S. **Do not** include `same_entity: false` surfaces.

- [ ] **Step 1: Failing tests**

```python
from stage2_asr.polish_cluster import (
    HomophoneCluster,
    cluster_allow_list,
    parse_partition_payload,
)

def _cluster(*surfaces: str) -> HomophoneCluster:
    return HomophoneCluster(
        cluster_id="c0",
        surfaces=tuple(surfaces),
        hits=[],
        tone_mismatch_pairs=[],
    )

def test_parse_drops_canonical_not_in_subset():
    raw = {"subsets": [{"surfaces": ["张三风", "涨三丰"], "canonical": "张三丰", "same_entity": True, "reason": "x"}]}
    assert parse_partition_payload(raw, _cluster("张三风", "涨三丰")) == []

def test_parse_unlisted_surface_not_in_allow_list():
    raw = {"subsets": [{"surfaces": ["张三风", "涨三丰"], "canonical": "涨三丰", "same_entity": True, "reason": "x"}]}
    subs = parse_partition_payload(raw, _cluster("张三风", "涨三丰", "张三峰"))
    allow = cluster_allow_list(subs)
    assert allow.get("张三风") == "涨三丰"
    assert "张三峰" not in allow

def test_false_multi_surface_empty_allow_list():
    raw = {"subsets": [{"surfaces": ["找张三丰", "签张三丰"], "canonical": None, "same_entity": False, "reason": "verbs"}]}
    subs = parse_partition_payload(raw, _cluster("找张三丰", "签张三丰"))
    assert cluster_allow_list(subs) == {}
```

- [ ] **Step 2: pytest** those three — FAIL until parse exists.

- [ ] **Step 3: Implement parse + allow-list.** `same_entity` accept JSON true/false. Empty `reason` ok. Ignore unknown keys.

- [ ] **Step 4: Pass + commit:** `Parse cluster partition JSON into an entity-subset allow-list.`

---

### Task 3: Subset hard check + leftover warning

**Files:**
- Modify: `stage2_asr/polish_cluster.py`
- Test: `tests/test_polish_cluster.py`

**Interfaces:**
- Produces:
  - `cluster_channel_edit(edit: dict, allow: dict[str, str]) -> bool`  
    True iff `str(edit.get("span_asr"))` in allow and `str(edit.get("span_out")) == allow[span_asr]`.
  - `revert_subset_edits(texts: dict[int, str], applied: list[dict], subset: EntitySubset) -> dict[int, str]`  
    `applied` items must include `turn_index`, `span_asr`, `span_out`, `prev_text` (text before that edit) or `start_char` sufficient to undo. Prefer storing `turn_index` + snapshot `before` on each cluster-channel audit and restore those turns’ cluster-channel spans only. Simplest v1: each applied cluster-channel audit has `turn_index` and `text_before_edit` for that turn; reverting S restores those turns to `text_before_edit` **only if** the audit belongs to S (span_asr in subset.surfaces). If a turn had mixed cluster-channel edits from two subsets, only undo audits whose span_asr ∈ S.surfaces (re-apply remaining edits from `text_before_first_S_edit` is messy). **v1 rule:** audits carry `turn_index`, `span_asr`, `span_out`, `start_char`, `end_char` after apply; revert S by replaying inverse replacements on current texts for those audits (sort by start_char ascending after inverse? Inverse: put `span_asr` back at `start_char` length `len(span_out)`). Implement inverse carefully; tests will lock it.
  - `subset_edit_texts_unique(applied_for_s: list[dict], canonical: str) -> bool` — all `span_out` == canonical.
  - `leftover_mentions(texts: dict[int, str], subset: EntitySubset, applied_for_s: list[dict]) -> list[dict]` — turns where a subset surface still occurs as a substring and was not fully replaced by cluster-channel edits to canonical. Warning rows: `{"pass": "polish_cluster", "path": "leftover_mix", "surfaces": ..., "turn_index": i, "canonical": ...}`.

- [ ] **Step 1: Failing tests**

```python
def test_hard_check_reverts_only_that_subset():
    # texts after edits: turn0 canonical, turn1 two different cluster-channel span_outs for subset S
    # implement using revert helper on a tiny fixture
    ...

def test_leftover_unedited_does_not_fail_unique():
    # 3 mentions, 2 edited to canonical, 1 still old writing → unique(applied) True; leftover non-empty
    ...
```

Fill the tests with real dicts once `revert_subset_edits` signature is written in this same task (do not leave `...` in the committed test file).

- [ ] **Step 2: pytest FAIL.**

- [ ] **Step 3: Implement check + leftover + revert inverse.**

- [ ] **Step 4: Pass + commit:** `Check entity-subset polish edits and warn on leftover mixed writings.`

---

### Task 4: Partition prompt + judge methods

**Files:**
- Create: `stage2_asr/polish_cluster_prompt.py`
- Modify: `stage2_asr/runners/llm_qwen36.py`
- Modify: `stage2_asr/runners/mock_llm.py`
- Test: `tests/test_polish_cluster.py` (prompt wording) and `tests/test_polish.py` if needed for generate_fn

**Interfaces:**
- Produces:
  - `PARTITION_SYSTEM_PROMPT: str`
  - `render_partition_user_prompt(*, cluster: HomophoneCluster) -> str`
  - `Qwen36LlmJudge.partition_cluster(self, *, cluster: HomophoneCluster | dict, unit_id: str = "") -> dict`  
    `_generate(..., pass_name="polish_cluster", enable_thinking=True, max_tokens=2048)` then `_parse_json`.
  - `MockLlmJudge.partition_cluster(self, **kwargs) -> dict`  
    Default `{"subsets": []}`. Optional `self.partition_fn` callable for tests.

Prompt **must** say: tone-mismatch is a **weak** signal among context and co-occurrence; do not weight it above contextual evidence; never invent a spelling; output the closed JSON schema from the spec; listing every surface is optional.

- [ ] **Step 1: Test** `test_partition_prompt_tone_is_weak` — blob must contain `weak` / `do not weight` (English) and must **not** contain `more willing to split`.

- [ ] **Step 2: FAIL then implement prompts + methods.**

- [ ] **Step 3: Test mock default empty allow-list via parse.**

- [ ] **Step 4: Commit:** `Add cluster partition prompts and judge methods with thinking on.`

---

### Task 5: Wire `run_polish`

**Files:**
- Modify: `stage2_asr/polish.py` (`run_polish`, `_run_polish_batched`, `_try_polish` / `_validate_polish_raw` only as needed)
- Modify: `stage2_asr/polish_prompt.py` — add `{cluster_mappings}` default `"(none)"` to the user template **Approved mappings only**
- Modify: `stage2_asr/pipeline.py` `_persist_polish` — pass `hyp_records` into `run_polish`; write cluster audits
- Test: `tests/test_polish.py`

**Interfaces:**
- Extend `run_polish(..., hyp_records: list[dict] | None = None)`  
  If `cfg.polish_cluster` is False or judge lacks `partition_cluster` or `hyp_records` is empty: skip partition (current behavior).
  Else: `clusters = build_homophone_clusters(hyp_records)`; for each cluster, retry `partition_cluster` with existing backoff; `parse_partition_payload`; merge allow-lists (if two subsets map the same span_asr to different canonicals, **drop that span_asr** from the allow-list — conflict); collect `polish_cluster` audits.
  Pass `cluster_mappings` into polish user prompt (pretty `张三风|涨三丰 → 涨三丰`).
  After a polish payload is accepted, tag each applied edit with `cluster_channel` via `cluster_channel_edit`.
  After all turns: for each approved subset, hard-check applied cluster-channel edits; revert that subset only if needed; append leftover warning audits.
- `run_polish` return stays `tuple[dict[int, str], list[dict]]` with both `pass=polish` and `pass=polish_cluster` rows in the same list (pipeline already dumps by pass name if it groups; `_rewrite_edits_keep_other_passes` is per pass — `_persist_polish` must write `polish` and `polish_cluster` separately, or split the list by `a.get("pass")`).

- [ ] **Step 1: Failing tests** in `tests/test_polish.py`:

```python
def test_run_polish_allow_list_empty_when_mock_partition_false():
    # hyps 找张三丰 / 签张三丰 as full runs; mock partition same_entity false
    # polish LLM tries span 找张三丰 → 签张三丰 entity; must NOT apply via cluster channel
    # (may still fail validator or be non-cluster entity — assert text unchanged for those spans)

def test_run_polish_cluster_does_not_clobber_asr_final(tmp_path):
    # existing pipeline polish test: after cluster wiring, mode_c_asr_final.json still equal

def test_run_polish_leftover_warns_without_revert():
    # mock partition same_entity true canonical 涨三丰; polish edits only turn 0; turn 1 still 张三风
    # turn 0 stays 涨三丰; audits include leftover_mix
```

- [ ] **Step 2: FAIL, then wire.** Keep polish generate `enable_thinking` unset/false. Partition uses `enable_thinking=True` inside `partition_cluster` only.

- [ ] **Step 3: `_persist_polish`:**  
  `run_polish(..., hyp_records=hyp_records)`  
  Split audits: `_rewrite_edits_keep_other_passes(..., "polish", polish_rows)` and `_rewrite_edits_keep_other_passes(..., "polish_cluster", cluster_rows)`.

- [ ] **Step 4: Full `tests/test_polish.py tests/test_polish_cluster.py` green.** Commit: `Run partition before polish span edits and honor the cluster allow-list.`

---

### Task 6: CLI, config, README

**Files:**
- Modify: `stage2_asr/types.py` — `polish_cluster: bool = True`
- Modify: `stage2_asr/cli.py` — `--no-polish-cluster` `store_true`; `_pipeline_config` sets `polish_cluster=not args.no_polish_cluster`
- Modify: `README.md` — staged polish bullet: default on; `--no-polish-cluster` freezes pre-cluster polish for A/B
- Test: `tests/test_cli.py` or a tiny argparse test if one exists; else assert `PipelineConfig().polish_cluster is True`

- [ ] **Step 1: Test default True / flag False.**

- [ ] **Step 2: Implement flag + README sentence** (behavior change vs 2026-09-04 polish).

- [ ] **Step 3: `python -m pytest tests -q --tb=line`** — full suite green.

- [ ] **Step 4: Commit:** `Default-on polish clusters with --no-polish-cluster for A/B freezes.`

- [ ] **Step 5: Set spec status** in `docs/superpowers/specs/2026-09-04-polish-homophone-cluster-design.md` to `Approved for implementation`. Commit with the README if not already: `Mark homophone-cluster spec approved.`

---

## Spec coverage (self-review)

| Spec item | Task |
|-----------|------|
| Full-run extract, no windows | T1 |
| Pairing 1–4, pinyin affix, ratio | T1 |
| Transitive closure | T1 |
| Disagreement gate ≥2 writings, engines/units | T1 |
| Tone mismatch weak signal | T4 prompt + T1 pairs |
| Partition one call/cluster, thinking on | T4–T5 |
| Coverage / false multi-surface | T2 |
| Allow-list + existing validator | T5 |
| Hard check vs leftover (a) | T3, T5 |
| Entity-subset revert only | T3, T5 |
| `llm_edits.jsonl` both passes | T5 |
| WER file untouched | T5 |
| `--no-polish-cluster`, README | T6 |
| Residual phrase runs + mock false group | T1 + T2 + T5 |
| Flagship / 张章 pair | T1 |

No placeholders left in task interfaces. Conflict: two subsets mapping the same `span_asr` to different canonicals → drop that key (not in spec; required for a total allow-list function).
