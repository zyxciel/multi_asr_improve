# Polish: Homophone Clusters and Entity-Subset Consistency (Design)

**Date:** 2026-09-04  
**Status:** Draft for spec review (do not implement until this file is approved)  
**Repo:** Stage-2 Multi-ASR + LLM Fusion  
**Related:** [Architecture](../../2026-07-24-stage2-multi-asr-llm-fusion-design.md), [README polish policy](../../../README.md)

## Goal

When the same spoken name or term is written differently across ASR engines or units (`张三风` / `涨三丰`), polish may unify **meeting-internal** variants using context. It must not invent a spelling that never appeared in hyps, and must not force two different words that merely sound alike into one surface.

This is a polish-only display pass. It does **not** change `mode_c_asr_final.json` or Pass A/B phonetic WER.

## Non-goals (v1)

- Do not enable thinking on polish **span-edit generation** (JSON only; existing `--llm-enable-thinking` stays off by default for generate).
- Do not use unconstrained `world` knowledge: the chosen surface must already appear as an ASR hyp string in the meeting.
- Do not auto-unify by frequency (minority → majority is not a rule).
- Do not put unambiguous terms in the polish prompt (one surface, no hyp disagreement → not a cluster).
- Do not treat a phonetic cluster as a rewrite atom. Cluster = recall. Rewrite and rollback = **entity subset**.
- Do not decide keep/change per mention in the partition call. Per-mention edits stay with polish span edits (neighbor context already exists).
- Do not change Pass A/B pinyin distance (still toneless, threshold 2). Cluster recall may annotate tone; it must not alter Pass A validators.
- Do not require a complete hotword list. Hotword `canonical|alias` remains a separate evidence channel.

## Why a cluster exists at all

A cluster is a **candidate bag** of ASR surfaces that might be the same word. It exists so the partition LLM sees global evidence (which surfaces occur in which turns / engines) in **one call per bag**, not one call per mention.

A cluster is **not** permission to rewrite every member to one spelling. False merges (`张三丰` the person vs `张三峰` another person) must be splittable.

## Pipeline

```text
hyps from asr_hypotheses.json (all engines, all units)
  → extract CJK surfaces (length ≥ 3)
  → build phonetic clusters (filters below)
  → drop clusters with < 2 distinct hyp surfaces
  → for each remaining cluster: one partition LLM call (thinking allowed)
       output: entity subsets + chosen hyp surface per subset | or "no subset"
       log pass=polish_cluster to llm_edits.jsonl
  → run_polish span edits as today
       extra allow-list: span in an approved subset may target that subset's chosen surface
       every edit still goes through validate_polish_edits (anchor/evidence)
  → deterministic check per approved entity subset
       if landed surfaces for that subset's edited spans are not unique → revert that subset's edits only
  → write mode_c_polished.json (WER files untouched)
```

Cost: **one partition call per multi-surface cluster**. Singleton surfaces never call the LLM and never occupy polish context as a cluster block.

## 1. Cluster construction (deterministic, from hyps)

Source of truth is **ASR hypotheses**, not the Pass B draft. Drafts may already have collapsed some variants; hyps still show engine/unit disagreement.

### Surface extraction

From each hyp text, take contiguous CJK runs. Emit every substring of length `L` where `3 ≤ L ≤ 6`, and also the full run if the run is longer than 6 (long names). Latin-only tokens are out of v1 (codeswitch already has `hyp` / `meeting_hyp`).

### Pairing / clustering

Two surfaces `A`, `B` may share a cluster only if **all** of:

1. CJK length ≥ 3 on both.
2. Toneless `pinyin_edit_distance(A, B) ≤ 2` (same numeric cap as Pass A; uses existing `pinyin_util.pinyin_edit_distance`).
3. They share a **character** prefix or suffix of at least one Han character (first char equal **or** last char equal). Homophone surnames written with different characters (`张` vs `章`) are **not** paired. Missed recall is accepted; false merge is not.
4. Extra brake: `dist / max(syllable_count(A), syllable_count(B), 1) ≤ 2/3`. Three-syllable pairs at distance 2 remain possible; long names at distance 2 stay easy.

Transitive closure of pairing builds a cluster.

### Disagreement gate (required)

A cluster is kept only if it contains **≥ 2 distinct writings** that each appear in hyps, and those writings come from **different engines or different units** (or both).

- 20 mentions of the same writing, all engines agree → **not a cluster**. No partition call. Do not inject into the polish prompt.
- 19× `张三丰` + 1× `张三风` in another unit's Qwen hyp → cluster (worth asking).

Moss Mode-C text counts as a hyp model (`moss`) when present.

### Tone annotation (not a hard reject)

`to_pinyin(text, tone=True)` (TONE3) already exists; Pass A stays `tone=False`. For each pair in a kept cluster, if toneless distance is small but TONE3 strings differ, mark the pair `tone_mismatch: true` in the partition prompt. Partition should be more willing to split those pairs. Do not drop them from recall solely for tone.

## 2. Partition call (one per kept cluster)

**Role:** group surfaces into **entity subsets**. Do not emit per-mention keep/change.

**Input:**

- Distinct surfaces and counts per `(model, unit_id)` / turn index.
- A few context snippets: turns (or hyp lines) where each surface occurred.
- Tone-mismatch flags.
- Instruction: split if they are different words; never invent a spelling.

**Output JSON (closed):**

```json
{
  "subsets": [
    {
      "surfaces": ["张三风", "涨三丰"],
      "canonical": "涨三丰",
      "same_entity": true,
      "reason": "..."
    },
    {
      "surfaces": ["张三峰"],
      "canonical": null,
      "same_entity": false,
      "reason": "different person"
    }
  ]
}
```

Rules:

- `canonical` is required when `same_entity` is true; it **must** be one of that subset's `surfaces` (a hyp-attested writing). Reject the subset if not.
- `same_entity: false` or a singleton leftover: no unify permission.
- Invalid JSON / schema: treat the whole cluster as **no subsets** (no extra edits from this cluster). Retry with existing `llm_max_retries` / backoff, then give up.

**Thinking:** allowed on this call only (`enable_thinking=True` for `pass=polish_cluster`), independent of polish generate.

**Audit:** append to `llm_edits.jsonl` with `"pass": "polish_cluster"`:

- cluster id / member surfaces
- raw subsets, `same_entity`, `reason`, `canonical`
- tone_mismatch flags
- ok / fallback

Polish span-edit rows stay `"pass": "polish"` so a failure can be attributed to **bad partition** vs **bad span edit**.

## 3. Polish span edits (unchanged contract + allow-list)

`run_polish` still produces `punc` / `entity` / `codeswitch` JSON with thinking **off**.

Additional allow-list, computed from approved subsets (`same_entity` and valid `canonical`):

- If `span_asr` is a surface in subset S and `span_out == S.canonical`, the edit **may** use `anchor=meeting_hyp` (or `hyp` if this unit's n-best already has `canonical`) provided `validate_polish_edits` agrees.
- Partition success only grants **eligibility**. If validator rejects (span not in text, missing evidence, CJK slack, etc.), drop that edit.
- Surfaces in a `same_entity: false` group must **not** be unified via this cluster channel. Existing neighbor/hyp entity repairs unrelated to the cluster remain allowed.

Polish prompt may include a short block listing **approved** mappings only (`张三风|涨三丰 → 涨三丰`), not raw unpartitioned clusters. Unapproved bags stay out of context.

Per-mention “should this occurrence change?” is **not** answered in the partition call. The span-edit LLM still sees neighbor_draft for that turn and may leave a mention unchanged. That is intended.

## 4. Consistency check (deterministic, entity-subset grain)

After applying polish edits, for **each approved subset S**:

- Collect landed text of spans that (a) belonged to S and (b) received a cluster-channel edit in this polish run.
- If those landed strings are not all equal to `S.canonical` (more than one distinct writing remains among them), **revert only S's cluster-channel edits** from this run. Other subsets and non-cluster polish edits stay.

Do **not** require the whole phonetic cluster to collapse to one writing. Unapproved surfaces legally coexist. Rolling back the whole cluster would undo a correct subset A because subset B was supposed to stay different.

Subsets that never produced edits are not checked for uniqueness (nothing to revert).

## 5. Config / CLI

- Default: on when running polish / `all`.
- `--no-polish-cluster` disables partition + cluster prompt block + subset check (polish as of 2026-09-04).
- No extra call when zero clusters survive the disagreement gate.

## 6. Tests (must exist before implementation is claimed done)

- Construction: 20 identical hyp surfaces → no cluster; two writings across units/engines → one cluster.
- Length 2 homophones (`意图`/`音图`) never cluster.
- Shared first character required: `会议`/`会意` (if they fail affix or length policy as specified) stay out or split; `张` vs `章` names do not pair.
- Partition invalid canonical (not in subset surfaces) → subset dropped.
- Allow-list: approved mapping still fails validator if `canonical` is absent from meeting hyps (should not happen if construction is correct; guard anyway).
- Two subsets in one cluster: A unified, B left mixed → check reverts nothing in B and does not revert A.
- Approved subset lands two writings after polish → revert **that subset only**.
- `llm_edits.jsonl` contains `polish_cluster` rows plus `polish` span rows.
- `mode_c_asr_final.json` unchanged.

## Invariants

1. Chosen `canonical` ∈ hyp surfaces of that entity subset.
2. Cluster recall ≠ rewrite permission.
3. Rollback grain = consistency grain = entity subset.
4. Partition call count = number of kept clusters, not mentions.
5. Phonetic WER artifact is never written by this pass.
