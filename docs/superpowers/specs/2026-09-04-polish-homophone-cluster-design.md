# Polish: Homophone Clusters and Entity-Subset Consistency (Design)

**Date:** 2026-09-04  
**Status:** Approved for implementation  
**Repo:** Stage-2 Multi-ASR + LLM Fusion  
**Related:** [Architecture](../../2026-07-24-stage2-multi-asr-llm-fusion-design.md), [README polish policy](../../../README.md)

## Goal

When the same spoken name or term is written differently across ASR engines or units (`张三风` / `涨三丰`), polish may unify **meeting-internal** variants using context. It must not invent a spelling that never appeared in hyps, and must not force two different words that merely sound alike into one surface.

Unifying is **best-effort**. Per-mention polish may leave an occurrence unchanged (neighbor context). Mixed writing in the final display is then a **reported leftover**, not a silent spec hole and not a reason to revert correct edits elsewhere in the same subset.

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
- Do not pair **sliding-window substrings** of a longer CJK run. Only **full runs** that appear as a complete CJK span in at least one hyp are cluster members. Context windows like `找张三丰` / `签张三丰` cut from `找张三丰签字` must not form a cluster (that path can rewrite a verb with a passing validator).
- Do not hard-fail an approved subset because polish left some mentions unedited. That leftover is reported, not rolled back (see §4).

## Why a cluster exists at all

A cluster is a **candidate bag** of ASR surfaces that might be the same word. It exists so the partition LLM sees global evidence (which surfaces occur in which turns / engines) in **one call per bag**, not one call per mention.

A cluster is **not** permission to rewrite every member to one spelling. False merges (`张三丰` the person vs `张三峰` another person) must be splittable.

## Pipeline

```text
hyps from asr_hypotheses.json (all engines, all units)
  → extract full CJK runs (length ≥ 3) only — no 3–6 sliding windows
  → pair / cluster (filters below)
  → drop clusters with < 2 distinct hyp surfaces
  → for each remaining cluster: one partition LLM call (thinking allowed)
       output: entity subsets + chosen hyp surface per subset | or "no subset"
       uncovered surfaces → no-permission singletons
       log pass=polish_cluster to llm_edits.jsonl
  → run_polish span edits as today
       extra allow-list: span in an approved subset may target that subset's chosen surface
       every edit still goes through validate_polish_edits (anchor/evidence)
  → deterministic check per approved entity subset
       among cluster-channel edits in S: if landed writings are not unique → revert S's cluster-channel edits only
       leftover unedited mentions in S: do not revert; emit a warning record
  → write mode_c_polished.json (WER files untouched)
```

Cost: **one partition call per multi-surface cluster**. Singleton surfaces never call the LLM and never occupy polish context as a cluster block.

## 1. Cluster construction (deterministic, from hyps)

Source of truth is **ASR hypotheses**, not the Pass B draft. Drafts may already have collapsed some variants; hyps still show engine/unit disagreement.

### Surface extraction (full-run only)

From each hyp text, take contiguous CJK **runs** (maximal sequences of Han characters). A surface is eligible iff:

- its CJK length ≥ 3, and
- it appears as a **complete run** in at least one hyp in the meeting (the run as stored, not a proper substring of a longer run).

Do **not** emit every 3–6 character sliding window. Windows share the suffix/prefix pattern of real name variants (`找张三丰` vs `签张三丰` as **substrings** of a longer run) and can pass the entity validator after rewriting a verb. Full-run filtering **removes the common sliding-window class**. Residual risk remains: if two engines emit `找张三丰` and `签张三丰` as **complete runs** (punctuation/tokenization boundaries), they can still pair; that case is left to partition (context snippets) and span edit (neighbors), not to construction. True names that ASR emitted as their own run (`涨三丰`, `张三风`) still qualify.

Latin-only tokens are out of v1 (codeswitch already has `hyp` / `meeting_hyp`).

### Pairing / clustering

Two **eligible** surfaces `A`, `B` may share a cluster only if **all** of:

1. CJK length ≥ 3 on both.
2. Toneless `pinyin_edit_distance(A, B) ≤ 2` (same numeric cap as Pass A; uses existing `pinyin_util.pinyin_edit_distance`).
3. They share a **toneless pinyin syllable** prefix or suffix: first syllable equal **or** last syllable equal (`to_pinyin(..., tone=False)` split on spaces). Compare syllables, not Han characters. This is what makes the flagship pair reachable: `张三风` / `涨三丰` share first syllable `zhang` even though 张≠涨 and 风≠丰. Character-level first-or-last would only recall mid-string edits and would **exclude** the typical ASR name pattern (wrong surname character or wrong final character).
4. Extra brake: `dist / max(syllable_count(A), syllable_count(B), 1) ≤ 2/3`. Three-syllable pairs at distance 2 remain possible; long names at distance 2 stay easy.

Short homophones stay out via length ≥ 3 (`会议` / `会意` are two characters). Construction does **not** block homophone surnames written with different characters (`张三丰` vs `章三丰`): first syllable `zhang` matches, last syllable `feng` matches, distance 0 → they **enter the cluster**. Partition plus span-edit context plus the validator decide whether that is two people or one name's variant. Hard-blocking 张/章 at construction would also block 张/涨.

**Transitive closure** of pairing builds a cluster. Surfaces inside one cluster need **not** all pairwise satisfy (1)–(4): if A–B and B–C are legal edges, A and C join the same bag even when `pinyin_edit_distance(A, C) > 2`. That is intended. Partition splits the bag; implementations must **not** secretly drop A–C pairs by re-checking pairwise distance inside the cluster.

### Disagreement gate (required)

A cluster is kept only if it contains **≥ 2 distinct writings** that each appear as eligible full runs in hyps, and those writings come from **different engines or different units** (or both).

- 20 mentions of the same writing, all engines agree → **not a cluster**. No partition call. Do not inject into the polish prompt.
- 19× `张三丰` + 1× `张三风` in another unit's Qwen hyp → cluster (worth asking).

Moss Mode-C text counts as a hyp model (`moss`) when present.

### Tone annotation (not a hard reject)

`to_pinyin(text, tone=True)` (TONE3) already exists; Pass A stays `tone=False`. For each pair in a kept cluster, if toneless distance is small but TONE3 strings differ, mark the pair `tone_mismatch: true` in the partition prompt. Treat tone-mismatch as **one weak signal** among context and co-occurrence; **do not weight it above contextual evidence**. Surname variants often differ in tone (`张` zhang1 vs `涨` zhang3); instructing the model to prefer splitting on tone-mismatch would systematically suppress true unifications. Do not drop pairs from recall solely for tone.

## 2. Partition call (one per kept cluster)

**Role:** group surfaces into **entity subsets**. Do not emit per-mention keep/change.

**Input:**

- Distinct eligible surfaces and counts per `(model, unit_id)` / turn index.
- A few context snippets: turns (or hyp lines) where each surface occurred as a full run.
- Tone-mismatch flags.
- Instruction: split if they are different words; never invent a spelling; listing a surface is optional (see coverage below).

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

### Coverage

`subsets` need not list every cluster surface. Any surface **absent from all subsets** is a **no-permission singleton**: same as `same_entity: false`. It must not appear on the cluster allow-list.

### `same_entity: false` groups

A false group may contain one or many surfaces. Semantics: **each listed surface is independent; none may be unified with any other via this cluster channel** (not with siblings in the same false group, and not with other subsets). Implementations must not require false groups to be singletons. A multi-surface false group is a compact way to say “these are not one entity.”

### Other rules

- `canonical` is required when `same_entity` is true; it **must** be one of that subset's `surfaces` (a hyp-attested **full-run** writing). Reject the subset if not.
- Invalid JSON / schema: treat the whole cluster as **no subsets** (no extra edits from this cluster). Retry with existing `llm_max_retries` / backoff, then give up.

**Thinking:** allowed on this call only (`enable_thinking=True` for `pass=polish_cluster`), independent of polish generate.

**Audit:** append to `llm_edits.jsonl` with `"pass": "polish_cluster"`:

- cluster id / member surfaces
- raw subsets, `same_entity`, `reason`, `canonical`
- uncovered surfaces (implicit singletons)
- tone_mismatch flags
- ok / fallback

Polish span-edit rows stay `"pass": "polish"` so a failure can be attributed to **bad partition** vs **bad span edit**.

## 3. Polish span edits (unchanged contract + allow-list)

`run_polish` still produces `punc` / `entity` / `codeswitch` JSON with thinking **off**.

Additional allow-list, computed from approved subsets (`same_entity` and valid `canonical`):

- If `span_asr` is a surface in subset S and `span_out == S.canonical`, the edit **may** use `anchor=meeting_hyp` (or `hyp` if this unit's n-best already has `canonical`) provided `validate_polish_edits` agrees.
- Partition success only grants **eligibility**. If validator rejects (span not in text, missing evidence, CJK slack, etc.), drop that edit.
- No unify permission for: `same_entity: false` surfaces (including multi-surface false groups), uncovered surfaces, or rejected subsets. Existing neighbor/hyp entity repairs **unrelated** to the cluster remain allowed.

Polish prompt may include a short block listing **approved** mappings only (`张三风|涨三丰 → 涨三丰`), not raw unpartitioned clusters. Unapproved bags stay out of context.

Per-mention “should this occurrence change?” is **not** answered in the partition call. The span-edit LLM still sees neighbor_draft for that turn and may leave a mention unchanged. That is intended (see §4 leftover).

## 4. Consistency check (deterministic, entity-subset grain)

After applying polish edits, for **each approved subset S** two different things happen. They must not be collapsed into one rule.

### Hard check (edits that ran)

Collect landed text of spans that (a) belonged to S and (b) received a **cluster-channel** edit in this polish run.

If those landed strings are not all equal to `S.canonical` (the applied edits disagree with each other or with the chosen surface), **revert only S's cluster-channel edits** from this run. Other subsets and non-cluster polish edits stay.

### Leftover report (mentions polish left alone)

Mentions of S's surfaces that **did not** receive a cluster-channel edit may still show the old writing next to `canonical` in the final display. That follows mention autonomy and is **not** a hard failure.

v1 choice: **(a) accept leftover + warn.** Emit a deterministic record (stderr log and/or a `polish_cluster` audit row) listing, for each approved subset, remaining mixed writings and turn indices. Do **not** revert S's successful edits because other mentions were correctly left unchanged. Downstream / humans can see the mix; A/B experiments can count leftover rate.

Do **not** require the whole phonetic cluster to collapse to one writing. Unapproved surfaces legally coexist. Rolling back the whole cluster would undo a correct subset A because subset B was supposed to stay different.

Subsets that never produced edits are not hard-checked for uniqueness (nothing to revert). They may still appear in the leftover report if the subset was approved and the draft still mixes S's surfaces.

## 5. Config / CLI

- Default: **on** when running polish / `all`. This **changes polish output** versus the 2026-09-04 baseline. README must state the flag and that A/B / WER-adjacent display diffs need `--no-polish-cluster` to freeze the old polish behavior.
- `--no-polish-cluster` disables partition + cluster prompt block + subset check + leftover report (polish as of 2026-09-04).
- No extra call when zero clusters survive the disagreement gate.

## 6. Tests (must exist before implementation is claimed done)

- Construction: 20 identical hyp surfaces → no cluster; two writings across units/engines → one cluster.
- Length 2 homophones (`意图`/`音图`) never cluster.
- Flagship pair: full runs `张三风` and `涨三丰` (different engines or units) **must** form a kept cluster (first syllable `zhang`).
- `张三丰` vs `章三丰` **do pair** into a cluster (same first/last syllable). Construction must not drop them. Partition (mock) may split; they must not be auto-unified without a `same_entity: true` subset.
- **Full-run filter:** from hyp A `找张三丰签字` and hyp B `签张三丰`, sliding windows `找张三丰` / `签张三丰` are **not** members. Eligible surfaces are the full runs `找张三丰签字` and `签张三丰`, which do not pair (first syllables differ, last syllables 字 vs 丰 differ).
- **Full-run residual:** hyps that **are** the complete runs `找张三丰` and `签张三丰` (last syllable `feng` matches) **may** cluster. Drive the test with a **mock partition** that returns a `same_entity: false` group (or omits a unify subset). Assert the cluster-channel **allow-list is empty** for those surfaces — this is a permission test, not a live-LLM judgment. Do not require construction to drop the pair.
- **Transitive closure:** A–B and B–C legal, A–C distance > 2 → one cluster still; partition (or a mock partition) may split; construction must not drop C by pairwise revalidation.
- Partition invalid canonical (not in subset surfaces) → subset dropped.
- **Coverage:** partition omits a cluster surface → omitted surface has no allow-list permission.
- **`same_entity: false` multi-surface group:** no cluster-channel unify among listed surfaces; they may remain mixed in the output.
- Allow-list: approved mapping still fails validator if `canonical` is absent from meeting hyps (should not happen if construction is correct; guard anyway).
- Two subsets in one cluster: A unified, B left mixed → check reverts nothing in B and does not revert A.
- Approved subset: some mentions edited to canonical, some left unedited → **no** revert of the successful edits; leftover warning lists the unedited mentions.
- Approved subset lands two **different** writings **among cluster-channel edits** → revert **that subset's cluster-channel edits only**.
- `llm_edits.jsonl` contains `polish_cluster` rows plus `polish` span rows.
- `mode_c_asr_final.json` unchanged.

## Invariants

1. Chosen `canonical` ∈ hyp **full-run** surfaces of that entity subset.
2. Cluster recall ≠ rewrite permission.
3. Rollback grain = consistency grain = entity subset (hard check applies to cluster-channel **edits**, not to unedited leftovers).
4. Partition call count = number of kept clusters, not mentions.
5. Phonetic WER artifact is never written by this pass.
6. Sliding-window substrings of a longer CJK run are never cluster members.
7. Unlisted partition surfaces and `same_entity: false` surfaces have no cluster-channel unify permission.
8. Pairing affix is toneless **pinyin syllable** (first or last), not Han character identity.
