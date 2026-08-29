# Stage-2 Publish: Fluency, Inline Math, Glossary (Design)

**Date:** 2026-08-29  
**Status:** Approved for implementation (pending spec review)  
**Repo:** Stage-2 Multi-ASR + LLM Fusion  
**Related:** [Architecture](../../2026-07-24-stage2-multi-asr-llm-fusion-design.md), [README polish policy](../../../README.md)

## Goal

Turn a phonetic meeting transcript into **accurate, fluent, display-formatted text** (Chinese, English, or mixed) without touching WER artifacts.

Publish must:

- Smooth the **whole recording** (moderate): drop fillers, resolve explicit self-corrections, stitch readable sentences with necessary punctuation, and apply **value-preserving ITN** for readability. Keep spoken wording otherwise; do not paraphrase into publication prose.
- Insert **inline LaTeX math** (`$...$`) for spoken formulas. Whole-recording file is **Markdown + math**, not a `.tex` document.
- Load a **seed glossary JSON** (terms + special spellings + optional `latex`) and write an **enriched** glossary with LLM-extracted keywords, rare words, and new terms.
- Preserve timestamps and `speaker_id` on a per-turn published JSON **and** emit a concatenated `transcript.md`.

**Chinese and English are equal.** A meeting may be Chinese-only, English-only, or code-switched; the same rules apply. Neither language is a guest of the other. **Code-switch is a hard constraint:** do not translate CN↔EN, and do not delete mixed terms that survived polish (see §5.2). Accents and dialects are handled by existing multi-ASR plus glossary **aliases**, not by a new ASR model in this spec.

## Non-goals (v1)

- Do not overwrite `mode_c_asr_final.json` or `mode_c_polished.json`.
- Do not compute corpus CER / cpCER on published text.
- Do not rewrite numbers in a way that changes their meaning (serial ≠ place-value; see §2.1). Polish remains no-ITN. Publish **may** do value-preserving ITN toward a compact written form; it may **not** expand compact digits into spoken words (TTS-style TN).
- Do not emit a compile-ready `.tex` file or Unicode-only symbol substitution as the math path.
- Do not replace `docs/hotwords.txt`; hotwords remain ASR / Pass A/B / polish hints.
- Do not shard a single meeting across multiple LLM prompts (`--publish-batch-size` > 1 packs **meetings**, not turns).
- Do not add DSP / waveform smoothing.
- Do not enable thinking on publish span-edits or extract (quality judge thinking is on; see §5.1).

## Architecture

Keep evidenced polish frozen. Add `--stage publish` after it.

```text
mode_c.json + wav
  → ASR → Pass A → Pass B
  → mode_c_asr_final.json              # WER / cpCER (never overwritten)
  → merge same-speaker units
  → evidenced polish (unchanged)
  → mode_c_polished.json               # optional; used if present
  → --stage publish
       1. LLM span edits (filler / repair / punc / latex / itn)
       2. LLM extract (keywords / rare_words / new_terms)
       3. LLM quality judge (faithfulness gate)
       → mode_c_published.json
       → transcript.md
       → glossary.json
       → publish_eval.json
```

`--stage all` runs publish after polish. `--stage llm` stays Pass A+B only. `--stage publish` is rerunnable from work-dir artifacts (no ASR / Pass A/B / polish reload).

**Input priority:** `mode_c_polished.json` if present, else `mode_c_asr_final_merged.json`. If both are missing, fail the sample with a path error (same style as polish requiring Pass B).

## Components

### 1. `run_publish`

New module beside `stage2_asr/polish.py`. One meeting = one document.

1. Concatenate merged-turn texts with frozen markers `⟦t{i}|{speaker_id}⟧` before each turn `i` (`speaker_id` is sanitized: no `⟦⟧|`). Different speakers stay visible to the LLM so a multi-person meeting is not one unlabeled stream.
2. Call LLM `publish(...)` (**thinking off**, JSON only). Prompt includes the full meeting, seed glossary, and hotwords.
3. Validate and apply span edits. `judgment.text` / a whole-string rewrite without edits is untrusted (same contract as polish).
4. Split the edited string back to turns by the markers.

New judge method: `publish` / `publish_many` (mirror `polish` / `polish_many`). Mock judge implements both.

### 2. Edit kinds (closed set)

| kind | Rule | Example |
|------|------|---------|
| `filler` | `span_out` is empty; `span_asr` ∈ filler lexicon | `嗯` / `啊` / `那个` / `就是说`; `um` / `uh` / `ah` / `you know` |
| `repair` | `span_out` is a **contiguous substring** of `span_asr` and strictly shorter; both the retracted words and the correction appear in `span_asr` | `周二不周三` → `周三`; `Tuesday no Wednesday` → `Wednesday` |
| `punc` | Word/CJK/digit skeleton unchanged; use Chinese marks in Chinese spans and English marks in English spans; do not force `。` onto English | `大家好明天见` → `大家好，明天见`; `lets meet tomorrow` → `let's meet tomorrow` |
| `latex` | `span_out` contains `$...$`; symbols/commands from seed glossary or the built-in math lexicon | `x平方` → `$x^{2}$`; `x squared` → `$x^{2}$` |
| `itn` | Compact written form of the **same** number; serial and place-value readings must not be swapped (see §2.1) | `伍柒叁` → `573`; `五百三十七` → `537`; `百分之五十` → `50%`; `five hundred thirty-seven` → `537` |

Publish does not merge or split turns. “Stitching” is punctuation inside a turn (`punc`). A span cannot cross a marker, so two speakers cannot be fused into one edit. Self-corrections that already sit in **one** same-speaker merged turn still apply (`周二不周三` → `周三`). If A says `周二` and B says `不周三`, that is two turns, not a repair. Publish does **not** redo polish `entity` / `codeswitch` repairs, and must **not undo** them.

**Multi-speaker:** consecutive same-speaker turns are already merged before polish. The publish concat is therefore typically A, B, A, B. The LLM must treat `speaker_id` changes as hard boundaries (no monologue punctuation, no borrowing B’s words to finish A). A filler must not wipe a turn whose remaining `_core` would be empty (listener backchannels such as a whole-turn `嗯` / `um` stay).

**Filler lexicon:** closed **bilingual** list in code — Chinese `嗯` / `啊` / `那个` / `就是说` / `呃` and English `um` / `uh` / `ah` / `er` / `you know` / `like` (discourse `like` only when the span is exactly that token, not `I like this`). Plus any glossary entries with `"kind": "filler"`. A filler span must be **only** lexicon tokens (optional surrounding punctuation). Content words in either language (`会议`, `meeting`, `GPU`, `Windows`, `Qwen`) are never fillers.

**Built-in math lexicon (small, in code):** spoken patterns in **both** languages — `平方` / `squared` → `^2`, `下标` / `subscript` → `_` — plus glossary `kind=symbol` / `kind=formula` with a `latex` field (`阿尔法` / `alpha` → `\alpha`).

### 2.1 ITN (readability, meaning locked)

Polish still forbids number rewrites (WER). Publish allows **ITN toward a compact written form** only when a deterministic checker says the meaning is unchanged.

Two readings, never mixed:

| Reading | How to detect | Allowed `span_out` | Forbidden |
|---------|----------------|--------------------|-----------|
| **Serial** (digit-by-digit) | Two or more consecutive Chinese numerals (`一二三` / `伍柒叁`) with **no** `十百千万亿点`; or English number-words with **no** `hundred` / `thousand` / `million` (`five seven three`) | The **same digit string** in Arabic: `伍柒叁` → `573`, `five seven three` → `573` | Place-value expansion: `伍柒叁` → `五百七十三` or `五百三十七`; `573` → `five hundred seventy-three` |
| **Place-value** (quantity) | Contains `十百千万亿` or English `hundred` / `thousand` / `million` / `percent` | The **same numeric value** in compact form: `五百三十七` → `537`; `百分之五十` → `50%`; `five hundred thirty-seven` → `537` | A different value: `五百三十七` → `573`; serializing into a different digit order |

**Hard fail (drop the edit), matching the product example:** `伍柒叁` → `五百三十七` (serial 5-7-3 vs quantity 537). Also fail `伍柒叁` → `五百七十三` (serial must not become a quantity phrase).

Other locks:

- Direction is **compact only**. TTS-style TN is forbidden: `573` → `五百七十三` / `five hundred seventy-three`; `0.61` → `zero point sixty-one`.
- If ambiguous, keep the source: `三点` (clock vs score vs 3.0), `three o'clock` vs `3:00` only when the span is an explicit clock phrase (`三点钟` / `three o'clock`), not bare `三点`.
- Digit **order** is meaning: `[5,7,3]` ≠ `[5,3,7]`.
- `punc` / `latex` still must not change the digit sequence; number changes go through `itn` only.
- Checker lives in code (numeral maps + serial vs place-value parse). The LLM does not self-certify meaning.

### 3. Validator

**Whole-payload reject** (retry, then skip smoothing for the meeting):

- Edits not a list of dicts, or missing `span_asr` / `span_out` / `kind`.
- Unknown `kind`.
- Any edit whose span overlaps a marker `⟦tN|speaker⟧` or deletes/alters a marker.

**Per-edit drop** (rest of payload may apply):

- `span_asr` not found in the meeting string; overlapping spans; empty `span_asr` insertions.
- `filler` whose `span_asr` is not in the lexicon, or whose application would leave that turn with an empty `_core` (whole-turn backchannel).
- `repair` whose `span_out` is not a contiguous substring of `span_asr`, or not strictly shorter.
- `punc` that changes `_core` (letters / CJK / digits).
- `latex` with unknown TeX command/symbol (not in glossary `latex` fields and not in the built-in lexicon).
- `itn` that fails the §2.1 checker (serial/place-value mismatch, digit-order change, spoken expansion, ambiguous span).
- Number changes under `punc` / `latex` / `filler` / `repair` (those kinds must keep the digit sequence).
- **Language break** (any kind except allowed `itn` / `latex`): translating English → Chinese or Chinese → English is always dropped. A Latin run of length ≥ 2 present in `span_asr` missing from `span_out` (case-insensitive) is dropped, except a `latex` edit whose `span_asr` is a glossary `kind=symbol|formula` surface or a built-in math token, or an `itn` edit that passes §2.1 (so `five seven three` → `573` is allowed). Mixed terms (`Windows产品`) must stay mixed. `punc` must not delete spaces inside a Latin token, and must not replace English punctuation with Chinese (or the reverse) inside a span that is otherwise unchanged.

### 4. LLM extract

Second call, on the **already-smoothed** meeting (fillers and retracted repairs gone). **Thinking off.** Schema:

```json
{
  "keywords": [{"surface": "...", "score": 0.0}],
  "rare_words": [{"surface": "...", "count": 1}],
  "new_terms": [{"surface": "...", "aliases": [], "kind": "product|symbol|formula|other", "latex": null}]
}
```

Keywords, rare words, and new terms may be Chinese, English, or mixed. Mixed CN–EN terms are **one** `surface` (`Windows产品`, `Qwen3.8`), not split into Chinese / English rows. Latin-only product names stay Latin; Chinese-only terms stay Chinese.

Merge into `glossary.json`: seed `surface` wins on collision. Extracted-only rows get `"source": "extract"`; seed rows keep `"source": "seed"`. Invalid extract JSON: retry, then write seed-only glossary with empty extract lists.

### 5. LLM quality judge (default on)

Third call. Compares unsmoothed concatenation vs published meeting. Does **not** rewrite. **Thinking is on** for this call only (semantic faithfulness). JSON after `</think>` drives the gate; leaked think blocks are stripped the same way as Pass A/B.

```json
{
  "faithful": true,
  "clearer": true,
  "more_concise": true,
  "easier": true,
  "scores": {"faithfulness": 0.0, "clarity": 0.0, "concision": 0.0, "ease": 0.0},
  "issues": []
}
```

Scores are floats in `[0, 1]`. Booleans are the gate/report bits; do not infer booleans from scores.

**Faithfulness gate:** if `faithful` is false, **revert published turns to the unsmoothed input** for that meeting. Markdown and glossary are produced from the reverted text. Log `publish_eval.rejected`. Clarity / concision / ease are reported only; they never revert.

Deleting, translating, or wrapping a code-switch term in `$...$` without a formula glossary hit **must** set `faithful: false`. Fusing two speakers, or deleting a listener’s only backchannel turn, **must** also set `faithful: false`.

`--no-publish-eval` skips this call (no gate). `--no-publish-eval-thinking` keeps the judge but sets `enable_thinking=False` (debug / latency). Judge JSON invalid: retry, then do not revert (treat as no score).

Write `publish_eval.json` always when the judge ran.

### 5.1 LLM inference (locked)

Publish uses the same Qwen3.8 judge process as Pass A/B / polish. Do **not** raise temperature. Do **not** turn thinking on for span edits.

| Call | `enable_thinking` | Temperature | `max_tokens` |
|------|-------------------|-------------|--------------|
| publish (span edits) | **false** | 0.1 | **2048** |
| extract | **false** | 0.1 | **1024** |
| quality judge | **true** (default) | 0.1 | **2048** |

- 2048 on publish: whole-meeting edit lists exceed the Pass A/B default of 1024.
- 2048 on the quality judge: thinking tokens would starve a 512 budget; JSON must still fit after CoT.
- `--llm-enable-thinking` continues to control **Pass A/B / polish only**. It does **not** enable thinking on publish edits or extract. Eval thinking is the `--no-publish-eval-thinking` switch above.
- Existing JSON extract (`extract_json_and_reasoning`) stays the only parser. Thinking never becomes the edit list.

### 5.2 Bilingual and code-switch preservation

Neither language may absorb the other.

Invariant after applying publish edits (before eval):

- Every Latin run of length ≥ 2 in the unsmoothed meeting still occurs in the published meeting (case-insensitive), except runs fully covered by an allowed `latex` edit or an allowed `itn` edit (English number-words may become Arabic digits).
- Non-filler CJK content is not deleted except as part of an allowed `repair`.
- Mixed CN–EN spans keep both scripts (`Windows产品`, `用 GPU 训练`).
- English-only turns stay English; Chinese-only turns stay Chinese.

Prompt is **bilingual with equal few-shots**, not a Chinese prompt with English extras. Include at least:

- CN filler/repair/punc: `嗯我们周二不周三开会` → `我们周三开会`
- EN filler/repair/punc: `um let's meet on Tuesday no Wednesday` → `let's meet on Wednesday`
- Mixed: `以前那个 Windows产品`, `用 GPU 训练`, `Qwen3.8`
- Math both languages: `x平方` / `x squared` → `$x^{2}$`
- ITN: `伍柒叁` → `573`, `五百三十七` → `537`; REJECT `伍柒叁` → `五百三十七`

Deleting, translating, or wrapping a code-switch or English/Chinese content term in `$...$` without a formula glossary hit **must** set eval `faithful: false`. An `itn` that the §2.1 checker would reject (including `伍柒叁` → `五百三十七`) **must** also set `faithful: false`.

### 6. Markdown renderer

`transcript.md`: for each published turn, a heading with `speaker_id` and `[start–end]`, then that turn’s `text` (inline `$math$` already in the string). `mode_c_published.json` uses the same turn grid as the merge input; `text` is the smoothed string. No separate `latex_text` field (inline-only choice).

## CLI and batch

Additive flags; existing polish / Pass A/B flags unchanged.

```text
stage2-asr run ... --stage publish \
  --glossary docs/glossary.json \
  --publish-batch-size 1

stage2-asr run-batch \
  --wav-benchmark ... --mode-c-benchmark ... --work-root ... \
  --backend real --enable-real --stage publish \
  --glossary docs/glossary.json
```

| Flag | Default | Meaning |
|------|---------|---------|
| `--glossary` | unset (empty seed) | Seed glossary JSON path |
| `--publish-batch-size` | `1` | Meetings packed into one `publish_many` (and matching extract/eval batches). `1` = one meeting at a time. |
| `--no-publish-eval` | off | Skip quality judge |
| `--no-publish-eval-thinking` | off | Quality judge with thinking off |

`run-batch --stage publish` is a normal stage: each `{dataset}/{stem}` reuses that sample’s `work-dir`. Cross-sample packing uses `--publish-batch-size N` the same way polish uses `--polish-batch-size` (snapshot N samples). v1 does not split one long meeting.

Empty turns and turns below `min_asr_seconds` keep original text but still get a marker so indices stay aligned.

## Artifacts

Per sample `work-dir`:

| File | Role |
|------|------|
| `mode_c_published.json` | Merged turn grid; `text` = published (or reverted) |
| `transcript.md` | Whole-recording Markdown + `$math$` |
| `glossary.json` | Seed ∪ extract |
| `publish_eval.json` | Quality-judge payload (absent if `--no-publish-eval`) |
| `llm_edits.jsonl` | Append `pass=publish` / `pass=extract` / `pass=publish_eval` |
| `llm_infer.jsonl` | Same traces as Pass A/B / polish |
| `pass_stats.json` | Add `publish`, `extract`, `publish_eval` counts (`n_edits` by kind, `n_reverted`, `n_fallback`) |

## Seed glossary schema

```json
{
  "terms": [
    {
      "surface": "Qwen3.8",
      "aliases": ["千问三点八"],
      "kind": "product",
      "latex": null
    },
    {
      "surface": "alpha",
      "aliases": ["阿尔法"],
      "kind": "symbol",
      "latex": "\\alpha"
    }
  ]
}
```

`keywords` / `rare_words` may be omitted in the seed; publish always writes them on output (`[]` if extract failed).

Hotwords file is still passed via `--hotwords` and included in the publish prompt as extra term hints. It is not merged into `glossary.json` unless a hotword also appears as an extracted `new_term` / keyword.

## Error policy (summary)

| Failure | Behavior |
|---------|----------|
| Missing polished and missing merged final | Fail sample |
| Publish schema / marker violation | Retry ≤2; then skip smoothing (published = input turns) |
| Per-edit rule fail | Drop that edit |
| Extract invalid JSON | Retry; seed-only glossary |
| Eval invalid JSON | Retry; do not revert |
| Eval `faithful: false` | Revert turns; still write all artifacts from reverted text |

`llm_max_retries` from `PipelineConfig` applies to each of the three calls independently.

## Testing

Do **not** score WER on published text.

Mock tests (no weights), next to `tests/test_polish.py`:

1. Filler removed in **both** languages when in lexicon (`嗯`, `um`); unknown filler dropped.
2. Repair contiguous-substring applied for CN (`周二不周三` → `周三`) and EN (`Tuesday no Wednesday` → `Wednesday`); non-substring rejected.
3. LaTeX from glossary / built-in lexicon applied for `x平方` and `x squared`; unknown TeX dropped.
3b. ITN: `伍柒叁` → `573` and `五百三十七` → `537` apply; `伍柒叁` → `五百三十七` and `伍柒叁` → `五百七十三` dropped; `573` → `five hundred seventy-three` dropped; bare `三点` kept.
4. Edit touching `⟦t1|s1⟧` dumps payload; turns unchanged after retries.
4b. Multi-speaker: concat embeds `⟦t{i}|{speaker_id}⟧`; `周二` then other-speaker `不周三` is not a repair; a whole-turn listener `嗯` is not deleted.
5. Extract merge: seed `surface` wins; bad JSON → seed-only. Mixed term stays one `surface`.
6. Quality judge `faithful: false` reverts; `faithful: true` keeps edits. Judge output with a `<think>` wrapper still parses JSON.
7. Bilingual / code-switch: `Windows产品` / `GPU` / English content words survive filler+punc; `GPU`→`显卡` and `会议`→`meeting` are dropped; wrapping `Windows` in `$...$` is dropped unless glossary formula.
8. `--stage publish` writes published JSON + `transcript.md` + `glossary.json` and does not overwrite `mode_c_asr_final.json` or `mode_c_polished.json`.
9. `run-batch --stage publish --limit 1` in mock: one sample dir contains the three files.

Success for this spec:

1. Phonetic WER artifacts unchanged.
2. Mock tests above pass.
3. One mock meeting round-trips to published JSON + Markdown + enriched glossary + `publish_eval.json`.

## Implementation notes (for the plan)

- Reuse LLM JSON extract, infer log, retry, and `PipelineConfig` patterns from polish.
- Do not relax polish validators or prompts (polish stays no-ITN).
- ITN checker is a unit-tested function (`serial` digit string vs `place_value` number); do not ask the LLM whether two number spans “mean the same thing”.
- Judge adapter must support **per-call** `enable_thinking` / `max_tokens` (eval on + 2048; publish/extract off). Do not reuse a process-wide thinking flag for all three calls.
- Default filler list is **bilingual and equal** (Chinese and English tokens, neither marked extra).
- Prompt language: bilingual instructions; equal Chinese, English, and mixed few-shots. Do not write a Chinese-only system prompt.
