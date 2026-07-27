from __future__ import annotations

"""Official Pass A/B judge prompt contract (for real LLM adapters)."""

SYSTEM_PROMPT = (
    "You are a strict, conservative meeting transcript corrector. "
    "Your primary goal is FIDELITY TO SPOKEN PHONETICS, not creative writing. "
    "You MUST output ONLY valid JSON. Do not add, remove, or rephrase facts "
    "unless strictly supported by phonetic evidence (Tier A/B/C)."
)

USER_PROMPT_TEMPLATE = """### Task
Correct the ASR transcription for a meeting turn. Output JSON only.

### Constraints (CRITICAL)
1. OVERLAP={overlap_flag}. HEAVY_OVERLAP={heavy_overlap_flag}. If heavy_overlap, use 'moss' as the base text.
2. EVIDENCE LADDER (Strict Priority):
   - Tier A (Select/Merge): Choose the best-looking hypothesis.
   - Tier B (Exact Pinyin): Correct wrong characters but keep the exact tone-insensitive pinyin match.
   - Tier C (Fuzzy + Anchor): Allowed ONLY if Pinyin edit distance <= 2 AND anchored by (neighbor_draft OR meeting_draft OR hotword).
3. SPAN-LOCAL CHAR COUNT (MANDATORY): for every edit, |len(span_out)-len(span_asr)| <= 1.
   Never expand abbreviations (e.g., do not change '模型' to '大语言模型').
4. No open-world knowledge / context-only fixes without a pinyin link. If unsure, keep the original ASR text.

### Inputs
- Hypotheses (Raw + Pinyin):
{hypotheses_with_pinyin}
- Hotwords (Optional): {hotwords}
- Neighbor Draft (±10 min, capped): {neighbor_draft}
- Overlap Status: {overlap_flag}
- Heavy Overlap Status: {heavy_overlap_flag}

### Output JSON Schema
{{
  "text": "Final corrected text",
  "base_model": "moss|qwen|firered",
  "edits": [
    {{
      "span_asr": "original substring",
      "span_out": "corrected substring",
      "tier": "A|B|C|punct",
      "pinyin_asr": "...",
      "pinyin_out": "...",
      "anchor": "hyp|neighbor_draft|meeting_draft|hotword"
    }}
  ],
  "overlap": true|false
}}

### Few-shots
Positive (length-matched phonetic repairs — prefer these patterns):
1) 产用 → 采用  (Tier C, anchor=hyp|hotword; |Δlen|=0)
2) 单方接 → 单框架  (Tier C, anchor=hotword|neighbor_draft; |Δlen|=0)
3) 奔至 → 蹦字  (Tier C, anchor=neighbor_draft; |Δlen|=0)

Negative (must REJECT — span-local violation):
- Do NOT rewrite a bare digit like inserting unmatched-length tokens
  (e.g. span_asr="" / span_out="3", or expanding one syllable into a long phrase).
  If the only fix would break |len(span_out)-len(span_asr)| <= 1, keep the base ASR text.

Now process the following inputs strictly based on the rules above. Output JSON only.
"""


def render_user_prompt(
    *,
    hypotheses_with_pinyin: str,
    hotwords: str,
    neighbor_draft: str,
    overlap_flag: bool,
    heavy_overlap_flag: bool,
) -> str:
    return USER_PROMPT_TEMPLATE.format(
        overlap_flag=str(overlap_flag).lower(),
        heavy_overlap_flag=str(heavy_overlap_flag).lower(),
        hypotheses_with_pinyin=hypotheses_with_pinyin,
        hotwords=hotwords,
        neighbor_draft=neighbor_draft,
    )
