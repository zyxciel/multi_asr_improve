from __future__ import annotations

"""Polish prompt: display + hyp/context/world-knowledge recovery.

Pass A/B stay phonetic (span-local + pinyin). This pass may change word
identity when an ASR hyp, neighbors, or meeting-consistent world knowledge
supports it. No char-count or pinyin cap.
"""

POLISH_SYSTEM_PROMPT = (
    "You are a meeting-transcript display and recovery editor. "
    "The input is the phonetic Pass A/B final, which is often still wrong on "
    "code-switched English, named entities, and topic-level substitutions "
    "because those passes forbid |Δlen|>1 and context-only fixes. "
    "Your job: punctuation, entity orthography, code-switched terms, ITN, "
    "AND recovering the intended words using (1) the three ASR hypotheses, "
    "(2) neighbor/meeting context, (3) LLM world knowledge consistent with "
    "this meeting. "
    "You MUST output ONLY valid JSON. "
    "Do NOT output chain-of-thought, analysis, or <think> blocks."
)

POLISH_USER_TEMPLATE = """### Task
Polish one meeting turn. Output JSON only.

This pass is NOT bound by Pass A/B span-local char-count (|Δlen|≤1) or pinyin
edit-distance. Those rules already blocked valid repairs such as 温度→Windows.

### Evidence ladder (prefer earlier rungs)
1. hyp: span_out (or a close canonical form) already appears in another ASR hypothesis for this turn/unit. Strongest evidence. Use even if lengths differ a lot.
2. neighbor_draft / meeting_draft / hotword: topic or name consistency with neighbors or hotwords.
3. world: world knowledge that is the only reading coherent with this meeting (e.g. a product-name discussion where 温度 is a mishearing of Windows). Do not use world knowledge to invent facts that the meeting never implied.

### Allowed edit kinds (use these exact strings)
- punc: Chinese punctuation (，。？！、：；).
- entity: named entity / topic term already intended in the meeting (爱情→娃娃亲 when neighbors discuss 娃娃亲).
- codeswitch: mixed Chinese-English / product names (温度→Windows when a hyp has Windows; gpu→GPU).
- itn: spoken numbers/dates/times/money/percentages → written form.

### Constraints
1. Every change MUST be listed in edits. Empty edits cannot rewrite the turn.
2. span_asr MUST be an exact substring of the input (or "" for insertion at start_char).
3. Prefer small spans. Do not summarize or add clauses that nobody said.
4. Do not expand abbreviations into definitions (GPU ↛ Graphics Processing Unit).
5. If truly unsure, keep the input (edits=[]).

### Inputs
- Turn index: {turn_index}
- Current phonetic-final text: {text}
- ASR hypotheses (MOSS / Qwen / FireRed for this turn's unit; may include unit_text):
{hypotheses}
- Hotwords: {hotwords}
- Neighbor turns (same meeting, capped): {neighbor_draft}

### Output JSON schema
{{
  "text": "polished turn text",
  "edits": [
    {{
      "span_asr": "exact substring of the input",
      "span_out": "replacement (length unrestricted)",
      "kind": "punc|entity|codeswitch|itn",
      "anchor": "hyp|neighbor_draft|meeting_draft|hotword|world",
      "start_char": 0
    }}
  ]
}}

### Few-shots
1) punc: 大家好明天见 → 大家好，明天见。
2) itn: 明天下午三点五十开会 → 三点五十 → 3:50
3) codeswitch casing: gpu → GPU
4) codeswitch + hyp (MUST APPLY): text 以前那个温度的问题; qwen hyp 以前那个Windows的问题 → 温度 → Windows, kind=codeswitch, anchor=hyp. |Δlen|=5 is allowed.
5) entity + neighbor (MUST APPLY): text 他们说的爱情到底怎么办; neighbor 娃娃亲这件事 → 爱情 → 娃娃亲, kind=entity, anchor=neighbor_draft. No pinyin link required.
6) REJECT: 很好 → 非常好 (synonym, no hyp/context need)
7) REJECT: 微信 → WeChat when no hyp/neighbor used English and the speaker used Chinese only

Now polish the turn. Output JSON only.
"""


def format_polish_hypotheses(hypotheses) -> str:
    """Render hyp list (Hypothesis objects or dicts) for the polish prompt."""
    if not hypotheses:
        return "(none)"
    lines: list[str] = []
    for h in hypotheses:
        if h is None:
            continue
        if isinstance(h, dict):
            model = str(h.get("model") or "?")
            text = str(h.get("text") or "")
            meta = h.get("meta") if isinstance(h.get("meta"), dict) else {}
        else:
            model = str(getattr(h, "model", "?") or "?")
            text = str(getattr(h, "text", "") or "")
            meta = getattr(h, "meta", None) or {}
            if not isinstance(meta, dict):
                meta = {}
        unit_text = str(meta.get("unit_text") or "")
        extra = f" [unit: {unit_text}]" if unit_text and unit_text != text else ""
        lines.append(f"- {model}: {text}{extra}")
    return "\n".join(lines) if lines else "(none)"


def render_polish_user_prompt(
    *,
    text: str,
    neighbor_draft: str,
    hotwords: str,
    turn_index: int,
    hypotheses: str = "(none)",
) -> str:
    return POLISH_USER_TEMPLATE.format(
        turn_index=turn_index,
        text=text,
        hotwords=hotwords,
        neighbor_draft=neighbor_draft,
        hypotheses=hypotheses,
    )
