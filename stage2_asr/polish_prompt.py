from __future__ import annotations

"""Polish prompt: minimal evidenced edits on the phonetic-final ASR text.

Pass A/B stay phonetic (span-local + pinyin). This pass may substitute an
entity or code-switched term when a hyp, neighbor, other-turn ASR, or
hotword canonical supports the replacement. No ITN. No world-knowledge-only swaps.
"""

POLISH_SYSTEM_PROMPT = (
    "You are a conservative meeting-transcript editor. "
    "The input is the phonetic Pass A/B final. Copy it; change as little as possible. "
    "Allowed work: (1) necessary punctuation only, (2) minimal entity substitutions, "
    "(3) Chinese-English code-switch corrections — each with explicit evidence. "
    "Do not add content, insert sentences, delete repeated characters, rewrite style, "
    "or normalize numbers. "
    "You MUST output ONLY valid JSON. "
    "Do NOT output chain-of-thought, analysis, or <think> blocks."
)

POLISH_USER_TEMPLATE = """### Task
Polish one meeting turn. Base = the phonetic-final text below. Output JSON only.

### Hard constraints (violations are discarded)
1. Base is the current phonetic-final text. List every change in edits. Empty edits cannot rewrite the turn.
2. Minimal entity substitution or correction only. Do NOT add content, insert sentences, continue the utterance, paraphrase, or delete repetitive characters (好好好 stays 好好好).
3. Chinese→Chinese substitution: |len(span_asr)-len(span_out)| ≤ 2 UNLESS span_out is a hotword canonical and span_asr is pinyin-near (edit distance ≤ 2) or a listed alias (`canon|alt`). Examples: 张三风→张三丰 (|Δ|=0); 玛→玛曲县 when 玛曲县 is a hotword.
4. Chinese→English / mixed CN–EN may change character count because scripts differ. span_out MUST appear in this turn's hyp, a neighbor, another turn's ASR (meeting_hyp), or be a hotword canonical (case-insensitive for English). Examples: 温度→Windows when a hyp or other-turn ASR contains Windows.
5. Punctuation: only necessary additions or fixes of existing marks. Do not punctuate every turn. Do not punctuate backchannels (嗯、对、好、好的). Do not invent English punctuation in Chinese.
6. NO number normalization / ITN. Keep spoken and written numbers as in the input. REJECT 三点→3点, 0.61→zero point sixty-one, 532→five hundred thirty-two, 百分之五十→50%.
7. Do not expand abbreviations (GPU ↛ Graphics Processing Unit). Do not swap entities from world knowledge without a hyp/neighbor/meeting_hyp/hotword span.
8. If this turn's ASR hypotheses already match the base, keep the input (edits=[]). If unsure, keep the input.

### Evidence (required on entity / codeswitch except latin casing like gpu→GPU)
- hyp: span_out already appears in an ASR hypothesis for this turn/unit.
- neighbor_draft / meeting_draft: span_out already appears in a neighbor turn or the meeting draft.
- meeting_hyp: span_out already appears in another turn's ASR hypothesis (other-turn n-best).
- hotword: span_out is a provided hotword canonical; span_asr is a listed alias or pinyin-near that canonical. Do not pick an unrelated hotword.
Write a short evidence string stating where span_out was found. Do not use world.

### Allowed edit kinds (exact strings)
- punc: necessary Chinese punctuation only; must not change letters/CJK/digits.
- entity: Chinese entity correction with evidence; |Δlen|≤2 unless hotword pinyin-near/alias (张三风→张三丰, 爱情→娃娃亲).
- codeswitch: mixed CN-EN / product names with evidence (温度→Windows, 温度的问题→Windows产品); or latin casing of a token already in the text (gpu→GPU).

### Inputs
- Turn index: {turn_index}
- Current phonetic-final text: {text}
- ASR hypotheses (MOSS / Qwen / FireRed for this turn's unit; may include unit_text):
{hypotheses}
- Other-turn ASR forms (meeting_hyp evidence; may be truncated):
{meeting_hyps}
- Hotwords: {hotwords}
- Neighbor turns (same meeting, capped): {neighbor_draft}

### Output JSON schema
{{
  "text": "polished turn text",
  "edits": [
    {{
      "span_asr": "exact substring of the input",
      "span_out": "replacement",
      "kind": "punc|entity|codeswitch",
      "anchor": "hyp|neighbor_draft|meeting_draft|meeting_hyp|hotword",
      "evidence": "where span_out was found",
      "start_char": 0
    }}
  ]
}}

### Few-shots
1) punc (necessary clause mark only): 大家好明天见 → 大家好，明天见 (do NOT also append 。 if not needed)
2) codeswitch casing: gpu → GPU (latin already in the text)
3) codeswitch + hyp: text 以前那个温度的问题; qwen hyp 以前那个windows产品 → 温度的问题 → Windows产品, kind=codeswitch, anchor=hyp, evidence="qwen hyp contains windows产品"
4) entity + neighbor, |Δ|=0: text 找张三风签字; neighbor 张三丰已经到了 → 张三风 → 张三丰, kind=entity, anchor=neighbor_draft, evidence="neighbor contains 张三丰"
5) entity + meeting_hyp: text 找张三风签字; this-turn hyps also 张三风; other-turn qwen 张三丰已经到了 → 张三风 → 张三丰, kind=entity, anchor=meeting_hyp, evidence="other-turn qwen hyp contains 张三丰"
6) entity + neighbor, |Δ|=1: text 他们说的爱情到底怎么办; neighbor 娃娃亲这件事 → 爱情 → 娃娃亲, kind=entity, anchor=neighbor_draft, evidence="neighbor contains 娃娃亲"
7) entity + hotword pinyin-near: text 去玛开会; hotword 玛曲县 → 玛 → 玛曲县, kind=entity, anchor=hotword, evidence="hotword canonical 玛曲县"
8) REJECT: 爱情 → 娃娃亲的事 (|Δlen|=3 exceeds ±2 slack and 娃娃亲的事 is not a hotword); 很好 → 非常好 (synonym without span_out in hyp/neighbor/meeting_hyp)
9) REJECT: 微信 → WeChat when no hyp/neighbor/meeting_hyp/hotword contains WeChat
10) REJECT: 张三风 → 昇腾 just because 昇腾 is a hotword (not pinyin-near, not an alias)
11) REJECT: 三点 → 3点 ; 0.61 → zero point sixty-one ; 532 → five hundred thirty-two (no number normalization)
12) REJECT: insert a new sentence; 好好好 → 好 ; empty span_asr insertions

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
    meeting_hyps: str = "(none)",
) -> str:
    return POLISH_USER_TEMPLATE.format(
        turn_index=turn_index,
        text=text,
        hotwords=hotwords,
        neighbor_draft=neighbor_draft,
        hypotheses=hypotheses,
        meeting_hyps=meeting_hyps,
    )
