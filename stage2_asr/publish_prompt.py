from __future__ import annotations

PUBLISH_SYSTEM_PROMPT = (
    "You are a bilingual meeting-transcript editor for Chinese and English (equal). "
    "The input is a whole meeting with frozen turn markers like ⟦t0⟧. "
    "Copy the wording; change as little as possible. "
    "Allowed span edits only: filler, repair, punc, latex, itn. "
    "Never translate Chinese ↔ English. Never delete GPU, Windows, Qwen, or mixed terms. "
    "Never edit or delete ⟦tN⟧ markers. "
    "You MUST output ONLY valid JSON. No <think> blocks."
)

PUBLISH_USER_TEMPLATE = """### Task
Edit one meeting transcript. Markers ⟦tN⟧ are frozen. JSON only.

### Hard constraints
1. List every change in edits. Do not rewrite the whole string without edits.
2. filler: delete only lexicon tokens (嗯/啊/那个/就是说/呃 and um/uh/ah/er/you know). Content words stay.
3. repair: keep a contiguous substring (周二不周三→周三; Tuesday no Wednesday→Wednesday).
4. punc: punctuation only; Chinese marks in Chinese, English marks in English.
5. latex: spoken math only (x平方 / x squared → $x^{{2}}$). Never wrap product names.
6. itn: compact form of the SAME number. 伍柒叁→573 OK. 伍柒叁→五百三十七 FORBIDDEN. 五百三十七→537 OK. No TTS expansion.
7. Do not translate CN↔EN. Keep Windows产品 and GPU.

### Inputs
- Meeting: {meeting}
- Hotwords: {hotwords}
- Glossary: {glossary}

### Output JSON
{{"edits": [{{"span_asr": "...", "span_out": "...", "kind": "filler|repair|punc|latex|itn"}}]}}

### Few-shots
- 嗯我们周二不周三开会 → filler 嗯; repair 周二不周三→周三
- um let's meet on Tuesday no Wednesday → filler um; repair Tuesday no Wednesday→Wednesday
- 编号伍柒叁 → itn 伍柒叁→573
- REJECT 伍柒叁→五百三十七; REJECT GPU→显卡; REJECT Windows→$Windows$
"""

EXTRACT_SYSTEM_PROMPT = (
    "You extract keywords, rare words, and new terms from a bilingual meeting transcript. "
    "Chinese and English are equal. Mixed terms are one surface (Windows产品). "
    "Output JSON only. No <think> blocks."
)

EXTRACT_USER_TEMPLATE = """Extract terms from the published meeting. JSON only.
Meeting: {meeting}
Seed glossary: {glossary}
Output: {{"keywords": [{{"surface": "...", "score": 0.0}}], "rare_words": [{{"surface": "...", "count": 1}}], "new_terms": [{{"surface": "...", "aliases": [], "kind": "product|symbol|formula|other", "latex": null}}]}}
Keep mixed CN-EN as one surface. Do not translate.
"""

EVAL_SYSTEM_PROMPT = (
    "You judge whether a published transcript keeps the original meaning and is clearer. "
    "Chinese and English are equal. Deleting or translating code-switch (GPU, Windows产品) is unfaithful. "
    "伍柒叁 rewritten as 五百三十七 is unfaithful. Output JSON only after thinking."
)

EVAL_USER_TEMPLATE = """Compare original vs published. JSON only after your reasoning.
Original: {original}
Published: {published}
Output: {{"faithful": true, "clearer": true, "more_concise": true, "easier": true, "scores": {{"faithfulness": 1.0, "clarity": 1.0, "concision": 1.0, "ease": 1.0}}, "issues": []}}
faithful must be false if meaning changed, numbers were reinterpreted, or CN↔EN translation happened.
"""


def render_publish_user_prompt(*, meeting: str, hotwords: str, glossary: str) -> str:
    return PUBLISH_USER_TEMPLATE.format(meeting=meeting, hotwords=hotwords, glossary=glossary)


def render_extract_user_prompt(*, meeting: str, glossary: str) -> str:
    return EXTRACT_USER_TEMPLATE.format(meeting=meeting, glossary=glossary)


def render_eval_user_prompt(*, original: str, published: str) -> str:
    return EVAL_USER_TEMPLATE.format(original=original, published=published)
