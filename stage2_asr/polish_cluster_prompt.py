from __future__ import annotations

"""Partition prompt for one homophone cluster (spec §2).

One LLM call per kept cluster groups the candidate surfaces into **entity
subsets**. It does NOT decide per-mention keep/change (that stays with the
polish span-edit pass). Thinking is allowed on this call only.

Spec: docs/superpowers/specs/2026-09-04-polish-homophone-cluster-design.md §2.
"""
import json

from stage2_asr.polish_cluster import HomophoneCluster

PARTITION_SYSTEM_PROMPT = (
    "You are a careful meeting-transcript entity resolver. "
    "You receive a candidate bag of Chinese ASR surface writings that may or may not "
    "be the same spoken name/term. Group them into entity subsets. "
    "Do NOT decide per-mention keep/change; that is a later pass. "
    "Tone mismatch between two surfaces (e.g. 张 zhang1 vs 涨 zhang3) is a WEAK signal "
    "among context and co-occurrence evidence. Do not weight it above contextual "
    "evidence; do not let tone alone push you toward splitting. "
    "Never invent a spelling that did not appear as an ASR surface. "
    "You MUST output ONLY valid JSON matching the closed schema. "
    "Do NOT output chain-of-thought outside the JSON."
)

_PARTITION_USER_TEMPLATE = """### Task
Partition the candidate surfaces of ONE homophone cluster into entity subsets.
A cluster is a recall bag of ASR writings that might be the same word; it is NOT
permission to merge. Output JSON only.

### Hard constraints (violations discard the answer)
1. Group surfaces that refer to the same entity into one subset with `same_entity: true`; group surfaces that are different entities with `same_entity: false`. A multi-surface `same_entity: false` group is allowed (compact "these are not one entity").
2. `canonical` is required when `same_entity` is true and MUST be one of that subset's `surfaces` (a hyp-attested full-run writing). Set `canonical: null` when `same_entity` is false.
3. Never invent a spelling. Every surface you list must come from the cluster's surfaces below.
4. Tone mismatch is a WEAK signal among context and co-occurrence. Do not weight it above contextual evidence. Surname variants often differ in tone (张 zhang1 vs 涨 zhang3); do not let tone alone push you toward splitting.
5. Listing every cluster surface is OPTIONAL. Any surface absent from all subsets is an implicit no-permission singleton (treated like `same_entity: false`); it must not be unified via this cluster.
6. Output the closed JSON schema only. No extra keys.

### Output JSON schema (closed)
{{
  "subsets": [
    {{
      "surfaces": ["张三风", "涨三丰"],
      "canonical": "涨三丰",
      "same_entity": true,
      "reason": "..."
    }},
    {{
      "surfaces": ["张三峰"],
      "canonical": null,
      "same_entity": false,
      "reason": "different person"
    }}
  ]
}}

### Inputs
- Cluster id: {cluster_id}
- Cluster surfaces (sorted): {surfaces}
- Per-surface occurrence evidence (model / unit_id / turn_indices / count):
{occurrences}
- Tone-mismatch pairs (TONE3 strings differ; treat as a WEAK signal only):
{tone_mismatch_pairs}
- Context snippets (hyp lines where each surface occurred as a full run; may be truncated):
{snippets}

Now partition the cluster. Output JSON only.
"""


def _render_occurrences(cluster: HomophoneCluster) -> str:
    if not cluster.hits:
        return "(none)"
    # Aggregate counts per (surface, model, unit_id).
    agg: dict[tuple[str, str, str], list[int]] = {}
    for h in cluster.hits:
        key = (str(h.get("surface", "")), str(h.get("model", "")), str(h.get("unit_id", "")))
        agg.setdefault(key, []).extend(h.get("turn_indices") or [])
    lines: list[str] = []
    for (surface, model, unit_id), turns in agg.items():
        lines.append(
            f"- {surface}: model={model}, unit_id={unit_id}, "
            f"turn_indices={turns}, count={len(turns)}"
        )
    return "\n".join(lines) if lines else "(none)"


def _render_tone_mismatch(cluster: HomophoneCluster) -> str:
    if not cluster.tone_mismatch_pairs:
        return "(none)"
    return "\n".join(f"- {a} / {b}" for a, b in cluster.tone_mismatch_pairs)


def _render_snippets(cluster: HomophoneCluster) -> str:
    seen: set[tuple[str, str]] = set()
    lines: list[str] = []
    for h in cluster.hits:
        surface = str(h.get("surface", ""))
        hyp_text = str(h.get("hyp_text", ""))
        key = (surface, hyp_text)
        if key in seen or not hyp_text:
            continue
        seen.add(key)
        lines.append(f"- {surface}: {hyp_text}")
    return "\n".join(lines) if lines else "(none)"


def render_partition_user_prompt(*, cluster: HomophoneCluster) -> str:
    """Render the partition user prompt for one homophone cluster.

    Accepts a `HomophoneCluster` (preferred). Callers passing a dict should
    wrap it into a `HomophoneCluster` first; this function does not accept
    dicts because occurrences/snippets require the cluster's hit records.
    """
    return _PARTITION_USER_TEMPLATE.format(
        cluster_id=cluster.cluster_id,
        surfaces=json.dumps(list(cluster.surfaces), ensure_ascii=False),
        occurrences=_render_occurrences(cluster),
        tone_mismatch_pairs=_render_tone_mismatch(cluster),
        snippets=_render_snippets(cluster),
    )
