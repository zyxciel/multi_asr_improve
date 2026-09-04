"""Polish homophone cluster construction (deterministic, from ASR hyps).

Stage-2 Multi-ASR + LLM Fusion — Task 1.

This module only builds candidate homophone clusters from full CJK runs in ASR
hypotheses. It does not call any LLM, does not partition, and does not perform
span edits. Later tasks wire partition parsing and the polish allow-list.

Spec: docs/superpowers/specs/2026-09-04-polish-homophone-cluster-design.md §1.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field

from stage2_asr.pinyin_util import pinyin_edit_distance, to_pinyin

_CJK_RE = re.compile(r"[\u4e00-\u9fff]+")

# Minimum CJK length for a run to be eligible as a cluster surface.
_MIN_RUN_LEN = 3

# Toneless pinyin edit distance cap (same as Pass A).
_MAX_PINYIN_DIST = 2

# Extra brake: dist / max(syl_count(A), syl_count(B), 1) <= 2/3.
_DIST_RATIO = 2 / 3


def extract_full_cjk_runs(text: str) -> list[str]:
    """Return maximal Han runs of length >= 3.

    No sliding windows: a run is stored as-is. Proper substrings of a longer
    run are not emitted (they never appear as a separate match here because
    the regex is greedy on contiguous Han chars).
    """
    return [r for r in _CJK_RE.findall(text) if len(r) >= _MIN_RUN_LEN]


def pair_surfaces(a: str, b: str) -> bool:
    """Spec §1 pairing (1)-(4). Both surfaces must independently be eligible."""
    # (1) CJK length >= 3 on both.
    if len(a) < _MIN_RUN_LEN or len(b) < _MIN_RUN_LEN:
        return False

    # (2) Toneless pinyin edit distance <= 2.
    dist = pinyin_edit_distance(a, b)
    if dist > _MAX_PINYIN_DIST:
        return False

    # (3) First or last toneless syllable equal.
    sa = to_pinyin(a, tone=False).split()
    sb = to_pinyin(b, tone=False).split()
    if not sa or not sb:
        return False
    first_eq = sa[0] == sb[0]
    last_eq = sa[-1] == sb[-1]
    if not (first_eq or last_eq):
        return False

    # (4) Extra brake: dist / max(syl_count(A), syl_count(B), 1) <= 2/3.
    max_syl = max(len(sa), len(sb), 1)
    if (dist / max_syl) > _DIST_RATIO:
        return False

    return True


@dataclass
class HomophoneCluster:
    cluster_id: str
    surfaces: tuple[str, ...]
    hits: list[dict]
    tone_mismatch_pairs: list[tuple[str, str]] = field(default_factory=list)


@dataclass(frozen=True)
class EntitySubset:
    """One subset of a homophone cluster after LLM partition.

    - `surfaces`: the writings grouped together by the partitioner.
    - `canonical`: the chosen canonical writing (None when `same_entity` is
      False, or when the partitioner declined to pick one).
    - `same_entity`: whether the subset is one entity (merge) or distinct
      entities (do not merge).
    - `reason`: free-form partitioner rationale (may be empty).
    """

    surfaces: frozenset[str]
    canonical: str | None
    same_entity: bool
    reason: str


def parse_partition_payload(
    raw: dict | None, cluster: HomophoneCluster
) -> list[EntitySubset]:
    """Parse a partition payload into a list of `EntitySubset`.

    Robust to malformed input: invalid JSON/type -> ``[]``.

    Rules:
    - `same_entity: true` without a `canonical` that is in that subset's
      surfaces -> drop that subset.
    - Surfaces not listed in any subset are not returned (implicit
      no-permission singletons).
    - `same_entity` accepts JSON true/false. Empty `reason` is ok. Unknown
      keys are ignored.
    """
    if not isinstance(raw, dict):
        return []
    subsets_raw = raw.get("subsets")
    if not isinstance(subsets_raw, list):
        return []

    cluster_surfaces = set(cluster.surfaces)
    out: list[EntitySubset] = []
    for entry in subsets_raw:
        if not isinstance(entry, dict):
            continue
        surfaces_raw = entry.get("surfaces")
        if not isinstance(surfaces_raw, list):
            continue
        # Keep only string surfaces that are members of the cluster.
        surfaces = frozenset(
            s for s in surfaces_raw if isinstance(s, str) and s in cluster_surfaces
        )
        if not surfaces:
            continue

        canonical = entry.get("canonical")
        if canonical is not None and not isinstance(canonical, str):
            continue
        if canonical is not None and canonical not in cluster_surfaces:
            canonical = None

        same_entity = entry.get("same_entity", False)
        if not isinstance(same_entity, bool):
            continue

        reason = entry.get("reason", "")
        if not isinstance(reason, str):
            reason = ""

        # same_entity=True requires a canonical that is in this subset's surfaces.
        if same_entity and (canonical is None or canonical not in surfaces):
            continue

        out.append(
            EntitySubset(
                surfaces=surfaces,
                canonical=canonical,
                same_entity=same_entity,
                reason=reason,
            )
        )
    return out


def cluster_allow_list(subsets: list[EntitySubset]) -> dict[str, str]:
    """Build the polish allow-list from entity subsets.

    For each `same_entity` subset, map every surface in the subset (including
    the canonical itself, ``canonical -> canonical``) to the canonical.
    `same_entity: false` surfaces are not included.
    """
    allow: dict[str, str] = {}
    for sub in subsets:
        if not sub.same_entity or sub.canonical is None:
            continue
        for s in sub.surfaces:
            allow[s] = sub.canonical
    return allow


def _tone_mismatch(a: str, b: str) -> bool:
    return to_pinyin(a, tone=True) != to_pinyin(b, tone=True)


# --- Task 3: entity-subset hard check + leftover warning (spec §4) ---
#
# Applied-edit audits carry: turn_index, span_asr, span_out, start_char,
# end_char (coordinates are AFTER the edit was applied to the turn text).


def cluster_channel_edit(edit: dict, allow: dict[str, str]) -> bool:
    """True iff the edit used the cluster allow-list exactly as granted.

    ``span_asr`` must be an allow-listed surface and ``span_out`` must be the
    canonical that surface maps to. An edit that touched a subset surface but
    landed a different writing is NOT a cluster-channel edit by this
    predicate (it is what the hard check exists to catch).
    """
    span_asr = str(edit.get("span_asr"))
    if span_asr not in allow:
        return False
    return str(edit.get("span_out")) == allow[span_asr]


def subset_edit_texts_unique(applied_for_s: list[dict], canonical: str) -> bool:
    """Hard check: every landed cluster-channel edit for subset S is canonical."""
    return all(str(a.get("span_out")) == canonical for a in applied_for_s)


def revert_subset_edits(
    texts: dict[int, str], applied: list[dict], subset: EntitySubset
) -> dict[int, str]:
    """Undo the cluster-channel edits of one subset, leave everything else.

    For each audit whose ``span_asr`` is in ``subset.surfaces``, put
    ``span_asr`` back at ``start_char``, replacing the ``len(span_out)``
    characters that the edit landed. Audits are reverted per turn in
    DESCENDING ``start_char`` order so that reverting a later span (whose
    length may differ from the original) never shifts the coordinates of an
    earlier span that is still pending revert. Audits from other subsets and
    non-cluster edits are untouched. If the current text at the recorded
    coordinates no longer equals ``span_out``, that audit is skipped (a blind
    splice would corrupt drifted text).

    Returns a new dict; ``texts`` is not mutated.
    """
    out = dict(texts)
    per_turn: dict[int, list[dict]] = {}
    for a in applied:
        if str(a.get("span_asr")) not in subset.surfaces:
            continue
        turn = a.get("turn_index")
        start = a.get("start_char")
        if not isinstance(turn, int) or not isinstance(start, int):
            continue
        if turn not in out:
            continue
        per_turn.setdefault(turn, []).append(a)

    for turn, audits in per_turn.items():
        text = out[turn]
        for a in sorted(audits, key=lambda x: x["start_char"], reverse=True):
            start = a["start_char"]
            span_asr = str(a.get("span_asr"))
            span_out = str(a.get("span_out"))
            end = start + len(span_out)
            if text[start:end] != span_out:
                continue
            text = text[:start] + span_asr + text[end:]
        out[turn] = text
    return out


def leftover_mentions(
    texts: dict[int, str], subset: EntitySubset, applied_for_s: list[dict]
) -> list[dict]:
    """Warning rows for subset surfaces still visible in the final texts.

    A turn is flagged when a non-canonical surface of ``subset`` still occurs
    as a substring OUTSIDE spans that were replaced by cluster-channel edits
    to canonical (those spans are fully replaced by definition). Unedited
    mentions are mention autonomy, not a hard failure: they are reported, not
    reverted. Occurrences of the canonical itself never flag a turn.
    """
    canonical = subset.canonical
    # Spans per turn that were fully replaced by edits landing canonical.
    replaced: dict[int, list[tuple[int, int]]] = {}
    for a in applied_for_s:
        span_out = str(a.get("span_out"))
        if canonical is None or span_out != canonical:
            continue
        turn = a.get("turn_index")
        start = a.get("start_char")
        if not isinstance(turn, int) or not isinstance(start, int):
            continue
        replaced.setdefault(turn, []).append((start, start + len(span_out)))

    candidates = sorted(s for s in subset.surfaces if s != canonical)
    rows: list[dict] = []
    for i in sorted(texts):
        text = texts[i]
        spans = replaced.get(i, [])
        found: list[str] = []
        for s in candidates:
            idx = text.find(s)
            while idx != -1:
                covered = any(a <= idx and idx + len(s) <= b for a, b in spans)
                if not covered:
                    found.append(s)
                    break
                idx = text.find(s, idx + 1)
        if found:
            rows.append(
                {
                    "pass": "polish_cluster",
                    "path": "leftover_mix",
                    "surfaces": found,
                    "turn_index": i,
                    "canonical": canonical,
                }
            )
    return rows


class _UnionFind:
    def __init__(self) -> None:
        self.parent: dict[str, str] = {}

    def find(self, x: str) -> str:
        self.parent.setdefault(x, x)
        while self.parent[x] != x:
            self.parent[x] = self.parent[self.parent[x]]
            x = self.parent[x]
        return x

    def union(self, a: str, b: str) -> None:
        ra, rb = self.find(a), self.find(b)
        if ra != rb:
            self.parent[ra] = rb


def build_homophone_clusters(hyp_records: list[dict]) -> list[HomophoneCluster]:
    """Build homophone clusters from ASR hypothesis records.

    Each record: {"unit_id": str, "turn_indices": list[int],
                  "hyps": [{"model": str, "text": str}, ...]}.

    Returns clusters that pass the disagreement gate: >= 2 distinct surfaces
    that come from different `model` or different `unit_id`.
    """
    # Step 1: extract eligible surfaces and their hits.
    # A "hit" is one (surface, model, unit_id, turn_indices, hyp_text) tuple.
    hits: list[dict] = []
    surfaces_set: set[str] = set()
    for rec in hyp_records:
        unit_id = rec.get("unit_id", "")
        turn_indices = list(rec.get("turn_indices", []))
        for hyp in rec.get("hyps", []):
            model = hyp.get("model", "")
            hyp_text = hyp.get("text", "")
            for run in extract_full_cjk_runs(hyp_text):
                hit = {
                    "surface": run,
                    "model": model,
                    "unit_id": unit_id,
                    "turn_indices": list(turn_indices),
                    "hyp_text": hyp_text,
                }
                hits.append(hit)
                surfaces_set.add(run)

    if len(surfaces_set) < 2:
        return []

    # Step 2: union-find over eligible surfaces via pair_surfaces.
    uf = _UnionFind()
    for s in surfaces_set:
        uf.find(s)  # ensure each surface is its own root initially
    surfaces_list = sorted(surfaces_set)
    for i in range(len(surfaces_list)):
        for j in range(i + 1, len(surfaces_list)):
            a, b = surfaces_list[i], surfaces_list[j]
            if pair_surfaces(a, b):
                uf.union(a, b)

    # Step 3: group surfaces by root, attach hits.
    groups: dict[str, list[str]] = {}
    for s in surfaces_list:
        root = uf.find(s)
        groups.setdefault(root, []).append(s)

    # Step 4: disagreement gate — keep cluster only if >= 2 distinct surfaces
    # AND at least two writings from different model OR different unit_id.
    clusters: list[HomophoneCluster] = []
    for idx, (root, members) in enumerate(sorted(groups.items(), key=lambda kv: kv[0])):
        member_set = set(members)
        if len(member_set) < 2:
            continue
        member_hits = [h for h in hits if h["surface"] in member_set]
        if not member_hits:
            continue
        # Disagreement: need at least two hits that differ in model or unit_id.
        # Equivalent: not all hits share the same (model, unit_id).
        keys = {(h["model"], h["unit_id"]) for h in member_hits}
        if len(keys) < 2:
            continue

        # Tone mismatch pairs: all distinct surface pairs where TONE3 differs.
        tone_mismatch_pairs: list[tuple[str, str]] = []
        members_sorted = sorted(member_set)
        for i in range(len(members_sorted)):
            for j in range(i + 1, len(members_sorted)):
                a, b = members_sorted[i], members_sorted[j]
                if _tone_mismatch(a, b):
                    tone_mismatch_pairs.append((a, b))

        cluster_id = f"polish_cluster_{idx:04d}"
        clusters.append(
            HomophoneCluster(
                cluster_id=cluster_id,
                surfaces=tuple(members_sorted),
                hits=member_hits,
                tone_mismatch_pairs=tone_mismatch_pairs,
            )
        )

    return clusters
