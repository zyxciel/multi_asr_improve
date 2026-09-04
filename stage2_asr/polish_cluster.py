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


def _tone_mismatch(a: str, b: str) -> bool:
    return to_pinyin(a, tone=True) != to_pinyin(b, tone=True)


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

        # Name-variant gate: a real name variant differs at the first or last
        # Han character (wrong surname char or wrong final char — see spec §1
        # reasoning for pairing rule (3)). If every surface shares the same
        # first Han char AND the same last Han char, the differences are
        # internal/context (e.g. `张三丰来了` vs `张三丰走了` share name `张三丰`
        # and differ only in trailing verbs) — not a name-writing disagreement,
        # so the cluster is dropped. This is a cluster-level gate, not a
        # pairwise re-filter (spec: do not re-check pairwise distance inside
        # the cluster after union-find).
        first_chars = {s[0] for s in member_set}
        last_chars = {s[-1] for s in member_set}
        if len(first_chars) == 1 and len(last_chars) == 1:
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
