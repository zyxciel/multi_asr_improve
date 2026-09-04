from stage2_asr.polish_cluster import (
    build_homophone_clusters,
    extract_full_cjk_runs,
    pair_surfaces,
)


def test_extract_full_runs_no_windows():
    runs = extract_full_cjk_runs("找张三丰签字")
    assert runs == ["找张三丰签字"]
    assert "找张三丰" not in runs
    assert "张三丰" not in runs  # proper substring of a longer run


def test_length_2_never_pairs():
    assert pair_surfaces("意图", "音图") is False


def test_flagship_zhang_pairs():
    assert pair_surfaces("张三风", "涨三丰") is True


def test_zhang_vs_zhang_chapter_pairs():
    assert pair_surfaces("张三丰", "章三丰") is True


def test_identical_hyps_no_cluster():
    recs = [
        {"unit_id": "u0", "turn_indices": [0], "hyps": [{"model": "qwen", "text": "张三丰"}]},
        {"unit_id": "u1", "turn_indices": [1], "hyps": [{"model": "moss", "text": "张三丰"}]},
    ]
    assert build_homophone_clusters(recs) == []


def test_two_writings_across_engines_cluster():
    recs = [
        {
            "unit_id": "u0",
            "turn_indices": [0],
            "hyps": [
                {"model": "qwen", "text": "张三风"},
                {"model": "firered", "text": "涨三丰"},
            ],
        }
    ]
    clusters = build_homophone_clusters(recs)
    assert len(clusters) == 1
    assert set(clusters[0].surfaces) == {"张三风", "涨三丰"}


def test_windows_from_long_run_not_members():
    recs = [
        {"unit_id": "u0", "turn_indices": [0], "hyps": [{"model": "qwen", "text": "找张三丰签字"}]},
        {"unit_id": "u1", "turn_indices": [1], "hyps": [{"model": "moss", "text": "签张三丰"}]},
    ]
    clusters = build_homophone_clusters(recs)
    members = {s for c in clusters for s in c.surfaces}
    assert "找张三丰" not in members
    assert not any(
        "找张三丰签字" in c.surfaces and "签张三丰" in c.surfaces for c in clusters
    )
    # full runs 找张三丰签字 vs 签张三丰 must not pair (first/last syllables differ)


def test_full_run_phrase_residual_may_cluster():
    recs = [
        {"unit_id": "u0", "turn_indices": [0], "hyps": [{"model": "qwen", "text": "找张三丰"}]},
        {"unit_id": "u1", "turn_indices": [1], "hyps": [{"model": "moss", "text": "签张三丰"}]},
    ]
    clusters = build_homophone_clusters(recs)
    assert any(set(c.surfaces) == {"找张三丰", "签张三丰"} for c in clusters)


def test_transitive_closure_keeps_distant_pair():
    a, b, c = "张三丰来了", "张三丰走了", "张四海走了"
    assert pair_surfaces(a, b) is True
    assert pair_surfaces(b, c) is True
    assert pair_surfaces(a, c) is False  # toneless dist 3 > 2
    recs = [
        {"unit_id": "u0", "turn_indices": [0], "hyps": [{"model": "qwen", "text": a}]},
        {"unit_id": "u1", "turn_indices": [1], "hyps": [{"model": "moss", "text": b}]},
        {"unit_id": "u2", "turn_indices": [2], "hyps": [{"model": "firered", "text": c}]},
    ]
    clusters = build_homophone_clusters(recs)
    assert len(clusters) == 1
    assert set(clusters[0].surfaces) == {a, b, c}


def test_pair_surfaces_cache_does_not_change_rules():
    """The optional pinyin cache memoizes conversions; pairing rules are unchanged."""
    cache: dict = {}
    assert pair_surfaces("张三风", "涨三丰", cache=cache) is True
    assert ("张三风", False) in cache
    assert ("涨三丰", False) in cache
    # Cached and uncached calls agree, including the length gate.
    assert pair_surfaces("意图", "音图", cache=cache) is False
    assert pair_surfaces("张三丰", "章三丰", cache=cache) == pair_surfaces("张三丰", "章三丰")


# --- Task 2: parse partition payload -> allow-list ---

from stage2_asr.polish_cluster import (
    HomophoneCluster,
    cluster_allow_conflicts,
    cluster_allow_list,
    parse_partition_payload,
)


def _cluster(*surfaces: str) -> HomophoneCluster:
    return HomophoneCluster(
        cluster_id="c0",
        surfaces=tuple(surfaces),
        hits=[],
        tone_mismatch_pairs=[],
    )


def test_parse_drops_canonical_not_in_subset():
    raw = {"subsets": [{"surfaces": ["张三风", "涨三丰"], "canonical": "张三丰", "same_entity": True, "reason": "x"}]}
    assert parse_partition_payload(raw, _cluster("张三风", "涨三丰")) == []


def test_parse_unlisted_surface_not_in_allow_list():
    raw = {"subsets": [{"surfaces": ["张三风", "涨三丰"], "canonical": "涨三丰", "same_entity": True, "reason": "x"}]}
    subs = parse_partition_payload(raw, _cluster("张三风", "涨三丰", "张三峰"))
    allow = cluster_allow_list(subs)
    assert allow.get("张三风") == "涨三丰"
    assert "张三峰" not in allow


def test_false_multi_surface_empty_allow_list():
    raw = {"subsets": [{"surfaces": ["找张三丰", "签张三丰"], "canonical": None, "same_entity": False, "reason": "verbs"}]}
    subs = parse_partition_payload(raw, _cluster("找张三丰", "签张三丰"))
    assert cluster_allow_list(subs) == {}


def test_cluster_allow_list_drops_within_cluster_conflict():
    raw = {
        "subsets": [
            {
                "surfaces": ["张三风", "涨三丰"],
                "canonical": "涨三丰",
                "same_entity": True,
                "reason": "a",
            },
            {
                "surfaces": ["张三风", "张三峰"],
                "canonical": "张三峰",
                "same_entity": True,
                "reason": "b",
            },
        ]
    }
    subs = parse_partition_payload(raw, _cluster("张三风", "涨三丰", "张三峰"))
    allow = cluster_allow_list(subs)
    assert "张三风" not in allow
    assert cluster_allow_conflicts(subs) == ["张三风"]


# --- Task 3: subset hard check + leftover warning ---

from stage2_asr.polish_cluster import (
    EntitySubset,
    cluster_channel_edit,
    intra_subset_unify_edit,
    leftover_mentions,
    revert_subset_edits,
    subset_edit_texts_unique,
)


def _subset(surfaces, canonical):
    return EntitySubset(
        surfaces=frozenset(surfaces),
        canonical=canonical,
        same_entity=True,
        reason="test",
    )


def test_hard_check_reverts_only_that_subset():
    subset_s = _subset({"张三风", "涨三丰"}, "涨三丰")
    subset_t = _subset({"赵六兆", "赵六照"}, "赵六照")
    allow = cluster_allow_list([subset_s, subset_t])

    # Audits as the pipeline really emits them: start_char/end_char are
    # offsets in the turn's PRE-edit text (apply_polish_edits locates spans
    # in its input and applies edits right-to-left).
    #
    # Turn 0: a length-changing punc edit (2 chars -> 1) BEFORE the subset
    # entity edit 张三风->涨三丰. The landed 涨三丰 therefore sits at
    # post-apply offset 2, not 3; the revert must convert coordinates.
    # Turn 1: an intra-subset unify edit landing the OTHER in-subset writing
    # (涨三丰->张三风) -> hard check fails for S.
    # Turn 2: subset T's edit is a different subset and must stay.
    applied = [
        {"turn_index": 0, "span_asr": "，。", "span_out": "。", "kind": "punc",
         "start_char": 0, "end_char": 2},
        {"turn_index": 0, "span_asr": "张三风", "span_out": "涨三丰",
         "kind": "entity", "start_char": 3, "end_char": 6},
        {"turn_index": 1, "span_asr": "涨三丰", "span_out": "张三风",
         "kind": "entity", "start_char": 0, "end_char": 3},
        {"turn_index": 2, "span_asr": "赵六兆", "span_out": "赵六照",
         "kind": "entity", "start_char": 0, "end_char": 3},
    ]
    texts_after = {
        0: "。请涨三丰发言。",
        1: "张三风点头。",
        2: "赵六照也在场。",
    }

    applied_for_s = [a for a in applied if intra_subset_unify_edit(a, subset_s)]
    # The punc edit is not intra-subset; only the two entity edits count.
    assert applied_for_s == [applied[1], applied[2]]
    assert subset_edit_texts_unique(applied_for_s, "涨三丰") is False
    assert cluster_channel_edit(applied[1], allow) is True
    assert cluster_channel_edit(applied[2], allow) is False  # non-canonical landing

    sink: list[dict] = []
    reverted = revert_subset_edits(texts_after, applied, subset_s, audit_sink=sink)
    assert reverted == {
        0: "。请张三风发言。",  # entity reverted despite the punc length change
        1: "涨三丰点头。",
        2: "赵六照也在场。",  # other subset's landed edit stays
    }
    assert not [r for r in sink if r.get("path") == "subset_revert_skip"]
    # The caller's mapping is not mutated.
    assert texts_after[1] == "张三风点头。"


def test_revert_leaves_foreign_channel_edits_in_place():
    """A repair landing a spelling OUTSIDE the subset is not an intra-subset
    unify attempt: it neither counts for the hard check nor gets reverted."""
    subset_s = _subset({"张三风", "涨三丰"}, "涨三丰")

    # Turn 2: hotword repair 张三风->张三丰 (span_out not in S).
    applied = [
        {"turn_index": 0, "span_asr": "张三风", "span_out": "涨三丰",
         "kind": "entity", "start_char": 0, "end_char": 3},
        {"turn_index": 1, "span_asr": "涨三丰", "span_out": "张三风",
         "kind": "entity", "start_char": 0, "end_char": 3},
        {"turn_index": 2, "span_asr": "张三风", "span_out": "张三丰",
         "kind": "entity", "start_char": 0, "end_char": 3, "anchor": "hotword"},
    ]
    texts_after = {0: "涨三丰发言。", 1: "张三风点头。", 2: "张三丰到场。"}

    foreign = applied[2]
    assert intra_subset_unify_edit(foreign, subset_s) is False
    applied_for_s = [a for a in applied if intra_subset_unify_edit(a, subset_s)]
    assert applied_for_s == [applied[0], applied[1]]
    assert subset_edit_texts_unique(applied_for_s, "涨三丰") is False

    reverted = revert_subset_edits(texts_after, applied, subset_s)
    assert reverted == {
        0: "张三风发言。",
        1: "涨三丰点头。",
        2: "张三丰到场。",  # foreign-channel repair left alone
    }


def test_leftover_unedited_does_not_fail_unique():
    subset_s = _subset({"张三风", "涨三丰"}, "涨三丰")
    # 3 mentions: turns 0 and 1 edited to canonical, turn 2 left unedited.
    texts = {
        0: "涨三丰先到。",
        1: "大家问涨三丰好。",
        2: "张三风最后走。",
    }
    applied_for_s = [
        {"turn_index": 0, "span_asr": "张三风", "span_out": "涨三丰",
         "start_char": 0, "end_char": 3},
        {"turn_index": 1, "span_asr": "张三风", "span_out": "涨三丰",
         "start_char": 3, "end_char": 6},
    ]

    # All landed edits are canonical: hard check passes, nothing to revert.
    assert subset_edit_texts_unique(applied_for_s, "涨三丰") is True

    # The unedited mention is a leftover warning, not a failure.
    rows = leftover_mentions(texts, subset_s, applied_for_s)
    assert rows == [
        {
            "pass": "polish_cluster",
            "path": "leftover_mix",
            "surfaces": ["张三风"],
            "turn_index": 2,
            "canonical": "涨三丰",
        }
    ]


def test_leftover_mask_uses_post_apply_spans_when_prior_edit_changes_length():
    """Canonical 找张三丰 contains 张三丰. A 2-char punc shrink before the
    cluster edit must not leave a leftover_mix on already-unified text.
    """
    subset_s = _subset({"张三丰", "找张三丰"}, "找张三丰")
    applied = [
        {
            "turn_index": 0,
            "span_asr": "嗯嗯",
            "span_out": "",
            "kind": "punc",
            "start_char": 0,
            "end_char": 2,
        },
        {
            "turn_index": 0,
            "span_asr": "张三丰",
            "span_out": "找张三丰",
            "kind": "entity",
            "start_char": 2,
            "end_char": 5,
        },
    ]
    texts = {0: "找张三丰来了"}
    # Subset-only audits omit the punc length delta, so the pre-edit
    # entity offset masks the wrong post-edit slice and 张三丰 inside
    # 找张三丰 looks uncovered. Production must pass every landed polish
    # audit of the turn (see `_cluster_subset_sweep`).
    assert leftover_mentions(texts, subset_s, [applied[1]])
    assert leftover_mentions(texts, subset_s, applied) == []


def test_leftover_after_full_revert_does_not_warn_when_unmixed():
    """Stale applied_for_s after a full subset revert must not emit leftover_mix."""
    subset_s = _subset({"张三风", "涨三丰"}, "涨三丰")
    texts = {0: "张三风先到。", 1: "张三风最后走。"}
    applied_for_s = [
        {
            "turn_index": 0,
            "span_asr": "张三风",
            "span_out": "涨三丰",
            "start_char": 0,
            "end_char": 3,
        },
        {
            "turn_index": 1,
            "span_asr": "涨三丰",
            "span_out": "张三风",
            "start_char": 0,
            "end_char": 3,
        },
    ]
    assert leftover_mentions(texts, subset_s, applied_for_s) == []


# --- Task 4: partition prompt + judge methods ---

from stage2_asr.polish_cluster_prompt import (
    PARTITION_SYSTEM_PROMPT,
    render_partition_user_prompt,
)
from stage2_asr.runners.mock_llm import MockLlmJudge


def _cluster_with_hits(*surfaces: str) -> HomophoneCluster:
    hits = []
    for i, s in enumerate(surfaces):
        hits.append(
            {
                "surface": s,
                "model": "qwen" if i % 2 == 0 else "moss",
                "unit_id": f"u{i}",
                "turn_indices": [i],
                "hyp_text": s,
            }
        )
    return HomophoneCluster(
        cluster_id="c0",
        surfaces=tuple(surfaces),
        hits=hits,
        tone_mismatch_pairs=[(surfaces[0], surfaces[1])] if len(surfaces) >= 2 else [],
    )


def test_partition_prompt_tone_is_weak():
    """The partition prompt blob must treat tone-mismatch as a WEAK signal.

    It must contain `weak` and `do not weight` (English) and must NOT contain
    the forbidden phrase `more willing to split` (which would push the model
    to split on tone alone, suppressing true surname-variant unifications).
    """
    cluster = _cluster_with_hits("张三风", "涨三丰")
    user = render_partition_user_prompt(cluster=cluster)
    blob = (PARTITION_SYSTEM_PROMPT + "\n" + user).lower()

    assert "weak" in blob
    assert "do not weight" in blob
    assert "more willing to split" not in blob


def test_partition_snippets_are_truncated():
    long_hyp = "张三风" + ("啊" * 5000)
    cluster = HomophoneCluster(
        cluster_id="c0",
        surfaces=("张三风", "涨三丰"),
        hits=[
            {
                "surface": "张三风",
                "model": "qwen",
                "unit_id": "u0",
                "turn_indices": [0],
                "hyp_text": long_hyp,
            }
        ],
        tone_mismatch_pairs=[],
    )
    blob = render_partition_user_prompt(cluster=cluster)
    assert len(blob) < len(long_hyp)
    assert "..." in blob or len(blob) < 8000


def test_mock_partition_default_empty_allow_list():
    """MockLlmJudge.partition_cluster default -> empty subsets -> empty allow-list."""
    cluster = _cluster_with_hits("张三风", "涨三丰")
    mock = MockLlmJudge()
    raw = mock.partition_cluster(cluster=cluster, unit_id="u0")
    subsets = parse_partition_payload(raw, cluster)
    assert cluster_allow_list(subsets) == {}
    assert raw == {"subsets": []}


def test_mock_partition_fn_override_used_when_set():
    """When self.partition_fn is set, MockLlmJudge.partition_cluster calls it."""
    cluster = _cluster_with_hits("张三风", "涨三丰")

    def fake_fn(**kwargs):
        return {
            "subsets": [
                {
                    "surfaces": ["张三风", "涨三丰"],
                    "canonical": "涨三丰",
                    "same_entity": True,
                    "reason": "same person",
                }
            ]
        }

    mock = MockLlmJudge()
    mock.partition_fn = fake_fn
    raw = mock.partition_cluster(cluster=cluster, unit_id="u0")
    subsets = parse_partition_payload(raw, cluster)
    allow = cluster_allow_list(subsets)
    assert allow == {"张三风": "涨三丰", "涨三丰": "涨三丰"}


def test_qwen_polish_many_passes_cluster_mappings():
    """Qwen36LlmJudge.polish_many threads cluster_mappings into the user prompt."""
    from stage2_asr.runners.llm_qwen36 import Qwen36LlmJudge

    captured_users: list[str] = []

    def gen(system, user):
        captured_users.append(user)
        return '{"text": "涨三丰", "edits": []}'

    judge = Qwen36LlmJudge(enabled=False, generate_fn=gen)
    mapping = "张三风|涨三丰 → 涨三丰"
    jobs = [
        {
            "unit_id": "u0",
            "text": "张三风来了",
            "neighbor_draft": [],
            "hotwords": [],
            "turn_index": 0,
            "hypotheses": [],
            "meeting_hyps": "(none)",
            "cluster_mappings": mapping,
        }
    ]
    judge.polish_many(jobs, max_workers=1)
    assert captured_users
    assert mapping in captured_users[0]


def test_qwen_partition_cluster_uses_polish_cluster_pass_with_thinking_on():
    """Qwen36LlmJudge.partition_cluster calls _generate with
    pass_name='polish_cluster', enable_thinking=True, max_tokens=2048."""
    from stage2_asr.runners.llm_qwen36 import Qwen36LlmJudge

    logs: list[dict] = []
    captured: dict = {}

    def gen(system, user):
        captured["system"] = system
        captured["user"] = user
        return '{"subsets": []}'

    judge = Qwen36LlmJudge(enabled=True, generate_fn=gen, log_fn=logs.append)
    cluster = _cluster_with_hits("张三风", "涨三丰")
    out = judge.partition_cluster(cluster=cluster, unit_id="pc0")
    assert out == {"subsets": []}

    pc_logs = [e for e in logs if e.get("pass") == "polish_cluster"]
    assert pc_logs, "expected a polish_cluster log row"
    assert pc_logs[0].get("enable_thinking") is True
    assert pc_logs[0].get("max_tokens") == 2048
    assert pc_logs[0].get("unit_id") == "pc0"
    # The prompt blob must still carry the weak-signal instruction.
    blob = (captured["system"] + "\n" + captured["user"]).lower()
    assert "weak" in blob
    assert "do not weight" in blob
