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
    # Construct A-B and B-C legal, A-C dist > 2 if possible; if hard with real names,
    # use three surfaces where union-find still yields one cluster of size 3.
    recs = [
        {"unit_id": "u0", "turn_indices": [0], "hyps": [{"model": "qwen", "text": "张三风"}]},
        {"unit_id": "u1", "turn_indices": [1], "hyps": [{"model": "moss", "text": "涨三丰"}]},
        {"unit_id": "u2", "turn_indices": [2], "hyps": [{"model": "firered", "text": "章三丰"}]},
    ]
    clusters = build_homophone_clusters(recs)
    assert len(clusters) == 1
    assert set(clusters[0].surfaces) == {"张三风", "涨三丰", "章三丰"}


# --- Task 2: parse partition payload -> allow-list ---

from stage2_asr.polish_cluster import (
    HomophoneCluster,
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


# --- Task 3: subset hard check + leftover warning ---

from stage2_asr.polish_cluster import (
    EntitySubset,
    cluster_channel_edit,
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
    subset_s = _subset({"欧阳娜", "欧阳娜娜"}, "欧阳娜娜")
    subset_t = _subset({"赵六兆", "赵六照"}, "赵六照")
    allow = cluster_allow_list([subset_s, subset_t])

    # S edits: turn 0 landed canonical; turn 1 had two mentions, one landed
    # canonical (length-changing) and one landed a wrong writing -> hard-check
    # failure for S. T edit on turn 2 is a different subset and must stay.
    applied = [
        {"turn_index": 0, "span_asr": "欧阳娜", "span_out": "欧阳娜娜",
         "start_char": 1, "end_char": 5},
        {"turn_index": 1, "span_asr": "欧阳娜", "span_out": "欧阳娜娜",
         "start_char": 0, "end_char": 4},
        {"turn_index": 1, "span_asr": "欧阳娜", "span_out": "欧阳哪",
         "start_char": 5, "end_char": 8},
        {"turn_index": 2, "span_asr": "赵六兆", "span_out": "赵六照",
         "start_char": 0, "end_char": 3},
    ]
    texts_after = {
        0: "请欧阳娜娜发言。",
        1: "欧阳娜娜和欧阳哪都来了。",
        2: "赵六照也在场。",
    }

    applied_for_s = [a for a in applied if a["span_asr"] in subset_s.surfaces]
    assert subset_edit_texts_unique(applied_for_s, "欧阳娜娜") is False
    assert cluster_channel_edit(applied[0], allow) is True
    assert cluster_channel_edit(applied[2], allow) is False  # wrong landed writing

    reverted = revert_subset_edits(texts_after, applied, subset_s)
    assert reverted == {
        0: "请欧阳娜发言。",
        1: "欧阳娜和欧阳娜都来了。",
        2: "赵六照也在场。",  # other subset's landed edit stays
    }
    # The caller's mapping is not mutated.
    assert texts_after[1] == "欧阳娜娜和欧阳哪都来了。"


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
