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
        {"unit_id": "u0", "turn_indices": [0], "hyps": [{"model": "qwen", "text": "张三丰来了"}]},
        {"unit_id": "u1", "turn_indices": [1], "hyps": [{"model": "moss", "text": "张三丰走了"}]},
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
