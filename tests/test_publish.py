from __future__ import annotations

import json

from pathlib import Path

from stage2_asr.publish import (
    apply_publish_edits,
    concat_meeting,
    filter_publish_edits,
    latin_runs,
    load_glossary,
    run_publish,
    split_meeting,
    validate_publish_payload,
)


def test_concat_split_roundtrip_keeps_turn_texts():
    texts = {0: "嗯你好", 1: "um hello"}
    meeting = concat_meeting(texts)
    assert "⟦t0⟧" in meeting and "⟦t1⟧" in meeting
    assert split_meeting(meeting) == texts


def test_filler_edits_drop_bilingual_fillers():
    meeting = concat_meeting({0: "嗯我们开会", 1: "um we meet"})
    kept, err = filter_publish_edits(
        [
            {"span_asr": "嗯", "span_out": "", "kind": "filler"},
            {"span_asr": "um", "span_out": "", "kind": "filler"},
        ],
        meeting=meeting,
        glossary_terms=[],
    )
    assert err is None
    out, _ = apply_publish_edits(meeting, kept)
    split = split_meeting(out)
    assert split[0] == "我们开会"
    assert split[1] == " we meet" or split[1].strip() == "we meet"


def test_unknown_filler_dropped():
    meeting = concat_meeting({0: "会议开始"})
    kept, _ = filter_publish_edits(
        [{"span_asr": "会议", "span_out": "", "kind": "filler"}],
        meeting=meeting,
        glossary_terms=[],
    )
    assert kept == []


def test_latin_filler_does_not_match_inside_words():
    meeting = concat_meeting({0: "the number later yeah"})
    kept, _ = filter_publish_edits(
        [
            {"span_asr": "um", "span_out": "", "kind": "filler"},
            {"span_asr": "er", "span_out": "", "kind": "filler"},
            {"span_asr": "ah", "span_out": "", "kind": "filler"},
        ],
        meeting=meeting,
        glossary_terms=[],
    )
    assert kept == []
    out, _ = apply_publish_edits(meeting, kept)
    assert split_meeting(out)[0] == "the number later yeah"


def test_latin_filler_still_drops_standalone_um():
    meeting = concat_meeting({0: "um the number"})
    kept, _ = filter_publish_edits(
        [{"span_asr": "um", "span_out": "", "kind": "filler"}],
        meeting=meeting,
        glossary_terms=[],
    )
    out, _ = apply_publish_edits(meeting, kept)
    assert split_meeting(out)[0].strip() == "the number"


def test_cjk_filler_does_not_match_inside_demonstrative_phrase():
    meeting = concat_meeting({0: "以前那个温度的问题"})
    kept, _ = filter_publish_edits(
        [{"span_asr": "那个", "span_out": "", "kind": "filler"}],
        meeting=meeting,
        glossary_terms=[],
    )
    assert kept == []
    out, _ = apply_publish_edits(meeting, kept)
    assert split_meeting(out)[0] == "以前那个温度的问题"


def test_cjk_filler_still_drops_punctuated_nage():
    meeting = concat_meeting({0: "那个，我们开会"})
    kept, _ = filter_publish_edits(
        [{"span_asr": "那个", "span_out": "", "kind": "filler"}],
        meeting=meeting,
        glossary_terms=[],
    )
    out, _ = apply_publish_edits(meeting, kept)
    assert split_meeting(out)[0] == "，我们开会"


def test_repair_cn_and_en():
    meeting = concat_meeting({0: "周二不周三", 1: "Tuesday no Wednesday"})
    kept, _ = filter_publish_edits(
        [
            {"span_asr": "周二不周三", "span_out": "周三", "kind": "repair"},
            {"span_asr": "Tuesday no Wednesday", "span_out": "Wednesday", "kind": "repair"},
        ],
        meeting=meeting,
        glossary_terms=[],
    )
    out, _ = apply_publish_edits(meeting, kept)
    split = split_meeting(out)
    assert split[0] == "周三"
    assert split[1] == "Wednesday"


def test_repair_non_substring_dropped():
    meeting = concat_meeting({0: "周二开会"})
    kept, _ = filter_publish_edits(
        [{"span_asr": "周二开会", "span_out": "周三", "kind": "repair"}],
        meeting=meeting,
        glossary_terms=[],
    )
    assert kept == []


def test_latex_squared_allowed():
    meeting = concat_meeting({0: "算x平方"})
    kept, _ = filter_publish_edits(
        [{"span_asr": "x平方", "span_out": "$x^{2}$", "kind": "latex"}],
        meeting=meeting,
        glossary_terms=[],
    )
    out, _ = apply_publish_edits(meeting, kept)
    assert "$x^{2}$" in out


def test_latex_windows_product_rejected():
    meeting = concat_meeting({0: "用Windows"})
    kept, _ = filter_publish_edits(
        [{"span_asr": "Windows", "span_out": "$Windows$", "kind": "latex"}],
        meeting=meeting,
        glossary_terms=[],
    )
    assert kept == []


def test_translate_gpu_rejected():
    meeting = concat_meeting({0: "用GPU训练"})
    kept, _ = filter_publish_edits(
        [{"span_asr": "GPU", "span_out": "显卡", "kind": "repair"}],
        meeting=meeting,
        glossary_terms=[],
    )
    assert kept == []
    assert "gpu" in latin_runs(meeting)


def test_marker_edit_rejects_whole_payload():
    meeting = concat_meeting({0: "你好"})
    ok, err = validate_publish_payload(
        {"edits": [{"span_asr": "⟦t0⟧", "span_out": "", "kind": "filler"}]},
        meeting=meeting,
    )
    assert ok is False
    assert err


def test_itn_serial_allowed_in_filter():
    meeting = concat_meeting({0: "编号伍柒叁"})
    kept, _ = filter_publish_edits(
        [{"span_asr": "伍柒叁", "span_out": "573", "kind": "itn"}],
        meeting=meeting,
        glossary_terms=[],
    )
    out, _ = apply_publish_edits(meeting, kept)
    assert "573" in out
    kept_bad, _ = filter_publish_edits(
        [{"span_asr": "伍柒叁", "span_out": "五百三十七", "kind": "itn"}],
        meeting=concat_meeting({0: "编号伍柒叁"}),
        glossary_terms=[],
    )
    assert kept_bad == []


def test_run_publish_drops_fillers_and_itn():
    from stage2_asr.runners.mock_llm import MockLlmJudge
    from stage2_asr.types import Turn

    turns = [
        Turn(start=0, end=1, speaker_id="s0", text="嗯编号伍柒叁"),
        Turn(start=1, end=2, speaker_id="s0", text="um GPU"),
    ]
    texts = {0: turns[0].text, 1: turns[1].text}
    out, audits, glossary, ev = run_publish(
        turns, texts, llm_judge=MockLlmJudge()
    )
    assert "嗯" not in out[0]
    assert "573" in out[0]
    assert "um" not in out[1]
    assert "GPU" in out[1]
    assert any(a.get("pass") == "publish" for a in audits)
    assert any(k.get("surface") == "GPU" for k in glossary.get("keywords") or [])
    assert ev is not None and ev.get("faithful") is True


def test_run_publish_eval_reverts_unfaithful():
    from stage2_asr.types import Turn

    class BadEval:
        def publish(self, **kwargs):
            return {"edits": [{"span_asr": "嗯", "span_out": "", "kind": "filler"}]}

        def eval_publish(self, **kwargs):
            return {"faithful": False, "clearer": True, "more_concise": True, "easier": True}

        def extract_terms(self, **kwargs):
            return {"keywords": [], "rare_words": [], "new_terms": []}

    turns = [Turn(start=0, end=1, speaker_id="s0", text="嗯你好")]
    texts = {0: "嗯你好"}
    out, audits, _, ev = run_publish(turns, texts, llm_judge=BadEval())
    assert out[0] == "嗯你好"
    assert ev and ev.get("faithful") is False
    assert any(a.get("path") == "reverted" for a in audits)


def test_run_publish_marker_payload_keeps_original():
    from stage2_asr.types import Turn

    class TouchMarker:
        def publish(self, **kwargs):
            return {"edits": [{"span_asr": "⟦t0⟧", "span_out": "", "kind": "filler"}]}

    turns = [Turn(start=0, end=1, speaker_id="s0", text="你好")]
    out, audits, _, _ = run_publish(turns, {0: "你好"}, llm_judge=TouchMarker())
    assert out[0] == "你好"
    assert any(a.get("fallback") for a in audits)


def test_merge_glossary_seed_wins():
    from stage2_asr.publish import merge_glossary

    seed = {"terms": [{"surface": "GPU", "kind": "product", "aliases": ["显卡"]}]}
    extract = {
        "keywords": [{"surface": "GPU", "score": 0.2}],
        "rare_words": [],
        "new_terms": [{"surface": "GPU", "kind": "other", "aliases": []}],
    }
    merged = merge_glossary(seed, extract)
    gpu = next(t for t in merged["terms"] if t["surface"] == "GPU")
    assert gpu.get("source") == "seed"
    assert gpu.get("kind") == "product"


def test_load_glossary_invalid_json_returns_empty(tmp_path: Path):
    bad = tmp_path / "bad.json"
    bad.write_text("{not json", encoding="utf-8")
    loaded = load_glossary(bad)
    assert loaded == {"terms": [], "keywords": [], "rare_words": []}
    assert load_glossary(None)["terms"] == []


def test_pipeline_publish_stage_writes_artifacts_without_clobber(tmp_path: Path):
    from pathlib import Path as P

    from stage2_asr.pipeline import run_pipeline
    from stage2_asr.runners.mock_asr import MockAsrRunner
    from stage2_asr.runners.mock_llm import MockLlmJudge
    from stage2_asr.types import PipelineConfig

    fixtures = P(__file__).parent / "fixtures"
    out = tmp_path / "work"
    run_pipeline(
        input_json=fixtures / "mode_c.json",
        audio_path=tmp_path / "missing.wav",
        work_dir=out,
        asr_runner=MockAsrRunner(),
        llm_judge=MockLlmJudge(),
        config=PipelineConfig(),
        stage="all",
    )
    assert (out / "mode_c_published.json").exists()
    polished = (out / "mode_c_polished.json").read_text(encoding="utf-8")
    final = (out / "mode_c_asr_final.json").read_text(encoding="utf-8")
    result = run_pipeline(
        input_json=fixtures / "mode_c.json",
        audio_path=tmp_path / "missing.wav",
        work_dir=out,
        asr_runner=MockAsrRunner(),
        llm_judge=MockLlmJudge(),
        config=PipelineConfig(),
        stage="publish",
    )
    assert result.get("published_path") is not None
    assert (out / "transcript.md").exists()
    assert (out / "glossary.json").exists()
    assert (out / "mode_c_polished.json").read_text(encoding="utf-8") == polished
    assert (out / "mode_c_asr_final.json").read_text(encoding="utf-8") == final


def test_pipeline_all_includes_publish(tmp_path: Path):
    from pathlib import Path as P

    from stage2_asr.pipeline import run_pipeline
    from stage2_asr.runners.mock_asr import MockAsrRunner
    from stage2_asr.runners.mock_llm import MockLlmJudge
    from stage2_asr.types import PipelineConfig

    fixtures = P(__file__).parent / "fixtures"
    out = tmp_path / "work"
    result = run_pipeline(
        input_json=fixtures / "mode_c.json",
        audio_path=tmp_path / "missing.wav",
        work_dir=out,
        asr_runner=MockAsrRunner(),
        llm_judge=MockLlmJudge(),
        config=PipelineConfig(),
        stage="all",
    )
    polished = json.loads((out / "mode_c_polished.json").read_text(encoding="utf-8"))
    published = json.loads((out / "mode_c_published.json").read_text(encoding="utf-8"))
    assert len(published["turns"]) == len(polished["turns"])
    if polished["turns"] and "turn_indices" in polished["turns"][0]:
        assert published["turns"][0]["turn_indices"] == polished["turns"][0]["turn_indices"]
    assert result.get("published_path") is not None
    stats = json.loads((out / "pass_stats.json").read_text(encoding="utf-8"))
    assert "polish" in stats
    assert "publish" in stats
    llm_out = tmp_path / "llm_only"
    run_pipeline(
        input_json=fixtures / "mode_c.json",
        audio_path=tmp_path / "missing.wav",
        work_dir=llm_out,
        asr_runner=MockAsrRunner(),
        llm_judge=MockLlmJudge(),
        config=PipelineConfig(),
        stage="asr",
    )
    run_pipeline(
        input_json=fixtures / "mode_c.json",
        audio_path=tmp_path / "missing.wav",
        work_dir=llm_out,
        asr_runner=MockAsrRunner(),
        llm_judge=MockLlmJudge(),
        config=PipelineConfig(),
        stage="llm",
    )
    assert (llm_out / "mode_c_asr_final.json").exists()
    assert not (llm_out / "mode_c_published.json").exists()
    assert not (llm_out / "mode_c_polished.json").exists()


def test_pipeline_publish_requires_input(tmp_path: Path):
    from pathlib import Path as P

    import pytest

    from stage2_asr.pipeline import run_pipeline
    from stage2_asr.runners.mock_asr import MockAsrRunner
    from stage2_asr.runners.mock_llm import MockLlmJudge
    from stage2_asr.types import PipelineConfig

    fixtures = P(__file__).parent / "fixtures"
    with pytest.raises(FileNotFoundError):
        run_pipeline(
            input_json=fixtures / "mode_c.json",
            audio_path=tmp_path / "missing.wav",
            work_dir=tmp_path / "empty",
            asr_runner=MockAsrRunner(),
            llm_judge=MockLlmJudge(),
            config=PipelineConfig(),
            stage="publish",
        )
