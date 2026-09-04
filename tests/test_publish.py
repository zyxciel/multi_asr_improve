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
    speakers = {0: "s0", 1: "s1"}
    meeting = concat_meeting(texts, speakers)
    assert "⟦t0|s0⟧" in meeting and "⟦t1|s1⟧" in meeting
    assert split_meeting(meeting) == texts


def test_concat_speaker_markers_keep_dialogue_boundaries():
    texts = {0: "我们明天", 1: "不行改后天"}
    meeting = concat_meeting(texts, {0: "alice", 1: "bob"})
    assert meeting.index("⟦t0|alice⟧") < meeting.index("我们明天")
    assert meeting.index("我们明天") < meeting.index("⟦t1|bob⟧")
    assert "我们明天不行改后天" not in meeting


def test_cross_speaker_repair_is_not_contiguous():
    meeting = concat_meeting({0: "周二", 1: "不周三"}, {0: "s0", 1: "s1"})
    kept, _ = filter_publish_edits(
        [{"span_asr": "周二不周三", "span_out": "周三", "kind": "repair"}],
        meeting=meeting,
        glossary_terms=[],
    )
    assert kept == []
    split = split_meeting(meeting)
    assert split[0] == "周二"
    assert split[1] == "不周三"


def test_filler_does_not_wipe_whole_turn_backchannel():
    meeting = concat_meeting(
        {0: "我们开会", 1: "嗯", 2: "好的"},
        {0: "s0", 1: "s1", 2: "s0"},
    )
    kept, _ = filter_publish_edits(
        [{"span_asr": "嗯", "span_out": "", "kind": "filler"}],
        meeting=meeting,
        glossary_terms=[],
    )
    out, _ = apply_publish_edits(meeting, kept)
    split = split_meeting(out)
    assert split[1] == "嗯"
    assert split[0] == "我们开会"
    assert split[2] == "好的"


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


def test_span_out_marker_is_rejected():
    meeting = concat_meeting({0: "算x平方"}, {0: "s0"})
    payload = {
        "edits": [{"span_asr": "x平方", "span_out": "$⟦t99|evil⟧$", "kind": "latex"}]
    }
    ok, err = validate_publish_payload(payload, meeting=meeting)
    assert ok is False
    assert err
    kept, _ = filter_publish_edits(payload["edits"], meeting=meeting, glossary_terms=[])
    assert kept == []
    out, _ = apply_publish_edits(meeting, kept)
    assert split_meeting(out) == {0: "算x平方"}


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


def test_apply_publish_edits_keeps_glossary_when_some_edits_lack_offsets():
    meeting = concat_meeting({0: "阿尔法那个啥"})
    start = meeting.find("阿尔法")
    edits = [
        {
            "span_asr": "阿尔法",
            "span_out": r"$\alpha$",
            "kind": "latex",
            "start_char": start,
            "end_char": start + len("阿尔法"),
        },
        {"span_asr": "那个啥", "span_out": "", "kind": "filler"},
    ]
    terms = [
        {"surface": "阿尔法", "kind": "symbol", "latex": r"\alpha"},
        {"surface": "那个啥", "kind": "filler"},
    ]
    out, located = apply_publish_edits(meeting, edits, glossary_terms=terms)
    assert r"$\alpha$" in out
    assert "那个啥" not in split_meeting(out)[0]
    assert {e["kind"] for e in located} == {"latex", "filler"}


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


def test_run_publish_keeps_other_speaker_backchannel():
    from stage2_asr.runners.mock_llm import MockLlmJudge
    from stage2_asr.types import Turn

    turns = [
        Turn(start=0, end=1, speaker_id="s0", text="我们用GPU"),
        Turn(start=1, end=2, speaker_id="s1", text="嗯"),
    ]
    out, _, _, ev = run_publish(
        turns, {0: turns[0].text, 1: turns[1].text}, llm_judge=MockLlmJudge()
    )
    assert "GPU" in out[0]
    assert out[1] == "嗯"
    assert ev is None or ev.get("faithful") is True


def test_qwen_publish_calls_use_raised_max_tokens():
    from stage2_asr.runners.llm_qwen36 import Qwen36LlmJudge

    logs: list[dict] = []

    def gen(system, user):
        _ = (system, user)
        if "faithful" in system.lower() or "quality" in system.lower():
            return '{"faithful": true, "clearer": true, "more_concise": true, "easier": true}'
        if "keyword" in system.lower() or "extract" in system.lower() or "glossary" in system.lower():
            return '{"keywords": [], "rare_words": [], "new_terms": []}'
        return '{"edits": []}'

    judge = Qwen36LlmJudge(enabled=True, generate_fn=gen, log_fn=logs.append)
    judge.publish(meeting="你好", unit_id="p")
    judge.extract_terms(meeting="你好", unit_id="e")
    judge.eval_publish(original="你好", published="你好。", unit_id="v")
    by_pass = {e["pass"]: e.get("max_tokens") for e in logs if e.get("max_tokens") is not None}
    assert by_pass.get("publish") == 4096
    assert by_pass.get("extract") == 2048
    assert by_pass.get("publish_eval") == 8192


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


def test_merge_glossary_unions_and_extract_covers_same_surface():
    from stage2_asr.publish import merge_glossary

    seed = {
        "terms": [
            {"surface": "GPU", "kind": "product", "aliases": ["显卡"], "source": "seed"},
            {"surface": "OldTerm", "kind": "other", "aliases": []},
        ],
        "keywords": [{"surface": "OldKeyword", "score": 0.9}],
        "rare_words": [{"surface": "OldRare"}],
    }
    extract = {
        "keywords": [{"surface": "GPU", "score": 0.2}, {"surface": "NewKeyword", "score": 1.0}],
        "rare_words": [{"surface": "NewRare"}],
        "new_terms": [
            {"surface": "GPU", "kind": "other", "aliases": ["graphics"]},
            {"surface": "Windows产品", "kind": "product", "aliases": []},
        ],
    }
    merged = merge_glossary(seed, extract)
    by_surface = {t["surface"]: t for t in merged["terms"]}
    assert set(by_surface) == {"GPU", "OldTerm", "Windows产品"}
    assert by_surface["GPU"].get("kind") == "other"
    assert by_surface["GPU"].get("aliases") == ["graphics"]
    assert by_surface["GPU"].get("source") == "extract"
    assert by_surface["OldTerm"].get("kind") == "other"
    kw = {k["surface"]: k for k in merged["keywords"]}
    assert set(kw) == {"OldKeyword", "GPU", "NewKeyword"}
    assert kw["GPU"]["score"] == 0.2
    rare = {r["surface"] for r in merged["rare_words"]}
    assert rare == {"OldRare", "NewRare"}


def test_merge_glossary_none_extract_keeps_seed_lists():
    from stage2_asr.publish import merge_glossary

    seed = {
        "terms": [{"surface": "KeepMe", "kind": "product"}],
        "keywords": [{"surface": "K"}],
        "rare_words": [{"surface": "R"}],
    }
    merged = merge_glossary(seed, None)
    assert [t["surface"] for t in merged["terms"]] == ["KeepMe"]
    assert [k["surface"] for k in merged["keywords"]] == ["K"]
    assert [r["surface"] for r in merged["rare_words"]] == ["R"]


def test_union_glossary_cli_covers_same_surface():
    from stage2_asr.publish import union_glossary

    prior = {
        "terms": [
            {"surface": "Old", "kind": "other"},
            {"surface": "Shared", "kind": "other", "aliases": ["a"]},
        ],
        "keywords": [{"surface": "K1"}],
        "rare_words": [],
    }
    cli = {
        "terms": [{"surface": "Shared", "kind": "product", "aliases": ["b"]}],
        "keywords": [{"surface": "K2"}],
        "rare_words": [],
    }
    merged = union_glossary(prior, cli)
    by = {t["surface"]: t for t in merged["terms"]}
    assert set(by) == {"Old", "Shared"}
    assert by["Shared"]["kind"] == "product"
    assert by["Shared"]["aliases"] == ["b"]
    kw = {k["surface"] for k in merged["keywords"]}
    assert kw == {"K1", "K2"}


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


def test_publish_stage_reuses_work_dir_glossary(tmp_path: Path):
    from stage2_asr.pipeline import run_pipeline
    from stage2_asr.runners.mock_asr import MockAsrRunner
    from stage2_asr.runners.mock_llm import MockLlmJudge
    from stage2_asr.types import PipelineConfig

    fixtures = Path(__file__).parent / "fixtures"
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
    (out / "glossary.json").write_text(
        json.dumps(
            {
                "terms": [
                    {
                        "surface": "AlphaProductXYZ",
                        "aliases": [],
                        "kind": "product",
                        "source": "seed",
                    }
                ],
                "keywords": [{"surface": "OldKeyword", "score": 0.8}],
                "rare_words": [{"surface": "OldRare"}],
            },
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )

    class ExtractUnionJudge(MockLlmJudge):
        def extract_terms(self, *, meeting: str, glossary: dict | None = None, **kwargs):
            raw = super().extract_terms(meeting=meeting, glossary=glossary, **kwargs)
            raw.setdefault("new_terms", []).append(
                {
                    "surface": "NewTermThisRound",
                    "aliases": ["nt"],
                    "kind": "other",
                    "latex": None,
                }
            )
            raw["keywords"] = [{"surface": "NewKeyword", "score": 1.0}]
            raw["rare_words"] = [{"surface": "NewRare"}]
            return raw

    run_pipeline(
        input_json=fixtures / "mode_c.json",
        audio_path=tmp_path / "missing.wav",
        work_dir=out,
        asr_runner=MockAsrRunner(),
        llm_judge=ExtractUnionJudge(),
        config=PipelineConfig(glossary=None),
        stage="publish",
    )
    loaded = json.loads((out / "glossary.json").read_text(encoding="utf-8"))
    surfaces = {t.get("surface") for t in loaded.get("terms") or [] if isinstance(t, dict)}
    assert "AlphaProductXYZ" in surfaces
    assert "NewTermThisRound" in surfaces
    kw = {k.get("surface") for k in loaded.get("keywords") or [] if isinstance(k, dict)}
    assert "OldKeyword" in kw
    assert "NewKeyword" in kw
    rare = {r.get("surface") for r in loaded.get("rare_words") or [] if isinstance(r, dict)}
    assert "OldRare" in rare
    assert "NewRare" in rare


def test_publish_cli_glossary_unions_with_work_dir(tmp_path: Path):
    from stage2_asr.pipeline import run_pipeline
    from stage2_asr.runners.mock_asr import MockAsrRunner
    from stage2_asr.runners.mock_llm import MockLlmJudge
    from stage2_asr.types import PipelineConfig

    fixtures = Path(__file__).parent / "fixtures"
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
    (out / "glossary.json").write_text(
        json.dumps(
            {
                "terms": [
                    {"surface": "FromWorkDir", "aliases": [], "kind": "other"},
                    {
                        "surface": "SharedTerm",
                        "aliases": ["old"],
                        "kind": "other",
                        "source": "extract",
                    },
                ],
                "keywords": [{"surface": "OldKeyword", "score": 0.1}],
                "rare_words": [],
            },
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    run_pipeline(
        input_json=fixtures / "mode_c.json",
        audio_path=tmp_path / "missing.wav",
        work_dir=out,
        asr_runner=MockAsrRunner(),
        llm_judge=MockLlmJudge(),
        config=PipelineConfig(
            glossary={
                "terms": [
                    {"surface": "FromCli", "aliases": [], "kind": "other"},
                    {
                        "surface": "SharedTerm",
                        "aliases": ["cli"],
                        "kind": "product",
                    },
                ],
                "keywords": [{"surface": "CliKeyword", "score": 1.0}],
                "rare_words": [],
            }
        ),
        stage="publish",
    )
    loaded = json.loads((out / "glossary.json").read_text(encoding="utf-8"))
    by_surface = {t.get("surface"): t for t in loaded.get("terms") or [] if isinstance(t, dict)}
    assert "FromCli" in by_surface
    assert "FromWorkDir" in by_surface
    assert by_surface["SharedTerm"].get("kind") == "product"
    assert by_surface["SharedTerm"].get("aliases") == ["cli"]
    kw = {k.get("surface") for k in loaded.get("keywords") or [] if isinstance(k, dict)}
    assert "OldKeyword" in kw
    assert "CliKeyword" in kw


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
