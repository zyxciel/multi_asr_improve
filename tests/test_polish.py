from __future__ import annotations

import json
from pathlib import Path

import pytest

from stage2_asr.pipeline import run_pipeline
from stage2_asr.polish import apply_polish_edits, run_polish, validate_polish_edits
from stage2_asr.polish_prompt import POLISH_SYSTEM_PROMPT, render_polish_user_prompt
from stage2_asr.runners.mock_asr import MockAsrRunner
from stage2_asr.runners.mock_llm import MockLlmJudge
from stage2_asr.types import Hypothesis, PipelineConfig, Turn


FIXTURES = Path(__file__).parent / "fixtures"


def test_apply_polish_edits_records_char_locations():
    text = "明天开会用gpu找张三风"
    new_text, located = apply_polish_edits(
        text,
        [
            {"span_asr": "gpu", "span_out": "GPU", "kind": "codeswitch"},
            {"span_asr": "张三风", "span_out": "张三丰", "kind": "entity"},
        ],
    )
    assert new_text == "明天开会用GPU找张三丰"
    by_kind = {e["kind"]: e for e in located}
    assert by_kind["codeswitch"]["start_char"] == 5
    assert by_kind["codeswitch"]["end_char"] == 8
    assert by_kind["entity"]["start_char"] == 9
    assert by_kind["entity"]["end_char"] == 12


def test_apply_polish_edits_skips_missing_and_overlapping_spans():
    text = "张三风开会"
    new_text, located = apply_polish_edits(
        text,
        [
            {"span_asr": "不存在", "span_out": "x", "kind": "entity"},
            {"span_asr": "张三风开会", "span_out": "张三丰开会", "kind": "entity"},
            {"span_asr": "张三风", "span_out": "张三丰", "kind": "entity"},
        ],
    )
    assert new_text == "张三丰开会"
    assert len(located) == 1
    assert located[0]["span_asr"] == "张三风开会"


def test_apply_polish_edits_respects_llm_start_char_when_valid():
    text = "用gpu还是gpu"
    new_text, located = apply_polish_edits(
        text,
        [{"span_asr": "gpu", "span_out": "GPU", "kind": "codeswitch", "start_char": 6}],
    )
    assert new_text == "用gpu还是GPU"
    assert located[0]["start_char"] == 6


def test_apply_polish_edits_skips_itn_kind():
    text = "明天三点开会"
    new_text, located = apply_polish_edits(
        text,
        [{"span_asr": "三点", "span_out": "3点", "kind": "itn"}],
    )
    assert new_text == text
    assert located == []


def test_validate_rejects_itn_and_number_rewrites():
    text = "读0.61再报532"
    ok, err = validate_polish_edits(
        [{"span_asr": "三点", "span_out": "3点", "kind": "itn"}],
        text="明天三点开会",
    )
    assert ok is False
    assert "itn" in (err or "").lower() or "kind" in (err or "").lower()

    ok, err = validate_polish_edits(
        [
            {
                "span_asr": "0.61",
                "span_out": "zero point sixty-one",
                "kind": "entity",
                "anchor": "hyp",
                "evidence": "style",
            }
        ],
        text=text,
        hypotheses=[Hypothesis("qwen", "zero point sixty-one")],
    )
    assert ok is False
    assert "number" in (err or "").lower() or "digit" in (err or "").lower()

    ok, err = validate_polish_edits(
        [
            {
                "span_asr": "532",
                "span_out": "five hundred thirty-two",
                "kind": "entity",
                "anchor": "hyp",
                "evidence": "style",
            }
        ],
        text=text,
        hypotheses=[Hypothesis("qwen", "five hundred thirty-two")],
    )
    assert ok is False


def test_validate_cjk_to_cjk_allows_one_or_two_char_slack():
    ok, err = validate_polish_edits(
        [
            {
                "span_asr": "爱情",
                "span_out": "娃娃亲",
                "kind": "entity",
                "anchor": "neighbor_draft",
                "evidence": "neighbor has 娃娃亲",
            }
        ],
        text="他们说的爱情到底怎么办",
        neighbors=[{"text": "娃娃亲这件事要处理"}],
    )
    assert ok is True, err


def test_validate_cjk_to_cjk_rejects_more_than_two_char_delta():
    ok, err = validate_polish_edits(
        [
            {
                "span_asr": "爱情",
                "span_out": "娃娃亲的事情",
                "kind": "entity",
                "anchor": "neighbor_draft",
                "evidence": "neighbor has 娃娃亲的事情",
            }
        ],
        text="他们说的爱情到底怎么办",
        neighbors=[{"text": "娃娃亲的事情要处理"}],
    )
    assert ok is False
    assert "length" in (err or "").lower() or "slack" in (err or "").lower() or "字数" in (err or "") or "count" in (err or "").lower()


def test_validate_cjk_to_cjk_same_length_with_evidence():
    ok, err = validate_polish_edits(
        [
            {
                "span_asr": "张三风",
                "span_out": "张三丰",
                "kind": "entity",
                "anchor": "neighbor_draft",
                "evidence": "neighbor turn contains 张三丰",
            }
        ],
        text="找张三风签字",
        neighbors=[{"text": "张三丰已经到了"}],
    )
    assert ok is True, err


def test_validate_cjk_to_en_length_may_change_with_hyp_evidence():
    ok, err = validate_polish_edits(
        [
            {
                "span_asr": "温度",
                "span_out": "Windows",
                "kind": "codeswitch",
                "anchor": "hyp",
                "evidence": "qwen hyp contains Windows",
            }
        ],
        text="以前那个温度的问题",
        hypotheses=[Hypothesis("qwen", "以前那个Windows的问题")],
    )
    assert ok is True, err


def test_validate_cjk_to_mixed_cn_en_with_hyp_evidence():
    ok, err = validate_polish_edits(
        [
            {
                "span_asr": "温度的问题",
                "span_out": "Windows产品",
                "kind": "codeswitch",
                "anchor": "hyp",
                "evidence": "qwen hyp contains windows产品",
            }
        ],
        text="以前那个温度的问题",
        hypotheses=[Hypothesis("qwen", "以前那个windows产品")],
    )
    assert ok is True, err


def test_validate_rejects_entity_without_evidence_source():
    ok, err = validate_polish_edits(
        [
            {
                "span_asr": "温度",
                "span_out": "Windows",
                "kind": "codeswitch",
                "anchor": "world",
                "evidence": "product name",
            }
        ],
        text="以前那个温度的问题",
        hypotheses=[Hypothesis("moss", "以前那个温度的问题")],
    )
    assert ok is False

    ok, err = validate_polish_edits(
        [{"span_asr": "帐号", "span_out": "账号", "kind": "entity"}],
        text="登录帐号",
    )
    assert ok is False


def test_validate_rejects_add_delete_continue_and_repeat_collapse():
    ok, _ = validate_polish_edits(
        [{"span_asr": "", "span_out": "。", "kind": "punc", "start_char": 4}],
        text="好的啊",
    )
    assert ok is False

    ok, _ = validate_polish_edits(
        [
            {
                "span_asr": "明天开会",
                "span_out": "明天开会然后我们再讨论预算问题",
                "kind": "entity",
                "anchor": "hyp",
                "evidence": "invented continuation",
            }
        ],
        text="明天开会",
        hypotheses=[Hypothesis("qwen", "明天开会然后我们再讨论预算问题")],
    )
    assert ok is False

    ok, _ = validate_polish_edits(
        [
            {
                "span_asr": "好好好",
                "span_out": "好",
                "kind": "entity",
                "anchor": "hyp",
                "evidence": "collapse repeat",
            }
        ],
        text="好好好",
        hypotheses=[Hypothesis("qwen", "好")],
    )
    assert ok is False


def test_validate_punc_may_only_change_punctuation():
    ok, err = validate_polish_edits(
        [{"span_asr": "大家好明天见", "span_out": "大家好，明天见", "kind": "punc"}],
        text="大家好明天见",
    )
    assert ok is True, err

    ok, _ = validate_polish_edits(
        [{"span_asr": "大家好明天见", "span_out": "大家好，明天见吧", "kind": "punc"}],
        text="大家好明天见",
    )
    assert ok is False


def test_run_polish_applies_typed_edits_and_keeps_unedited_turns():
    turns = [
        Turn(0, 2, "s0", "明天开会用gpu"),
        Turn(2, 4, "s1", "好的"),
    ]
    texts = {0: turns[0].text, 1: turns[1].text}

    class PolishFix:
        def polish(self, **kwargs):
            text = kwargs["text"]
            if "gpu" in text:
                return {
                    "text": "明天开会用GPU",
                    "edits": [
                        {
                            "span_asr": "gpu",
                            "span_out": "GPU",
                            "kind": "codeswitch",
                            "anchor": "hyp",
                            "evidence": "latin token already in the phonetic final",
                        },
                    ],
                }
            return {"text": "改成别的", "edits": []}

    out, audits = run_polish(turns, texts, llm_judge=PolishFix())
    assert out[0] == "明天开会用GPU"
    assert out[1] == "好的"
    kinds = {a["kind"] for a in audits if a.get("path") == "llm"}
    assert kinds == {"codeswitch"}
    hit = next(a for a in audits if a.get("kind") == "codeswitch")
    assert hit["turn_index"] == 0
    assert hit["span_asr"] == "gpu"
    assert hit["evidence"]
    assert hit["pass"] == "polish"


def test_run_polish_rejects_whole_turn_rewrite_without_edits():
    turns = [Turn(0, 2, "s0", "明天开会")]
    texts = {0: "明天开会"}

    class Rewrite:
        def polish(self, **kwargs):
            return {"text": "完全无关的摘要", "edits": []}

    out, audits = run_polish(turns, texts, llm_judge=Rewrite())
    assert out[0] == "明天开会"
    assert any(a.get("path") == "empty_edits_reject" for a in audits)


def test_run_polish_retries_then_keeps_original_on_invalid_json():
    turns = [Turn(0, 2, "s0", "明天三点开会")]
    texts = {0: "明天三点开会"}

    class Bad:
        def __init__(self):
            self.n = 0

        def polish(self, **kwargs):
            self.n += 1
            return {"not": "a polish payload"}

    out, audits = run_polish(
        turns,
        texts,
        llm_judge=Bad(),
        config=PipelineConfig(llm_max_retries=2),
    )
    assert out[0] == "明天三点开会"
    assert any(a.get("fallback") for a in audits)


def test_polish_prompt_forbids_itn_and_undisciplined_rewrites():
    blob = POLISH_SYSTEM_PROMPT + render_polish_user_prompt(
        text="明天三点用gpu连微信",
        neighbor_draft="[]",
        hotwords='["GPU"]',
        turn_index=0,
        hypotheses="- qwen: 以前那个Windows的问题",
    )
    lowered = blob.lower()
    assert "punc|entity|codeswitch" in lowered
    assert "punc|entity|codeswitch|itn" not in lowered
    assert "hyp|neighbor_draft|meeting_draft|hotword|world" not in lowered
    assert "punctuation" in lowered or "标点" in blob
    assert "entity" in lowered or "实体" in blob
    assert "code" in lowered or "中英" in blob
    assert "明天三点用gpu连微信" in blob
    assert "Windows" in blob
    assert "字数" in blob or "character count" in lowered
    assert "evidence" in lowered or "证据" in blob
    assert "0.61" in blob and "reject" in lowered
    assert "no number" in lowered or "number normalization" in lowered


def test_run_polish_recovers_windows_from_asr_hyp():
    """CN→EN may change length when a hyp already contains the English form."""
    turns = [Turn(0, 2, "s0", "以前那个温度的问题")]
    texts = {0: "以前那个温度的问题"}
    hyps = {
        0: [
            Hypothesis("moss", "以前那个温度的问题"),
            Hypothesis("qwen", "以前那个Windows的问题"),
            Hypothesis("firered", "以前那个温度的问题"),
        ]
    }
    out, audits = run_polish(
        turns, texts, llm_judge=MockLlmJudge(), hyp_by_turn=hyps
    )
    assert "Windows" in out[0]
    assert "温度" not in out[0]
    hit = next(a for a in audits if a.get("span_asr") == "温度")
    assert hit["span_out"] == "Windows"
    assert hit["kind"] == "codeswitch"
    assert hit["anchor"] == "hyp"
    assert hit["evidence"]
    assert hit["end_char"] - hit["start_char"] == 2
    assert abs(len("Windows") - 2) > 1


def test_run_polish_recovers_same_length_cjk_entity_from_neighbor():
    turns = [
        Turn(0, 2, "s0", "找张三风签字"),
        Turn(2, 4, "s1", "张三丰已经到了"),
    ]
    texts = {0: turns[0].text, 1: turns[1].text}
    out, audits = run_polish(turns, texts, llm_judge=MockLlmJudge())
    assert "张三丰" in out[0]
    assert "张三风" not in out[0]
    hit = next(a for a in audits if a.get("span_asr") == "张三风")
    assert hit["span_out"] == "张三丰"
    assert hit["kind"] == "entity"
    assert hit["anchor"] == "neighbor_draft"
    assert hit["evidence"]


def test_run_polish_recovers_wawapro_within_cjk_slack():
    turns = [
        Turn(0, 2, "s0", "他们说的爱情到底怎么办"),
        Turn(2, 4, "s1", "娃娃亲这件事要处理"),
    ]
    texts = {0: turns[0].text, 1: turns[1].text}
    out, audits = run_polish(turns, texts, llm_judge=MockLlmJudge())
    assert "娃娃亲" in out[0]
    assert "爱情" not in out[0]
    hit = next(a for a in audits if a.get("span_asr") == "爱情")
    assert hit["span_out"] == "娃娃亲"
    assert hit["kind"] == "entity"
    assert hit["anchor"] == "neighbor_draft"
    assert hit["evidence"]


def test_polish_batch_size_is_independent_of_pass_a():
    class Rec:
        def __init__(self):
            self.sizes: list[int] = []

        def polish(self, **kwargs):
            text = kwargs["text"]
            return {"text": text, "edits": []}

        def polish_many(self, jobs, max_workers=1):
            self.sizes.append(len(jobs))
            return [self.polish(**job) for job in jobs]

    turns = [
        Turn(0, 1, "s0", "你好啊"),
        Turn(1, 2, "s0", "大家好呀"),
        Turn(2, 3, "s0", "明天见吧"),
    ]
    texts = {i: t.text for i, t in enumerate(turns)}

    sequential = Rec()
    run_polish(
        turns,
        texts,
        llm_judge=sequential,
        config=PipelineConfig(pass_a_batch_size=64, polish_batch_size=1),
    )
    assert sequential.sizes == []

    batched = Rec()
    run_polish(
        turns,
        texts,
        llm_judge=batched,
        config=PipelineConfig(pass_a_batch_size=1, polish_batch_size=3),
    )
    assert batched.sizes == [3]


def test_pipeline_all_writes_polished_without_clobbering_asr_final(tmp_path: Path):
    out = tmp_path / "work"
    result = run_pipeline(
        input_json=FIXTURES / "mode_c.json",
        audio_path=tmp_path / "missing.wav",
        work_dir=out,
        asr_runner=MockAsrRunner(),
        llm_judge=MockLlmJudge(),
        config=PipelineConfig(),
        hotwords=["单框架|单方接"],
        stage="all",
    )
    assert (out / "mode_c_asr_final.json").exists()
    assert (out / "mode_c_draft.json").exists()
    assert (out / "mode_c_draft_merged.json").exists()
    assert (out / "mode_c_asr_final_merged.json").exists()
    assert (out / "mode_c_polished.json").exists()
    assert result.get("polished_path") is not None
    asr_final = json.loads((out / "mode_c_asr_final.json").read_text(encoding="utf-8"))
    merged = json.loads((out / "mode_c_asr_final_merged.json").read_text(encoding="utf-8"))
    polished = json.loads((out / "mode_c_polished.json").read_text(encoding="utf-8"))
    assert asr_final["meta"]["stage"] == "pass_b_final"
    assert merged["meta"]["grid"] == "merged"
    assert polished["meta"]["stage"] == "polish"
    assert polished["meta"]["grid"] == "merged"
    assert len(merged["turns"]) <= len(asr_final["turns"])
    assert len(polished["turns"]) == len(merged["turns"])
    edits = [
        json.loads(line)
        for line in (out / "llm_edits.jsonl").read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    assert any(e.get("pass") == "A" for e in edits)
    stats = json.loads((out / "pass_stats.json").read_text(encoding="utf-8"))
    assert "polish" in stats
    assert stats["polish"]["n_audits"] >= 0


def test_pipeline_llm_stage_skips_polish(tmp_path: Path):
    out = tmp_path / "work"
    run_pipeline(
        input_json=FIXTURES / "mode_c.json",
        audio_path=tmp_path / "missing.wav",
        work_dir=out,
        asr_runner=MockAsrRunner(),
        llm_judge=MockLlmJudge(),
        config=PipelineConfig(),
        hotwords=["单框架|单方接"],
        stage="asr",
        asr_models=["moss", "qwen"],
    )
    run_pipeline(
        input_json=FIXTURES / "mode_c.json",
        audio_path=tmp_path / "missing.wav",
        work_dir=out,
        asr_runner=MockAsrRunner(),
        llm_judge=MockLlmJudge(),
        config=PipelineConfig(),
        hotwords=["单框架|单方接"],
        stage="llm",
    )
    assert (out / "mode_c_asr_final.json").exists()
    assert not (out / "mode_c_polished.json").exists()


def test_pipeline_polish_stage_reads_asr_final(tmp_path: Path):
    out = tmp_path / "work"
    final_doc = {
        "meta": {"stage": "pass_b_final"},
        "turns": [
            {
                "start": 0.0,
                "end": 2.0,
                "speaker_id": "s0",
                "text": "明天三点开会用gpu",
                "asr_status": "final",
            }
        ],
    }
    (out).mkdir(parents=True, exist_ok=True)
    (out / "mode_c_asr_final.json").write_text(
        json.dumps(final_doc, ensure_ascii=False), encoding="utf-8"
    )
    result = run_pipeline(
        input_json=FIXTURES / "mode_c.json",
        audio_path=tmp_path / "missing.wav",
        work_dir=out,
        asr_runner=MockAsrRunner(),
        llm_judge=MockLlmJudge(),
        config=PipelineConfig(),
        stage="polish",
    )
    assert result["stage"] == "polish"
    polished = json.loads((out / "mode_c_polished.json").read_text(encoding="utf-8"))
    text = polished["turns"][0]["text"]
    assert "三点" in text
    assert "3点" not in text
    assert "GPU" in text
    assert not text.endswith("。")
    asr_final = json.loads((out / "mode_c_asr_final.json").read_text(encoding="utf-8"))
    assert asr_final["turns"][0]["text"] == "明天三点开会用gpu"


def test_pipeline_polish_stage_uses_saved_asr_hyps(tmp_path: Path):
    out = tmp_path / "work"
    out.mkdir(parents=True, exist_ok=True)
    final_doc = {
        "meta": {"stage": "pass_b_final"},
        "turns": [
            {
                "start": 0.0,
                "end": 2.0,
                "speaker_id": "s0",
                "text": "以前那个温度的问题",
                "asr_status": "final",
            }
        ],
    }
    (out / "mode_c_asr_final.json").write_text(
        json.dumps(final_doc, ensure_ascii=False), encoding="utf-8"
    )
    (out / "asr_hypotheses.json").write_text(
        json.dumps(
            {
                "records": [
                    {
                        "unit_id": "u0",
                        "turn_indices": [0],
                        "hyps": [
                            {"model": "moss", "text": "以前那个温度的问题"},
                            {"model": "qwen", "text": "以前那个Windows的问题"},
                            {"model": "firered", "text": "以前那个温度的问题"},
                        ],
                    }
                ]
            },
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    run_pipeline(
        input_json=FIXTURES / "mode_c.json",
        audio_path=tmp_path / "missing.wav",
        work_dir=out,
        asr_runner=MockAsrRunner(),
        llm_judge=MockLlmJudge(),
        config=PipelineConfig(),
        stage="polish",
    )
    polished = json.loads((out / "mode_c_polished.json").read_text(encoding="utf-8"))
    assert "Windows" in polished["turns"][0]["text"]
    edits = [
        json.loads(line)
        for line in (out / "llm_edits.jsonl").read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    win = next(e for e in edits if e.get("span_asr") == "温度")
    assert win.get("evidence")
    assert win.get("anchor") == "hyp"
    asr_final = json.loads((out / "mode_c_asr_final.json").read_text(encoding="utf-8"))
    assert asr_final["turns"][0]["text"] == "以前那个温度的问题"


def test_pipeline_polish_merges_fragments_before_llm(tmp_path: Path):
    out = tmp_path / "work"
    out.mkdir(parents=True, exist_ok=True)
    final_doc = {
        "meta": {"stage": "pass_b_final"},
        "turns": [
            {
                "start": 0.0,
                "end": 1.0,
                "speaker_id": "s0",
                "text": "以前那个温",
                "asr_status": "final",
            },
            {
                "start": 1.1,
                "end": 2.0,
                "speaker_id": "s0",
                "text": "度的问题",
                "asr_status": "final",
            },
        ],
    }
    (out / "mode_c_asr_final.json").write_text(
        json.dumps(final_doc, ensure_ascii=False), encoding="utf-8"
    )
    (out / "asr_units.json").write_text(
        json.dumps(
            {
                "units": [
                    {
                        "unit_id": "unit_0000",
                        "start": 0.0,
                        "end": 2.0,
                        "speaker_id": "s0",
                        "turn_indices": [0, 1],
                        "moss_merged": True,
                    }
                ]
            },
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    (out / "asr_hypotheses.json").write_text(
        json.dumps(
            {
                "records": [
                    {
                        "unit_id": "unit_0000",
                        "turn_indices": [0, 1],
                        "hyps": [
                            {"model": "moss", "text": "以前那个温度的问题"},
                            {"model": "qwen", "text": "以前那个Windows的问题"},
                        ],
                    }
                ]
            },
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    run_pipeline(
        input_json=FIXTURES / "mode_c.json",
        audio_path=tmp_path / "missing.wav",
        work_dir=out,
        asr_runner=MockAsrRunner(),
        llm_judge=MockLlmJudge(),
        config=PipelineConfig(),
        stage="polish",
    )
    merged = json.loads((out / "mode_c_asr_final_merged.json").read_text(encoding="utf-8"))
    assert len(merged["turns"]) == 1
    assert merged["turns"][0]["text"] == "以前那个温度的问题"
    polished = json.loads((out / "mode_c_polished.json").read_text(encoding="utf-8"))
    assert len(polished["turns"]) == 1
    assert polished["turns"][0]["text"].count("Windows") == 1
    assert "温度" not in polished["turns"][0]["text"]
    asr_final = json.loads((out / "mode_c_asr_final.json").read_text(encoding="utf-8"))
    assert len(asr_final["turns"]) == 2


def test_qwen_polish_logs_prompt_and_response():
    from stage2_asr.runners.llm_qwen36 import Qwen36LlmJudge

    logs: list[dict] = []

    def gen(system, user):
        assert "minimal" in system.lower() or "phonetic" in system.lower() or "证据" in system
        return json.dumps(
            {
                "text": "明天开会用GPU",
                "edits": [
                    {
                        "span_asr": "gpu",
                        "span_out": "GPU",
                        "kind": "codeswitch",
                        "evidence": "latin already in text",
                    }
                ],
            },
            ensure_ascii=False,
        )

    judge = Qwen36LlmJudge(enabled=True, generate_fn=gen, log_fn=logs.append)
    out = judge.polish(
        text="明天开会用gpu",
        neighbor_draft=[],
        hotwords=[],
        turn_index=0,
        unit_id="polish_t0",
    )
    assert out["edits"][0]["kind"] == "codeswitch"
    polish_logs = [e for e in logs if e.get("pass") == "polish"]
    assert polish_logs
    assert "明天开会用gpu" in (polish_logs[0].get("user") or "")
    assert polish_logs[0].get("response")


def test_pipeline_polish_stage_requires_asr_final(tmp_path: Path):
    out = tmp_path / "work"
    with pytest.raises(FileNotFoundError, match="mode_c_asr_final"):
        run_pipeline(
            input_json=FIXTURES / "mode_c.json",
            audio_path=tmp_path / "missing.wav",
            work_dir=out,
            asr_runner=MockAsrRunner(),
            llm_judge=MockLlmJudge(),
            config=PipelineConfig(),
            stage="polish",
        )
