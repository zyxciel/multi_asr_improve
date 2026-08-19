from __future__ import annotations

import json
from pathlib import Path

import pytest

from stage2_asr.pipeline import run_pipeline
from stage2_asr.polish import apply_polish_edits, run_polish
from stage2_asr.polish_prompt import POLISH_SYSTEM_PROMPT, render_polish_user_prompt
from stage2_asr.runners.mock_asr import MockAsrRunner
from stage2_asr.runners.mock_llm import MockLlmJudge
from stage2_asr.types import Hypothesis, PipelineConfig, Turn


FIXTURES = Path(__file__).parent / "fixtures"


def test_apply_polish_edits_records_char_locations():
    text = "明天三点开会用gpu"
    new_text, located = apply_polish_edits(
        text,
        [
            {"span_asr": "三点", "span_out": "3点", "kind": "itn"},
            {"span_asr": "gpu", "span_out": "GPU", "kind": "codeswitch"},
        ],
    )
    assert new_text == "明天3点开会用GPU"
    by_kind = {e["kind"]: e for e in located}
    assert by_kind["itn"]["start_char"] == 2
    assert by_kind["itn"]["end_char"] == 4
    assert text[2:4] == "三点"
    assert by_kind["codeswitch"]["start_char"] == 7
    assert by_kind["codeswitch"]["end_char"] == 10


def test_apply_polish_edits_skips_missing_and_overlapping_spans():
    text = "三点开会"
    new_text, located = apply_polish_edits(
        text,
        [
            {"span_asr": "不存在", "span_out": "x", "kind": "entity"},
            {"span_asr": "三点开会", "span_out": "3点开会", "kind": "itn"},
            {"span_asr": "三点", "span_out": "3点", "kind": "itn"},
        ],
    )
    assert new_text == "3点开会"
    assert len(located) == 1
    assert located[0]["span_asr"] == "三点开会"


def test_apply_polish_edits_respects_llm_start_char_when_valid():
    text = "用gpu还是gpu"
    new_text, located = apply_polish_edits(
        text,
        [{"span_asr": "gpu", "span_out": "GPU", "kind": "codeswitch", "start_char": 6}],
    )
    assert new_text == "用gpu还是GPU"
    assert located[0]["start_char"] == 6


def test_run_polish_applies_typed_edits_and_keeps_unedited_turns():
    turns = [
        Turn(0, 2, "s0", "明天三点开会用gpu"),
        Turn(2, 4, "s1", "好的"),
    ]
    texts = {0: turns[0].text, 1: turns[1].text}

    class PolishFix:
        def polish(self, **kwargs):
            text = kwargs["text"]
            if "三点" in text:
                return {
                    "text": "明天3点开会用GPU。",
                    "edits": [
                        {"span_asr": "三点", "span_out": "3点", "kind": "itn"},
                        {"span_asr": "gpu", "span_out": "GPU", "kind": "codeswitch"},
                        {
                            "span_asr": "",
                            "span_out": "。",
                            "kind": "punc",
                            "start_char": len(text),
                        },
                    ],
                }
            return {"text": "改成别的", "edits": []}

    out, audits = run_polish(turns, texts, llm_judge=PolishFix())
    assert out[0] == "明天3点开会用GPU。"
    assert out[1] == "好的"
    kinds = {a["kind"] for a in audits if a.get("path") == "llm"}
    assert kinds == {"itn", "codeswitch", "punc"}
    itn = next(a for a in audits if a.get("kind") == "itn")
    assert itn["turn_index"] == 0
    assert itn["start_char"] == 2
    assert itn["end_char"] == 4
    assert itn["pass"] == "polish"


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


def test_polish_prompt_covers_four_display_tasks():
    assert "punctuation" in POLISH_SYSTEM_PROMPT.lower() or "标点" in POLISH_SYSTEM_PROMPT
    prompt = render_polish_user_prompt(
        text="明天三点用gpu连微信",
        neighbor_draft="[]",
        hotwords='["GPU"]',
        turn_index=0,
        hypotheses="- qwen: 以前那个Windows的问题",
    )
    lowered = prompt.lower()
    assert "itn" in lowered or "inverse text" in lowered
    assert "entity" in lowered or "实体" in prompt
    assert "code" in lowered or "中英" in prompt
    assert "punct" in lowered or "标点" in prompt
    assert "明天三点用gpu连微信" in prompt
    assert "Windows" in prompt
    assert "world knowledge" in lowered or "世界知识" in prompt
    assert "娃娃亲" in prompt


def test_run_polish_recovers_windows_from_asr_hyp():
    """Pass A/B cannot apply 温度→Windows (|Δlen|=5); polish may, if a hyp has Windows."""
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
    assert hit["end_char"] - hit["start_char"] == 2
    assert abs(len("Windows") - 2) > 1


def test_run_polish_recovers_wawapro_from_neighbor_context():
    """Pass A/B rejects 爱情→娃娃亲 (no pinyin link); polish may use neighbors."""
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
    assert (out / "mode_c_polished.json").exists()
    assert result.get("polished_path") is not None
    asr_final = json.loads((out / "mode_c_asr_final.json").read_text(encoding="utf-8"))
    polished = json.loads((out / "mode_c_polished.json").read_text(encoding="utf-8"))
    assert asr_final["meta"]["stage"] == "pass_b_final"
    assert polished["meta"]["stage"] == "polish"
    edits = [
        json.loads(line)
        for line in (out / "llm_edits.jsonl").read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    assert any(e.get("pass") == "A" for e in edits)
    assert any(e.get("pass") == "polish" for e in edits)
    stats = json.loads((out / "pass_stats.json").read_text(encoding="utf-8"))
    assert "polish" in stats


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
    assert "3点" in text
    assert "GPU" in text
    assert text.endswith("。")
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
    asr_final = json.loads((out / "mode_c_asr_final.json").read_text(encoding="utf-8"))
    assert asr_final["turns"][0]["text"] == "以前那个温度的问题"


def test_qwen_polish_logs_prompt_and_response():
    from stage2_asr.runners.llm_qwen36 import Qwen36LlmJudge

    logs: list[dict] = []

    def gen(system, user):
        assert "display" in system.lower()
        return json.dumps(
            {
                "text": "明天3点开会。",
                "edits": [{"span_asr": "三点", "span_out": "3点", "kind": "itn"}],
            },
            ensure_ascii=False,
        )

    judge = Qwen36LlmJudge(enabled=True, generate_fn=gen, log_fn=logs.append)
    out = judge.polish(
        text="明天三点开会",
        neighbor_draft=[],
        hotwords=[],
        turn_index=0,
        unit_id="polish_t0",
    )
    assert out["edits"][0]["kind"] == "itn"
    polish_logs = [e for e in logs if e.get("pass") == "polish"]
    assert polish_logs
    assert "明天三点开会" in (polish_logs[0].get("user") or "")
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
