from __future__ import annotations

import argparse

from stage2_asr.cli import _add_common_run_args, _pipeline_config, resolve_llm_api_key
from stage2_asr.model_paths import (
    DEFAULT_FIRERED_ASR_MODEL_DIR,
    DEFAULT_FIRERED_LID_MODEL_DIR,
    DEFAULT_FIRERED_PUNC_MODEL_DIR,
    DEFAULT_QWEN_MODEL_ID,
    resolve_firered_model_dirs,
    resolve_qwen_model_id,
)


def test_cli_batch_scale_flags_parse():
    parser = argparse.ArgumentParser()
    _add_common_run_args(parser)
    args = parser.parse_args(
        [
            "--llm-log-mode",
            "off",
            "--neighbor-max-turns",
            "8",
            "--neighbor-window-seconds",
            "120",
            "--neighbor-char-budget",
            "2048",
            "--hotword-prompt-chars",
            "500",
        ]
    )
    cfg = _pipeline_config(args)
    assert args.llm_log_mode == "off"
    assert cfg.llm_log_mode == "off"
    assert cfg.neighbor_max_turns == 8
    assert cfg.neighbor_window_seconds == 120.0
    assert cfg.neighbor_char_budget == 2048
    assert cfg.hotword_prompt_chars == 500


def test_resolve_llm_api_key_prefers_cli_then_stage2_then_openai(monkeypatch):
    monkeypatch.delenv("STAGE2_LLM_API_KEY", raising=False)
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    assert resolve_llm_api_key(None) is None
    assert resolve_llm_api_key("") is None

    monkeypatch.setenv("OPENAI_API_KEY", "openai-secret")
    assert resolve_llm_api_key(None) == "openai-secret"

    monkeypatch.setenv("STAGE2_LLM_API_KEY", "stage2-secret")
    assert resolve_llm_api_key(None) == "stage2-secret"
    assert resolve_llm_api_key("cli-secret") == "cli-secret"


def test_resolve_qwen_model_id_cli_then_stage2_then_qwen_env(monkeypatch):
    monkeypatch.delenv("STAGE2_QWEN_MODEL_ID", raising=False)
    monkeypatch.delenv("QWEN_MODEL_ID", raising=False)
    assert resolve_qwen_model_id(None) == DEFAULT_QWEN_MODEL_ID

    monkeypatch.setenv("QWEN_MODEL_ID", "/prod/Qwen3-ASR")
    assert resolve_qwen_model_id(None) == "/prod/Qwen3-ASR"
    monkeypatch.setenv("STAGE2_QWEN_MODEL_ID", "/debug/Qwen3-ASR")
    assert resolve_qwen_model_id(None) == "/debug/Qwen3-ASR"
    assert resolve_qwen_model_id("/cli/Qwen3-ASR") == "/cli/Qwen3-ASR"


def test_resolve_firered_dirs_cli_then_env_then_defaults(monkeypatch):
    for key in (
        "STAGE2_FIRERED_ASR_MODEL_DIR",
        "STAGE2_FIRERED_LID_MODEL_DIR",
        "STAGE2_FIRERED_PUNC_MODEL_DIR",
        "FIRERED_ASR_MODEL_DIR",
        "FIRERED_LID_MODEL_DIR",
        "FIRERED_PUNC_MODEL_DIR",
    ):
        monkeypatch.delenv(key, raising=False)
    asr, lid, punc = resolve_firered_model_dirs()
    assert asr == DEFAULT_FIRERED_ASR_MODEL_DIR
    assert lid == DEFAULT_FIRERED_LID_MODEL_DIR
    assert punc == DEFAULT_FIRERED_PUNC_MODEL_DIR

    monkeypatch.setenv("FIRERED_ASR_MODEL_DIR", "/prod/FireRedASR2-AED")
    monkeypatch.setenv("STAGE2_FIRERED_ASR_MODEL_DIR", "/debug/FireRedASR2-AED")
    asr, lid, punc = resolve_firered_model_dirs()
    assert asr == "/debug/FireRedASR2-AED"
    asr, lid, punc = resolve_firered_model_dirs(asr_model_dir="/cli/FireRedASR2-AED")
    assert asr == "/cli/FireRedASR2-AED"
    assert lid == DEFAULT_FIRERED_LID_MODEL_DIR


def test_cli_parses_firered_model_dir_flags():
    parser = argparse.ArgumentParser()
    _add_common_run_args(parser)
    args = parser.parse_args(
        [
            "--qwen-model-id",
            "/debug/Qwen3-ASR",
            "--firered-asr-model-dir",
            "/debug/FireRedASR2-AED",
            "--firered-lid-model-dir",
            "/debug/FireRedLID",
            "--firered-punc-model-dir",
            "/debug/FireRedPunc",
        ]
    )
    assert args.qwen_model_id == "/debug/Qwen3-ASR"
    assert args.firered_asr_model_dir == "/debug/FireRedASR2-AED"
    assert args.firered_lid_model_dir == "/debug/FireRedLID"
    assert args.firered_punc_model_dir == "/debug/FireRedPunc"
