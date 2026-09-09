from __future__ import annotations

import argparse

from stage2_asr.cli import _add_common_run_args, _pipeline_config, resolve_llm_api_key
from stage2_asr.types import PipelineConfig


def test_pipeline_config_polish_cluster_default_true():
    assert PipelineConfig().polish_cluster is True


def test_no_polish_cluster_argparse_default_false():
    parser = argparse.ArgumentParser()
    _add_common_run_args(parser)
    args = parser.parse_args([])
    assert args.no_polish_cluster is False


def test_no_polish_cluster_sets_polish_cluster_false():
    parser = argparse.ArgumentParser()
    _add_common_run_args(parser)
    args = parser.parse_args(["--no-polish-cluster"])
    assert args.no_polish_cluster is True
    cfg = _pipeline_config(args)
    assert cfg.polish_cluster is False


def test_pipeline_config_polish_cluster_true_without_flag():
    parser = argparse.ArgumentParser()
    _add_common_run_args(parser)
    args = parser.parse_args([])
    cfg = _pipeline_config(args)
    assert cfg.polish_cluster is True


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
