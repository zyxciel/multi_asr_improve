from __future__ import annotations

from stage2_asr.cli import resolve_llm_api_key


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
