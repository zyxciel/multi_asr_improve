from __future__ import annotations

import json
from pathlib import Path

from stage2_asr.hotwords import load_hotwords


def test_load_hotwords_json_array(tmp_path: Path):
    path = tmp_path / "hw.json"
    path.write_text(json.dumps(["单框架", "账号|帐号", "单框架"], ensure_ascii=False), encoding="utf-8")
    assert load_hotwords(path) == ["单框架", "账号|帐号"]


def test_load_hotwords_json_object(tmp_path: Path):
    path = tmp_path / "hw.json"
    path.write_text(json.dumps({"hotwords": ["昇腾", "鸿蒙"]}, ensure_ascii=False), encoding="utf-8")
    assert load_hotwords(path) == ["昇腾", "鸿蒙"]


def test_load_hotwords_plaintext(tmp_path: Path):
    path = tmp_path / "hw.txt"
    path.write_text("昇腾\n\n# comment\n鸿蒙\n昇腾\n账号|帐号\n", encoding="utf-8")
    assert load_hotwords(path) == ["昇腾", "鸿蒙", "账号|帐号"]


def test_load_docs_hotwords_txt():
    path = Path(__file__).resolve().parents[1] / "docs" / "hotwords.txt"
    words = load_hotwords(path)
    assert len(words) >= 700
    assert "昇腾" in words
    assert "鸿蒙操作系统" in words
    assert len(words) == len(set(words))


def test_load_hotwords_none():
    assert load_hotwords(None) == []


def test_cap_hotword_list_keeps_whole_terms_under_budget():
    from stage2_asr.hotwords import cap_hotword_list

    words = ["昇腾", "鸿蒙操作系统", "单框架"]
    capped = cap_hotword_list(words, max_chars=10)
    joined = "".join(capped)
    assert len(joined) <= 10
    assert capped == ["昇腾"]
    assert cap_hotword_list(words, max_chars=0) == []
    assert cap_hotword_list(words, max_chars=10_000) == words
