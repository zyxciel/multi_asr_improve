from __future__ import annotations

from stage2_asr.publish_itn import itn_edit_allowed


def test_serial_financial_to_arabic_allowed():
    ok, err = itn_edit_allowed("伍柒叁", "573")
    assert ok is True, err


def test_serial_must_not_become_wrong_place_value():
    ok, err = itn_edit_allowed("伍柒叁", "五百三十七")
    assert ok is False
    assert err


def test_serial_must_not_become_matching_place_value_phrase():
    ok, err = itn_edit_allowed("伍柒叁", "五百七十三")
    assert ok is False
    assert err


def test_place_value_to_arabic_allowed():
    ok, err = itn_edit_allowed("五百三十七", "537")
    assert ok is True, err


def test_place_value_wrong_digits_rejected():
    ok, err = itn_edit_allowed("五百三十七", "573")
    assert ok is False
    assert err


def test_tts_expansion_rejected():
    ok, err = itn_edit_allowed("573", "five hundred seventy-three")
    assert ok is False
    assert err


def test_arabic_to_spoken_chinese_rejected():
    ok, err = itn_edit_allowed("573", "五百七十三")
    assert ok is False


def test_bare_san_dian_ambiguous_rejected():
    ok, err = itn_edit_allowed("三点", "3点")
    assert ok is False


def test_english_serial_to_arabic_allowed():
    ok, err = itn_edit_allowed("five seven three", "573")
    assert ok is True, err


def test_percent_place_value_allowed():
    ok, err = itn_edit_allowed("百分之五十", "50%")
    assert ok is True, err
