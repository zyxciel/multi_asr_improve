"""Resolve Qwen3-ASR / FireRed weight paths for debug vs production machines."""

from __future__ import annotations

import os

DEFAULT_QWEN_MODEL_ID = "Qwen/Qwen3-ASR-1.7B"
DEFAULT_FIRERED_ASR_MODEL_DIR = "pretrained_models/FireRedASR2-AED"
DEFAULT_FIRERED_LID_MODEL_DIR = "pretrained_models/FireRedLID"
DEFAULT_FIRERED_PUNC_MODEL_DIR = "pretrained_models/FireRedPunc"


def _first_nonempty(*values: str | None) -> str | None:
    for value in values:
        if value is None:
            continue
        text = str(value).strip()
        if text:
            return text
    return None


def resolve_qwen_model_id(cli_value: str | None = None) -> str:
    """CLI, then STAGE2_QWEN_MODEL_ID, then QWEN_MODEL_ID, then HuggingFace id."""
    return (
        _first_nonempty(
            cli_value,
            os.environ.get("STAGE2_QWEN_MODEL_ID"),
            os.environ.get("QWEN_MODEL_ID"),
            DEFAULT_QWEN_MODEL_ID,
        )
        or DEFAULT_QWEN_MODEL_ID
    )


def resolve_firered_model_dirs(
    *,
    asr_model_dir: str | None = None,
    lid_model_dir: str | None = None,
    punc_model_dir: str | None = None,
) -> tuple[str, str, str]:
    """CLI, then STAGE2_FIRERED_* / FIRERED_*, then relative pretrained_models defaults."""
    asr = (
        _first_nonempty(
            asr_model_dir,
            os.environ.get("STAGE2_FIRERED_ASR_MODEL_DIR"),
            os.environ.get("FIRERED_ASR_MODEL_DIR"),
            DEFAULT_FIRERED_ASR_MODEL_DIR,
        )
        or DEFAULT_FIRERED_ASR_MODEL_DIR
    )
    lid = (
        _first_nonempty(
            lid_model_dir,
            os.environ.get("STAGE2_FIRERED_LID_MODEL_DIR"),
            os.environ.get("FIRERED_LID_MODEL_DIR"),
            DEFAULT_FIRERED_LID_MODEL_DIR,
        )
        or DEFAULT_FIRERED_LID_MODEL_DIR
    )
    punc = (
        _first_nonempty(
            punc_model_dir,
            os.environ.get("STAGE2_FIRERED_PUNC_MODEL_DIR"),
            os.environ.get("FIRERED_PUNC_MODEL_DIR"),
            DEFAULT_FIRERED_PUNC_MODEL_DIR,
        )
        or DEFAULT_FIRERED_PUNC_MODEL_DIR
    )
    return asr, lid, punc
