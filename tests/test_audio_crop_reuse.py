from __future__ import annotations

from pathlib import Path
from unittest.mock import patch

import numpy as np

from stage2_asr.audio_io import crop_unit_wav, load_wav_mono16k, write_wav_mono16k


def test_crop_reuses_existing_file_without_reload(tmp_path: Path):
    wav = tmp_path / "full.wav"
    write_wav_mono16k(wav, np.zeros(16000, dtype=np.float32), sr=16000)
    work = tmp_path / "work"

    first = crop_unit_wav(wav, 0.0, 0.5, work_dir=work, unit_id="u0", sr=16000)
    assert first.exists()
    mtime1 = first.stat().st_mtime_ns

    with patch("stage2_asr.audio_io.load_wav_mono16k") as mocked_load:
        second = crop_unit_wav(wav, 0.0, 0.5, work_dir=work, unit_id="u0", sr=16000)
        mocked_load.assert_not_called()
    assert second == first
    assert second.stat().st_mtime_ns == mtime1


def test_crop_can_force_rewrite(tmp_path: Path):
    wav = tmp_path / "full.wav"
    write_wav_mono16k(wav, np.ones(16000, dtype=np.float32) * 0.1, sr=16000)
    work = tmp_path / "work"
    first = crop_unit_wav(wav, 0.0, 0.5, work_dir=work, unit_id="u1", sr=16000)
    audio1, _ = load_wav_mono16k(first)
    write_wav_mono16k(wav, np.ones(16000, dtype=np.float32) * 0.5, sr=16000)
    second = crop_unit_wav(
        wav, 0.0, 0.5, work_dir=work, unit_id="u1", sr=16000, reuse_existing=False
    )
    audio2, _ = load_wav_mono16k(second)
    assert second == first
    assert float(np.abs(audio2).mean()) > float(np.abs(audio1).mean())
