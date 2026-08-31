from __future__ import annotations

import wave
from pathlib import Path

import numpy as np


def load_wav_mono16k(path: Path, target_sr: int = 16000) -> tuple[np.ndarray, int]:
    """Load wav as float32 mono. Resample not implemented — expects 16 kHz for Stage-1 prepared audio."""
    with wave.open(str(path), "rb") as wf:
        sr = wf.getframerate()
        n = wf.getnframes()
        ch = wf.getnchannels()
        raw = wf.readframes(n)
        width = wf.getsampwidth()
    if width == 2:
        audio = np.frombuffer(raw, dtype=np.int16).astype(np.float32) / 32768.0
    else:
        raise ValueError(f"unsupported sample width {width}")
    if ch > 1:
        audio = audio.reshape(-1, ch).mean(axis=1)
    if sr != target_sr:
        # Lightweight linear resample for tests / mismatched files
        duration = len(audio) / sr
        new_n = int(duration * target_sr)
        x_old = np.linspace(0.0, 1.0, num=len(audio), endpoint=False)
        x_new = np.linspace(0.0, 1.0, num=new_n, endpoint=False)
        audio = np.interp(x_new, x_old, audio).astype(np.float32)
        sr = target_sr
    return audio, sr


def write_wav_mono16k(path: Path, audio: np.ndarray, sr: int = 16000) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    clipped = np.clip(audio, -1.0, 1.0)
    pcm = (clipped * 32767.0).astype(np.int16)
    with wave.open(str(path), "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(sr)
        wf.writeframes(pcm.tobytes())


def crop_unit_wav(
    audio_path: Path,
    start: float,
    end: float,
    *,
    work_dir: Path,
    unit_id: str,
    sr: int = 16000,
    reuse_existing: bool = True,
) -> Path:
    """Crop [start, end] from prepared wav into work_dir/crops/{unit_id}.wav.

    When reuse_existing is True and the crop file already exists, return it
    without re-decoding the full meeting wav (important for staged ASR re-runs).
    """
    out = work_dir / "crops" / f"{unit_id}.wav"
    if reuse_existing and out.is_file() and out.stat().st_size > 0:
        return out
    audio, file_sr = load_wav_mono16k(audio_path, target_sr=sr)
    s = max(0, int(start * file_sr))
    e = min(len(audio), max(s + 1, int(end * file_sr)))
    crop = audio[s:e]
    write_wav_mono16k(out, crop, sr=file_sr)
    return out


def make_silent_wav(path: Path, duration_s: float = 1.0, sr: int = 16000) -> Path:
    audio = np.zeros(int(duration_s * sr), dtype=np.float32)
    write_wav_mono16k(path, audio, sr=sr)
    return path
