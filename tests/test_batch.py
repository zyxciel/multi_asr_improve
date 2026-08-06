from __future__ import annotations

import json
from pathlib import Path

from stage2_asr.batch import discover_benchmark_pairs, run_batch
from stage2_asr.cli import main as cli_main


def _touch(path: Path, text: str = "") -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")
    return path


def test_discover_benchmark_pairs_matches_dataset_and_stem(tmp_path: Path):
    wav_bench = tmp_path / "test_datasets" / "benchmark"
    mc_bench = tmp_path / "DiarizenMossFusion" / "benchmark"

    _touch(wav_bench / "ds_a" / "Audio" / "utt1.wav", "fake")
    _touch(wav_bench / "ds_a" / "Audio" / "utt2.wav", "fake")
    _touch(wav_bench / "ds_b" / "Audio" / "utt1.wav", "fake")
    _touch(
        mc_bench / "ds_a" / "Audio" / "utt1" / "mode_c.json",
        json.dumps({"turns": [{"start": 0, "end": 1, "speaker_id": "s0", "text": "hi"}]}),
    )
    _touch(
        mc_bench / "ds_a" / "Audio" / "utt2" / "mode_c.json",
        json.dumps({"turns": [{"start": 0, "end": 1, "speaker_id": "s0", "text": "hi"}]}),
    )
    # ds_b/utt1 missing mode_c → skip

    pairs, skips = discover_benchmark_pairs(wav_bench, mc_bench)
    assert {(p.dataset, p.stem) for p in pairs} == {("ds_a", "utt1"), ("ds_a", "utt2")}
    assert len(skips) == 1
    assert skips[0]["dataset"] == "ds_b"
    assert skips[0]["stem"] == "utt1"
    assert skips[0]["reason"] == "missing_mode_c"
    assert pairs[0].mode_c.name == "mode_c.json"
    assert pairs[0].wav.suffix == ".wav"


def test_discover_can_filter_datasets(tmp_path: Path):
    wav_bench = tmp_path / "benchmark"
    mc_bench = tmp_path / "mc_benchmark"
    _touch(wav_bench / "keep" / "Audio" / "a.wav", "x")
    _touch(wav_bench / "drop" / "Audio" / "a.wav", "x")
    _touch(mc_bench / "keep" / "Audio" / "a" / "mode_c.json", '{"turns":[]}')
    _touch(mc_bench / "drop" / "Audio" / "a" / "mode_c.json", '{"turns":[]}')

    pairs, skips = discover_benchmark_pairs(wav_bench, mc_bench, datasets=["keep"])
    assert len(pairs) == 1 and pairs[0].dataset == "keep"
    assert skips == []


def test_run_batch_mock_writes_per_sample_work_dirs(tmp_path: Path):
    wav_bench = tmp_path / "wav_benchmark"
    mc_bench = tmp_path / "mc_benchmark"
    work_root = tmp_path / "out"
    mode_c = {
        "meta": {"mode": "c"},
        "turns": [
            {
                "start": 0.0,
                "end": 1.0,
                "speaker_id": "s0",
                "text": "大家好",
                "asr_status": "provisional",
            }
        ],
    }
    _touch(wav_bench / "ds1" / "Audio" / "m1.wav", "x")
    _touch(mc_bench / "ds1" / "Audio" / "m1" / "mode_c.json", json.dumps(mode_c))

    summary = run_batch(
        wav_benchmark=wav_bench,
        mode_c_benchmark=mc_bench,
        work_root=work_root,
        backend="mock",
        stage="all",
        asr_models=["moss", "qwen"],
        hotwords=[],
        enable_real=False,
    )
    assert summary["n_ok"] == 1
    assert summary["n_skip"] == 0
    assert (work_root / "ds1" / "m1" / "mode_c_asr_final.json").exists()
    assert (work_root / "batch_summary.json").exists()


def test_cli_run_batch_dry_run(tmp_path: Path, capsys):
    wav_bench = tmp_path / "wav_benchmark"
    mc_bench = tmp_path / "mc_benchmark"
    work_root = tmp_path / "out"
    _touch(wav_bench / "ds1" / "Audio" / "m1.wav", "x")
    _touch(mc_bench / "ds1" / "Audio" / "m1" / "mode_c.json", '{"turns":[]}')

    code = cli_main(
        [
            "run-batch",
            "--wav-benchmark",
            str(wav_bench),
            "--mode-c-benchmark",
            str(mc_bench),
            "--work-root",
            str(work_root),
            "--dry-run",
            "--mock",
        ]
    )
    assert code == 0
    payload = json.loads(capsys.readouterr().out.strip())
    assert payload["n_paired"] == 1
    summary = json.loads((work_root / "batch_summary.json").read_text(encoding="utf-8"))
    assert summary["dry_run"] is True
    assert summary["results"][0]["status"] == "dry_run"
