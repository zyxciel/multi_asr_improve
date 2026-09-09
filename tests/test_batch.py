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
    assert {p.rel for p in pairs} == {"ds_a/Audio/utt1", "ds_a/Audio/utt2"}
    assert len(skips) == 1
    assert skips[0]["dataset"] == "ds_b"
    assert skips[0]["stem"] == "utt1"
    assert skips[0]["reason"] == "missing_mode_c"
    assert pairs[0].mode_c.name == "mode_c.json"
    assert pairs[0].wav.suffix == ".wav"
    assert pairs[0].work_dir(tmp_path / "out") == tmp_path / "out" / "ds_a" / "Audio" / "utt1"


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


def test_discover_recurses_nested_dirs_without_audio_folder(tmp_path: Path):
    wav_root = tmp_path / "wavs"
    mc_root = tmp_path / "mode_c"
    _touch(wav_root / "proj" / "day1" / "meet_a.wav", "x")
    _touch(wav_root / "proj" / "day1" / "nested" / "meet_b.wav", "x")
    _touch(wav_root / "other" / "skip_me.wav", "x")
    _touch(mc_root / "proj" / "day1" / "meet_a" / "mode_c.json", '{"turns":[]}')
    _touch(mc_root / "proj" / "day1" / "nested" / "meet_b" / "mode_c.json", '{"turns":[]}')

    pairs, skips = discover_benchmark_pairs(wav_root, mc_root)
    assert {(p.rel, p.stem) for p in pairs} == {
        ("proj/day1/meet_a", "meet_a"),
        ("proj/day1/nested/meet_b", "meet_b"),
    }
    assert len(skips) == 1
    assert skips[0]["stem"] == "skip_me"
    assert skips[0]["reason"] == "missing_mode_c"
    work_root = tmp_path / "out"
    by_stem = {p.stem: p for p in pairs}
    assert by_stem["meet_b"].work_dir(work_root) == work_root / "proj" / "day1" / "nested" / "meet_b"


def test_discover_pairs_when_audio_folder_only_on_one_side(tmp_path: Path):
    wav_root = tmp_path / "wavs"
    mc_root = tmp_path / "mode_c"
    _touch(wav_root / "ds_a" / "Audio" / "utt1.wav", "x")
    _touch(mc_root / "ds_a" / "utt1" / "mode_c.json", '{"turns":[]}')

    pairs, skips = discover_benchmark_pairs(wav_root, mc_root)
    assert len(pairs) == 1 and not skips
    assert pairs[0].stem == "utt1"
    assert pairs[0].work_dir(tmp_path / "out") == tmp_path / "out" / "ds_a" / "Audio" / "utt1"


def test_discover_pairs_single_files_even_if_names_differ(tmp_path: Path):
    wav = _touch(tmp_path / "clip.wav", "x")
    mode_c = _touch(tmp_path / "fusion.json", '{"turns":[]}')
    pairs, skips = discover_benchmark_pairs(wav, mode_c)
    assert skips == []
    assert len(pairs) == 1
    assert pairs[0].wav == wav.resolve()
    assert pairs[0].mode_c == mode_c.resolve()
    assert pairs[0].stem == "clip"
    assert pairs[0].rel == "clip"


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
    assert (work_root / "ds1" / "Audio" / "m1" / "mode_c_asr_final.json").exists()
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


def _one_sample_trees(tmp_path: Path, *, n: int = 1, mode_c: dict | None = None):
    wav_bench = tmp_path / "wav_benchmark"
    mc_bench = tmp_path / "mc_benchmark"
    work_root = tmp_path / "out"
    doc = mode_c or {
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
    for i in range(n):
        stem = f"m{i + 1}"
        _touch(wav_bench / "ds1" / "Audio" / f"{stem}.wav", "x")
        _touch(mc_bench / "ds1" / "Audio" / stem / "mode_c.json", json.dumps(doc))
    return wav_bench, mc_bench, work_root


def test_stage_complete_asr_requires_all_requested_models(tmp_path: Path):
    from stage2_asr.batch import stage_complete

    work = tmp_path / "ds1" / "Audio" / "m1"
    work.mkdir(parents=True)
    (work / "asr_hypotheses.json").write_text(
        json.dumps(
            {
                "meta": {"asr_models": ["qwen"]},
                "records": [
                    {"unit_id": "u0", "hyps": [{"model": "qwen", "text": "hi"}]}
                ],
            }
        ),
        encoding="utf-8",
    )
    assert stage_complete(work, "asr", ["qwen"]) is True
    assert stage_complete(work, "asr", ["qwen", "firered"]) is False


def test_run_batch_skips_existing_llm_outputs(tmp_path: Path):
    wav_bench, mc_bench, work_root = _one_sample_trees(tmp_path)
    sample = work_root / "ds1" / "Audio" / "m1"
    sample.mkdir(parents=True)
    sentinel = {"sentinel": True, "turns": []}
    (sample / "mode_c_asr_final.json").write_text(json.dumps(sentinel), encoding="utf-8")
    (sample / "mode_c_polished.json").write_text(json.dumps(sentinel), encoding="utf-8")
    (sample / "mode_c_published.json").write_text(json.dumps(sentinel), encoding="utf-8")

    summary = run_batch(
        wav_benchmark=wav_bench,
        mode_c_benchmark=mc_bench,
        work_root=work_root,
        backend="mock",
        stage="llm",
        skip_existing=True,
    )
    assert summary["n_ok"] == 0
    assert summary["n_cached"] == 1
    assert summary["results"][0]["status"] == "skipped_existing"
    published = json.loads((sample / "mode_c_published.json").read_text(encoding="utf-8"))
    assert published["sentinel"] is True


def test_run_batch_no_skip_existing_reruns(tmp_path: Path):
    wav_bench, mc_bench, work_root = _one_sample_trees(tmp_path)
    sample = work_root / "ds1" / "Audio" / "m1"
    sample.mkdir(parents=True)
    (sample / "mode_c_published.json").write_text(
        json.dumps({"sentinel": True, "turns": []}), encoding="utf-8"
    )

    summary = run_batch(
        wav_benchmark=wav_bench,
        mode_c_benchmark=mc_bench,
        work_root=work_root,
        backend="mock",
        stage="all",
        skip_existing=False,
    )
    assert summary["n_ok"] == 1
    published = json.loads((sample / "mode_c_published.json").read_text(encoding="utf-8"))
    assert "sentinel" not in published
    assert "turns" in published


def test_run_batch_sample_workers_completes_all(tmp_path: Path):
    wav_bench, mc_bench, work_root = _one_sample_trees(tmp_path, n=2)
    summary = run_batch(
        wav_benchmark=wav_bench,
        mode_c_benchmark=mc_bench,
        work_root=work_root,
        backend="mock",
        stage="all",
        sample_workers=2,
    )
    assert summary["n_ok"] == 2
    assert (work_root / "ds1" / "Audio" / "m1" / "mode_c_published.json").exists()
    assert (work_root / "ds1" / "Audio" / "m2" / "mode_c_published.json").exists()


def test_split_even_covers_all_items_without_overlap():
    from stage2_asr.batch import split_even

    parts = split_even(list(range(6000)), 4)
    assert [len(p) for p in parts] == [1500, 1500, 1500, 1500]
    assert [x for part in parts for x in part] == list(range(6000))

    uneven = split_even(list(range(7)), 4)
    assert [len(p) for p in uneven] == [2, 2, 2, 1]
    assert [x for part in uneven for x in part] == list(range(7))


def test_plan_npu_jobs_groups_devices_by_two():
    from stage2_asr.batch import plan_npu_jobs

    jobs = plan_npu_jobs([0, 1, 2, 3, 4, 5, 6, 7], npu_per_job=2)
    assert [(j["shard_index"], j["devices"]) for j in jobs] == [
        (0, [0, 1]),
        (1, [2, 3]),
        (2, [4, 5]),
        (3, [6, 7]),
    ]
    assert jobs[0]["n_shards"] == 4


def test_run_batch_shard_is_disjoint(tmp_path: Path):
    wav_bench, mc_bench, work_root = _one_sample_trees(tmp_path, n=4)
    a = run_batch(
        wav_benchmark=wav_bench,
        mode_c_benchmark=mc_bench,
        work_root=work_root,
        backend="mock",
        stage="all",
        shard="0/2",
        skip_existing=False,
    )
    b = run_batch(
        wav_benchmark=wav_bench,
        mode_c_benchmark=mc_bench,
        work_root=work_root,
        backend="mock",
        stage="all",
        shard="1/2",
        skip_existing=False,
    )
    assert a["n_paired"] == 2 and b["n_paired"] == 2
    ids_a = {r["sample_id"] for r in a["results"]}
    ids_b = {r["sample_id"] for r in b["results"]}
    assert ids_a.isdisjoint(ids_b)
    assert ids_a | ids_b == {"ds1/Audio/m1", "ds1/Audio/m2", "ds1/Audio/m3", "ds1/Audio/m4"}
    assert (work_root / "batch_summary.shard0.json").exists()
    assert (work_root / "batch_summary.shard1.json").exists()


def test_cli_npu_parallel_dry_run_waits_and_merges(tmp_path: Path, capsys):
    wav_bench, mc_bench, work_root = _one_sample_trees(tmp_path, n=4)
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
            "--devices",
            "0,1,2,3",
            "--npu-per-job",
            "2",
        ]
    )
    assert code == 0
    payload = json.loads(capsys.readouterr().out.strip().splitlines()[-1])
    assert payload["n_paired"] == 4
    assert payload["n_shards"] == 2
    merged = json.loads((work_root / "batch_summary.json").read_text(encoding="utf-8"))
    assert merged["n_paired"] == 4
    assert len(merged["results"]) == 4
    assert {r["status"] for r in merged["results"]} == {"dry_run"}
