from __future__ import annotations

import json
from pathlib import Path

from stage2_asr.glossary_union import (
    surface_key,
    to_seed_glossary,
    union_corpus_glossary,
    write_corpus_glossary,
)


def _write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def _sample_dir(work_root: Path, rel: str) -> Path:
    d = work_root.joinpath(*Path(rel).parts)
    d.mkdir(parents=True, exist_ok=True)
    return d


def _write_sample(
    work_root: Path,
    rel: str,
    *,
    glossary: dict,
    turns: list[dict] | None = None,
    units: list[dict] | None = None,
    hyps: list[dict] | None = None,
    crop_unit: str | None = None,
) -> Path:
    sample = _sample_dir(work_root, rel)
    _write_json(sample / "glossary.json", glossary)
    if turns is not None:
        _write_json(sample / "mode_c_published.json", {"turns": turns})
    if units is not None:
        _write_json(sample / "asr_units.json", {"units": units})
    if hyps is not None:
        _write_json(sample / "asr_hypotheses.json", {"records": hyps})
    if crop_unit:
        crop = sample / "crops" / f"{crop_unit}.wav"
        crop.parent.mkdir(parents=True, exist_ok=True)
        crop.write_bytes(b"RIFF")
    return sample


def test_surface_key_folds_latin_and_keeps_cjk_distinct():
    assert surface_key(" GPU ") == surface_key("gpu")
    assert surface_key("张三风") != surface_key("张三丰")
    assert surface_key("Windows产品") == surface_key("Windows产品")


def test_union_merges_same_surface_across_samples_without_last_write_wins(tmp_path: Path):
    work = tmp_path / "work"
    _write_sample(
        work,
        "ds/a",
        glossary={
            "terms": [
                {
                    "surface": "GPU",
                    "aliases": ["显卡"],
                    "kind": "product",
                    "latex": None,
                    "source": "extract",
                }
            ],
            "keywords": [{"surface": "gpu", "score": 0.4}],
            "rare_words": [],
        },
        turns=[{"text": "用 GPU 训练", "start": 0.0, "end": 1.0, "speaker_id": "s0"}],
    )
    _write_sample(
        work,
        "ds/b",
        glossary={
            "terms": [
                {
                    "surface": "GPU",
                    "aliases": ["graphics"],
                    "kind": "product",
                    "latex": None,
                    "source": "extract",
                }
            ],
            "keywords": [{"surface": "GPU", "score": 0.9}],
            "rare_words": [],
        },
        turns=[{"text": "GPU 很贵", "start": 0.0, "end": 1.0, "speaker_id": "s0"}],
    )

    corpus = union_corpus_glossary(work)
    terms = {t["surface"]: t for t in corpus["terms"]}
    assert set(terms) == {"GPU"}
    assert set(terms["GPU"]["aliases"]) == {"显卡", "graphics"}
    assert terms["GPU"]["n_docs"] == 2
    assert set(terms["GPU"]["sample_ids"]) == {"ds/a", "ds/b"}

    kw = {k["surface"]: k for k in corpus["keywords"]}
    assert set(kw) == {"GPU"}
    assert kw["GPU"]["n_docs"] == 2
    assert kw["GPU"]["score_max"] == 0.9
    assert kw["GPU"]["score_mean"] == 0.65


def test_kind_majority_and_homophones_stay_apart(tmp_path: Path):
    work = tmp_path / "work"
    _write_sample(
        work,
        "x/1",
        glossary={
            "terms": [
                {"surface": "张三风", "aliases": [], "kind": "other"},
                {"surface": "alpha", "aliases": ["阿尔法"], "kind": "symbol", "latex": "\\alpha"},
            ],
            "keywords": [],
            "rare_words": [],
        },
    )
    _write_sample(
        work,
        "x/2",
        glossary={
            "terms": [
                {"surface": "张三丰", "aliases": [], "kind": "other"},
                {"surface": "alpha", "aliases": [], "kind": "formula", "latex": "\\alpha"},
            ],
            "keywords": [],
            "rare_words": [],
        },
    )
    _write_sample(
        work,
        "x/3",
        glossary={
            "terms": [{"surface": "alpha", "aliases": [], "kind": "symbol"}],
            "keywords": [],
            "rare_words": [],
        },
    )

    corpus = union_corpus_glossary(work)
    surfaces = {t["surface"] for t in corpus["terms"]}
    assert "张三风" in surfaces and "张三丰" in surfaces
    alpha = next(t for t in corpus["terms"] if t["surface"] == "alpha")
    assert alpha["kind"] == "symbol"
    assert alpha["latex"] == "\\alpha"


def test_rare_words_recount_mentions_and_attach_context_crop_hyps(tmp_path: Path):
    work = tmp_path / "work"
    _write_sample(
        work,
        "meet/utt1",
        glossary={
            "terms": [],
            "keywords": [],
            "rare_words": [{"surface": "玛曲", "count": 99}],
        },
        turns=[
            {
                "text": "明天去玛曲开会然后回来",
                "start": 10.0,
                "end": 14.0,
                "speaker_id": "s0",
            }
        ],
        units=[
            {
                "unit_id": "unit_0003",
                "start": 9.5,
                "end": 14.2,
                "speaker_id": "s0",
                "turn_indices": [0],
            }
        ],
        hyps=[
            {
                "unit_id": "unit_0003",
                "hyps": [
                    {"model": "qwen", "text": "明天去玛曲开会然后回来"},
                    {"model": "moss", "text": "明天去马区开会然后回来"},
                ],
            }
        ],
        crop_unit="unit_0003",
    )

    corpus = union_corpus_glossary(work, context_chars=4)
    rare = {r["surface"]: r for r in corpus["rare_words"]}
    assert "玛曲" in rare
    row = rare["玛曲"]
    assert row["n_docs"] == 1
    assert row["n_mentions"] == 1
    assert row.get("count") != 99
    occ = row["occurrences"]
    assert len(occ) == 1
    hit = occ[0]
    assert hit["sample_id"] == "meet/utt1"
    assert hit["turn_index"] == 0
    assert hit["start"] == 10.0
    assert hit["end"] == 14.0
    assert hit["speaker_id"] == "s0"
    assert hit["unit_id"] == "unit_0003"
    assert "玛曲" in hit["context"]
    assert hit["crop"].endswith("meet/utt1/crops/unit_0003.wav")
    models = {h["model"]: h["text"] for h in hit["hyps"]}
    assert models["moss"] == "明天去马区开会然后回来"


def test_rare_word_missing_from_published_keeps_surface_without_occurrences(tmp_path: Path):
    work = tmp_path / "work"
    _write_sample(
        work,
        "ghost",
        glossary={
            "terms": [],
            "keywords": [],
            "rare_words": [{"surface": "幻觉词"}],
        },
        turns=[{"text": "今天天气不错", "start": 0.0, "end": 1.0, "speaker_id": "s0"}],
    )
    corpus = union_corpus_glossary(work)
    row = corpus["rare_words"][0]
    assert row["surface"] == "幻觉词"
    assert row["n_docs"] == 1
    assert row["n_mentions"] == 0
    assert row["occurrences"] == []


def test_write_corpus_and_seed_and_cli(tmp_path: Path):
    work = tmp_path / "work"
    _write_sample(
        work,
        "s1",
        glossary={
            "terms": [{"surface": "Qwen", "aliases": ["千问"], "kind": "product"}],
            "keywords": [{"surface": "Qwen", "score": 1.0}],
            "rare_words": [{"surface": "玛曲"}],
        },
        turns=[{"text": "去玛曲用 Qwen", "start": 0.0, "end": 1.0, "speaker_id": "s0"}],
    )
    corpus = union_corpus_glossary(work)
    out = tmp_path / "corpus_glossary.json"
    paths = write_corpus_glossary(corpus, out)
    loaded = json.loads(out.read_text(encoding="utf-8"))
    assert loaded["terms"] and loaded["keywords"] and loaded["rare_words"]
    assert "occurrences" in loaded["rare_words"][0]
    seed = json.loads(paths["seed"].read_text(encoding="utf-8"))
    assert seed["terms"][0]["surface"] == "Qwen"
    assert "aliases" in seed["terms"][0]
    assert "n_docs" not in seed["terms"][0]
    assert "occurrences" not in seed["rare_words"][0]
    assert seed["rare_words"][0]["count"] == 1
    stripped = to_seed_glossary(corpus)
    assert stripped["keywords"][0]["score"] == 1.0

    from stage2_asr.cli import main

    cli_out = tmp_path / "from_cli.json"
    rc = main(["union-glossary", "--work-root", str(work), "--out", str(cli_out)])
    assert rc == 0
    assert cli_out.exists()
    assert Path(str(cli_out).replace(".json", ".seed.json")).exists() or (
        cli_out.with_name(cli_out.stem + ".seed.json").exists()
    )
