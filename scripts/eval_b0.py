#!/usr/bin/env python3
"""CLI for Eval B0: MOSS-from-fusion baseline CER / cpCER."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from stage2_asr.eval_b0 import evaluate_b0  # noqa: E402


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Eval B0: MOSS-from-fusion CER / cpCER against a reference transcript."
    )
    parser.add_argument(
        "--hyp",
        type=Path,
        required=True,
        help="Mode-C JSON (or mode_c_draft / final) whose turn texts are the hypothesis",
    )
    parser.add_argument(
        "--ref",
        type=Path,
        required=True,
        help="Reference JSON with turns[].text (same ordering as hyp)",
    )
    parser.add_argument(
        "--json-out",
        type=Path,
        default=None,
        help="Optional path to write full per-turn metrics JSON",
    )
    args = parser.parse_args(argv)

    result = evaluate_b0(hyp_path=args.hyp, ref_path=args.ref)
    summary = {k: v for k, v in result.items() if k != "per_turn"}
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    if args.json_out is not None:
        args.json_out.parent.mkdir(parents=True, exist_ok=True)
        args.json_out.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
