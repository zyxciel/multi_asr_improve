# Stage-2 Multi-ASR + LLM Fusion

Architecture for improving ASR accuracy on top of **diarizen_moss_fusion** (DiariZen + MOSS Mode C) outputs.

## Docs

- [Stage-2 Multi-ASR + LLM Fusion Design (final)](docs/2026-07-24-stage2-multi-asr-llm-fusion-design.md)

## Summary

Consume fused diarization turns → same-speaker concat/split → **dynamic overlap gate** (MOSS-exclusive if overlap_ratio &gt; 30%, else MOSS-primary) → Qwen3-ASR + FireRed (+ MOSS) → skip LLM if CER=0 → Pass A Tier A–C with **span-local char-count** → Pass B → `asr_status=final`.

**Eval B0:** MOSS-from-fusion.
