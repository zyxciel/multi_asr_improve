# Stage-2 Multi-ASR + LLM Fusion

Architecture design for improving ASR accuracy on top of **diarizen_moss_fusion** (DiariZen + MOSS Mode C) outputs.

This repo currently holds the **design doc for teammate discussion**. Implementation comes after the design is agreed.

## Docs

- [Stage-2 Multi-ASR + LLM Fusion Design](docs/2026-07-24-stage2-multi-asr-llm-fusion-design.md)

## One-line summary

Consume fused diarization turns → same-speaker concat/split for ASR → Qwen3-ASR + FireRed (+ MOSS text) → skip LLM if CER=0 → else Pass A select/repair (Tier A–C) with MOSS-primary on overlap → required Pass B global consistency → `asr_status=final`.
