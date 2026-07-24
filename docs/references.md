# Upstream references

Pinned open-source components for Stage-2 adapters.

| Role | URL | Adapter |
| ---- | --- | ------- |
| Stage-1 fusion input | https://github.com/zyxciel/diarizen-moss-fusion | Consume `mode_c.json` |
| Qwen3-ASR | https://github.com/QwenLM/Qwen3-ASR | `stage2_asr/runners/qwen3_asr.py` → `qwen_asr.Qwen3ASRModel.transcribe` |
| FireRedASR2S | https://github.com/FireRedTeam/FireRedASR2S | `stage2_asr/runners/firered_asr2s.py` → `FireRedAsr2System` |
| Judge LLM | https://huggingface.co/Qwen/Qwen3.6-27B | `stage2_asr/runners/llm_qwen36.py` |
| Dialect eval (future) | https://github.com/ASLP-lab/WenetSpeech-Chuan | Not wired |

Local clones for API reference (gitignored): `third_party/Qwen3-ASR`, `third_party/FireRedASR2S`.

## FireRedASR2S policy

```python
FireRedAsr2SystemConfig(enable_vad=False, enable_lid=True, enable_punc=True)
```

- **VAD off** — unit crops from Stage-1/Stage-2 only (`vad_segments = [(0, dur)]` in upstream when disabled)
- **LID on** — `lid` on hyp
- **Punc on** — punctuated text before agreement / Pass A

## Qwen3-ASR

```python
from qwen_asr import Qwen3ASRModel
model = Qwen3ASRModel.from_pretrained("Qwen/Qwen3-ASR-1.7B", ...)
results = model.transcribe(audio=path, language="Chinese")
```

## Mock-first / real

```bash
# offline
stage2-asr run --input mode_c.json --audio prepared.wav --work-dir out --mock

# real (requires extras + local/remote weights; opt-in)
stage2-asr run --input mode_c.json --audio prepared.wav --work-dir out --backend real --enable-real
```

Real runners raise `UnsupportedRunnerError` until `--enable-real` and packages/weights are available. Unit tests inject fake model/system objects — **no downloads**.
