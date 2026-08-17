from __future__ import annotations

import json
import threading
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path

from stage2_asr.runners.llm_qwen36 import Qwen36LlmJudge
from stage2_asr.runners.openai_compat import chat_completion, normalize_openai_base_url
from stage2_asr.types import Hypothesis


def test_normalize_openai_base_url():
    assert normalize_openai_base_url("http://127.0.0.1:8000") == "http://127.0.0.1:8000/v1"
    assert normalize_openai_base_url("http://127.0.0.1:8000/v1") == "http://127.0.0.1:8000/v1"


def test_chat_completion_openai_compat():
    class Handler(BaseHTTPRequestHandler):
        def do_POST(self):  # noqa: N802
            length = int(self.headers.get("Content-Length", "0"))
            body = json.loads(self.rfile.read(length).decode("utf-8"))
            assert body["model"] == "Qwen/Qwen3.8-27B"
            if "chat_template_kwargs" in body:
                assert body["chat_template_kwargs"].get("enable_thinking") is False
            assert body["messages"][0]["role"] == "system"
            payload = {
                "choices": [
                    {
                        "message": {
                            "role": "assistant",
                            "content": json.dumps(
                                {
                                    "text": "采用",
                                    "base_model": "qwen",
                                    "edits": [],
                                    "overlap": False,
                                },
                                ensure_ascii=False,
                            ),
                        }
                    }
                ]
            }
            data = json.dumps(payload).encode("utf-8")
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(data)))
            self.end_headers()
            self.wfile.write(data)

        def log_message(self, format, *args):  # noqa: A003
            return

    server = HTTPServer(("127.0.0.1", 0), Handler)
    port = server.server_address[1]
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        text = chat_completion(
            base_url=f"http://127.0.0.1:{port}",
            model="Qwen/Qwen3.8-27B",
            messages=[{"role": "system", "content": "s"}, {"role": "user", "content": "u"}],
            temperature=0.1,
        )
        assert "采用" in text
        logs: list[dict] = []
        judge = Qwen36LlmJudge(
            enabled=True,
            backend="vllm",
            base_url=f"http://127.0.0.1:{port}",
            model_id="Qwen/Qwen3.8-27B",
            log_fn=logs.append,
        )
        out = judge.judge(
            hypotheses=[Hypothesis("qwen", "产用")],
            neighbor_draft=[],
            hotwords=[],
            overlap=False,
            heavy_overlap=False,
            unit_id="u0",
        )
        assert out["text"] == "采用"
        assert logs and logs[-1]["ok"] is True
        assert logs[-1]["backend"] == "vllm"
    finally:
        server.shutdown()


def test_pipeline_writes_llm_infer_log(tmp_path: Path):
    from stage2_asr.pipeline import run_pipeline
    from stage2_asr.runners.mock_asr import MockAsrRunner
    from stage2_asr.runners.mock_llm import MockLlmJudge
    from stage2_asr.types import PipelineConfig

    # Attachable mock that records via Qwen-like log_fn support
    class LoggingMock(MockLlmJudge):
        def __init__(self):
            super().__init__()
            self.log_fn = None

        def judge(self, **kwargs):
            if self.log_fn:
                self.log_fn(
                    {
                        "judge": "mock",
                        "backend": "mock",
                        "pass": "judge",
                        "unit_id": kwargs.get("unit_id"),
                        "ok": True,
                        "response": "{}",
                    }
                )
            return super().judge(**kwargs)

    fixtures = Path(__file__).parent / "fixtures"
    out = tmp_path / "work"
    result = run_pipeline(
        input_json=fixtures / "mode_c.json",
        audio_path=tmp_path / "missing.wav",
        work_dir=out,
        asr_runner=MockAsrRunner(),
        llm_judge=LoggingMock(),
        config=PipelineConfig(),
        stage="all",
    )
    log_path = Path(result["llm_log_path"])
    assert log_path.exists()
    lines = [json.loads(x) for x in log_path.read_text(encoding="utf-8").splitlines() if x.strip()]
    assert lines
    assert any(r.get("unit_id") for r in lines)
