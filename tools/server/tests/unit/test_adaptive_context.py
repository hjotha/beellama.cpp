"""HTTP coverage for the opt-in resident-weight adaptive context profile."""

import os
from pathlib import Path
from urllib.parse import quote

import pytest
from openai import OpenAI

from utils import ServerProcess


pytestmark = [pytest.mark.adaptive, pytest.mark.slow]


def _server(tmp_path: Path) -> ServerProcess:
    model = os.environ.get("ADAPTIVE_HTTP_MODEL")
    if not model:
        pytest.fail("ADAPTIVE_HTTP_MODEL must point to the local Qwen MTP GGUF")
    model_path = Path(model).expanduser()
    if not model_path.is_file():
        pytest.fail(f"adaptive model does not exist: {model_path}")

    server = ServerProcess()
    server.model_file = str(model_path)
    server.model_hf_repo = None
    server.model_hf_file = None
    server.n_gpu_layer = int(os.environ.get("ADAPTIVE_HTTP_GPU_LAYERS", "0"))
    server.n_ctx = 512
    server.ctx_size_mtp = 256
    server.mtp_max_tokens = 256
    server.fit = "off"
    server.spec_type = "draft-mtp"
    server.spec_draft_n_max = 2
    server.spec_draft_n_min = 0
    server.n_slots = 1
    server.n_batch = 32
    server.n_ubatch = 32
    server.n_predict = 2
    server.temperature = 0.0
    server.cache_ram = 2048
    server.server_slots = True
    server.jinja = True
    server.slot_save_path = str(tmp_path)
    server.log_path = os.environ.get("ADAPTIVE_HTTP_SERVER_LOG")
    return server


def _status(body, profile: str, context_size: int) -> None:
    status = body["adaptive_context"]
    assert status["enabled"] is True, status
    assert status["profile"] == profile, status
    assert status["state"] == "ready", status
    assert status["context_size"] == context_size, status
    assert status["context_size_long"] == 512, status
    if os.environ.get("ADAPTIVE_HTTP_EXPECT_GPU") == "1":
        assert status["mtp_weights_resident"] is (profile == "mtp"), status


def test_adaptive_http_contract(tmp_path):
    server = _server(tmp_path)
    server.start(timeout_seconds=int(os.environ.get("ADAPTIVE_HTTP_START_TIMEOUT", "900")))
    try:
        props = server.make_request("GET", "/props")
        models = server.make_request("GET", "/models")
        slots = server.make_request("GET", "/slots")
        assert props.status_code == models.status_code == slots.status_code == 200
        _status(props.body, "mtp", 256)
        _status(models.body["data"][0], "mtp", 256)
        _status(slots.body[0], "mtp", 256)

        short_prompt = "Explain resident MTP weights in one sentence."
        short = server.make_request("POST", "/completion", data={
            "prompt": short_prompt,
            "id_slot": 0,
            "cache_prompt": True,
            "n_predict": 1,
        })
        assert short.status_code == 200, short.body

        saved = server.make_request("POST", "/slots/0?action=save", data={"filename": "short.bin"})
        assert saved.status_code == 200 and saved.body["n_saved"] > 0, saved.body

        long_prompt = ("adaptive transition preserves the complete formatted prompt state. " * 50).strip()
        long = server.make_request("POST", "/completion", data={
            "prompt": long_prompt,
            "id_slot": 0,
            "cache_prompt": True,
            "n_predict": 1,
        })
        assert long.status_code == 200, long.body
        props_long = server.make_request("GET", "/props")
        assert props_long.status_code == 200
        _status(props_long.body, "long", 512)

        conversation_id = "adaptive-context::qwen35"
        stream = list(server.make_stream_request("POST", "/completion", data={
            "prompt": "stream after long profile",
            "id_slot": 0,
            "cache_prompt": True,
            "n_predict": 1,
            "stream": True,
        }, headers={"X-Conversation-Id": conversation_id}))
        assert stream, "adaptive SSE returned no events"
        replay = server.make_request(
            "GET", f"/v1/stream?conv_id={quote(conversation_id, safe='')}&from=0"
        )
        assert replay.status_code == 200 and "data: " in str(replay.body), replay.body

        client = OpenAI(api_key="dummy", base_url=f"http://{server.server_host}:{server.server_port}/v1")
        chat = client.chat.completions.create(
            model="qwen35-adaptive",
            messages=[{"role": "user", "content": "Say hello."}],
            max_tokens=1,
            temperature=0,
        )
        assert chat.choices and chat.choices[0].message is not None

        response = client.responses.create(
            model="qwen35-adaptive",
            input="Say hello.",
            max_output_tokens=1,
            temperature=0,
        )
        assert response.output

        tool = {
            "type": "function",
            "function": {
                "name": "noop",
                "description": "Return no result.",
                "parameters": {"type": "object", "properties": {}},
            },
        }
        tool_result = server.make_request("POST", "/v1/chat/completions", data={
            "model": "qwen35-adaptive",
            "messages": [{"role": "user", "content": "Say hello."}],
            "tools": [tool],
            "tool_choice": "none",
            "max_tokens": 1,
            "temperature": 0,
        })
        assert tool_result.status_code == 200, tool_result.body

        restored = server.make_request("POST", "/slots/0?action=restore", data={"filename": "short.bin"})
        assert restored.status_code == 200 and restored.body["n_restored"] > 0, restored.body
        props_short = server.make_request("GET", "/props")
        assert props_short.status_code == 200
        _status(props_short.body, "mtp", 256)
        reused = server.make_request("POST", "/completion", data={
            "prompt": short_prompt,
            "id_slot": 0,
            "cache_prompt": True,
            "n_predict": 1,
        })
        assert reused.status_code == 200, reused.body
        assert reused.body["timings"]["cache_n"] > 0, reused.body
    finally:
        server.stop()
