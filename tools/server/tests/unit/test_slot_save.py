import pytest
from utils import *
import base64
import json
import requests
import struct

# sequence state file: magic(4) version(4) payload_size(4), then payload_size llama_token words
STATE_FILE_HEADER_SIZE = 12

server = ServerPreset.tinyllama2()

@pytest.fixture(autouse=True)
def create_server(tmp_path):
    global server
    server = ServerPreset.tinyllama2()
    server.slot_save_path = str(tmp_path)
    server.temperature = 0.0


def test_slot_save_restore():
    global server
    server.start()

    # First prompt in slot 1 should be fully processed
    res = server.make_request("POST", "/completion", data={
        "prompt": "What is the capital of France?",
        "id_slot": 1,
        "cache_prompt": True,
    })
    assert res.status_code == 200
    assert match_regex("(Whiskers|Flana)+", res.body["content"])
    assert res.body["timings"]["prompt_n"] == 21  # all tokens are processed

    # Save state of slot 1
    res = server.make_request("POST", "/slots/1?action=save", data={
        "filename": "slot1.bin",
    })
    assert res.status_code == 200
    assert res.body["n_saved"] == 84
    slot_file = os.path.join(server.slot_save_path, "slot1.bin")
    assert res.body["n_written"] == os.path.getsize(slot_file)

    # Since we have cache, this should only process the last tokens
    res = server.make_request("POST", "/completion", data={
        "prompt": "What is the capital of Germany?",
        "id_slot": 1,
        "cache_prompt": True,
    })
    assert res.status_code == 200
    assert match_regex("(Jack|said)+", res.body["content"])
    assert res.body["timings"]["prompt_n"] == 6  # only different part is processed

    # Loading the saved cache into slot 0
    res = server.make_request("POST", "/slots/0?action=restore", data={
        "filename": "slot1.bin",
    })
    assert res.status_code == 200
    assert res.body["n_restored"] == 84
    assert res.body["n_read"] == os.path.getsize(slot_file)

    # Since we have cache, slot 0 should only process the last tokens
    res = server.make_request("POST", "/completion", data={
        "prompt": "What is the capital of Germany?",
        "id_slot": 0,
        "cache_prompt": True,
    })
    assert res.status_code == 200
    assert match_regex("(Jack|said)+", res.body["content"])
    assert res.body["timings"]["prompt_n"] == 6  # only different part is processed

    # For verification that slot 1 was not corrupted during slot 0 load, same thing should work
    res = server.make_request("POST", "/completion", data={
        "prompt": "What is the capital of Germany?",
        "id_slot": 1,
        "cache_prompt": True,
    })
    assert res.status_code == 200
    assert match_regex("(Jack|said)+", res.body["content"])
    assert res.body["timings"]["prompt_n"] == 1


def test_slot_restore_legacy_token_list():
    global server
    server.start()

    res = server.make_request("POST", "/completion", data={
        "prompt": "What is the capital of France?",
        "id_slot": 1,
        "cache_prompt": True,
    })
    assert res.status_code == 200

    res = server.make_request("POST", "/slots/1?action=save", data={
        "filename": "slot_legacy.bin",
    })
    assert res.status_code == 200
    assert res.body["n_saved"] == 84

    # rewrite the token payload into a plain token list, as written by servers that predate the packed server_tokens format
    path = os.path.join(server.slot_save_path, "slot_legacy.bin")
    with open(path, "rb") as f:
        data = bytearray(f.read())

    # the payload written by this server starts with a packed header: LLAMA_TOKEN_NULL(4) version(4) n_tokens(4)
    packed_header_size = 12

    payload_size = struct.unpack_from("=I", data, STATE_FILE_HEADER_SIZE - 4)[0]
    payload_end = STATE_FILE_HEADER_SIZE + payload_size * 4
    n_tokens = struct.unpack_from("=I", data, STATE_FILE_HEADER_SIZE + 8)[0]
    assert n_tokens == 84

    tokens_start = STATE_FILE_HEADER_SIZE + packed_header_size
    data = data[:STATE_FILE_HEADER_SIZE] + data[tokens_start:tokens_start + n_tokens * 4] + data[payload_end:]
    struct.pack_into("=I", data, STATE_FILE_HEADER_SIZE - 4, n_tokens)

    with open(path, "wb") as f:
        f.write(data)

    # the plain token list must restore, and the restored KV must be reusable
    res = server.make_request("POST", "/slots/0?action=restore", data={
        "filename": "slot_legacy.bin",
    })
    assert res.status_code == 200
    assert res.body["n_restored"] == 84

    res = server.make_request("POST", "/completion", data={
        "prompt": "What is the capital of Germany?",
        "id_slot": 0,
        "cache_prompt": True,
    })
    assert res.status_code == 200
    assert res.body["timings"]["prompt_n"] == 6  # only the different part is processed



def test_slot_erase():
    global server
    server.start()

    res = server.make_request("POST", "/completion", data={
        "prompt": "What is the capital of France?",
        "id_slot": 1,
        "cache_prompt": True,
    })
    assert res.status_code == 200
    assert match_regex("(Whiskers|Flana)+", res.body["content"])
    assert res.body["timings"]["prompt_n"] == 21  # all tokens are processed

    # erase slot 1
    res = server.make_request("POST", "/slots/1?action=erase")
    assert res.status_code == 200

    # re-run the same prompt, it should process all tokens again
    res = server.make_request("POST", "/completion", data={
        "prompt": "What is the capital of France?",
        "id_slot": 1,
        "cache_prompt": True,
    })
    assert res.status_code == 200
    assert match_regex("(Whiskers|Flana)+", res.body["content"])
    assert res.body["timings"]["prompt_n"] == 21  # all tokens are processed


#
# Multimodal server (mmproj loaded) slot save/restore.
#
# A pure-text slot on a multimodal server and a slot containing images must both support save/restore.
# Erase remains gated on the slot's content.
#

IMG_URL_CAT = "https://huggingface.co/ggml-org/tinygemma3-GGUF/resolve/main/test/91_cat.png"
IMG_URL_TRUCK = "https://huggingface.co/ggml-org/tinygemma3-GGUF/resolve/main/test/11_truck.png"


def _get_img_base64(url: str) -> str:
    response = requests.get(url)
    response.raise_for_status()  # Raise an exception for bad status codes
    return base64.b64encode(response.content).decode("utf-8")


@pytest.fixture
def mmproj_server():
    # tinygemma3 is a small multimodal model: the mmproj is provided by the HF registry API and auto-downloaded on first run.
    os.environ['LLAMA_MEDIA_MARKER'] = '<__media__>'
    mm_server = ServerPreset.tinygemma3()
    mm_server.slot_save_path = "./tmp"
    mm_server.temperature = 0.0
    return mm_server


def test_slot_save_restore_text_only_on_multimodal(mmproj_server):
    server = mmproj_server
    server.start()

    # A pure-text prompt processed on slot 1 of a multimodal server.
    res = server.make_request("POST", "/completion", data={
        "prompt": "The quick brown fox jumps over the lazy dog.",
        "id_slot": 1,
        "cache_prompt": True,
    })
    assert res.status_code == 200
    prompt_n = res.body["timings"]["prompt_n"]
    assert prompt_n > 0  # all tokens are processed

    # Saving a pure-text slot must succeed even though an mmproj is loaded.
    res = server.make_request("POST", "/slots/1?action=save", data={
        "filename": "mm_slot1.bin",
    })
    assert res.status_code == 200
    n_saved = res.body["n_saved"]
    assert n_saved > 0  # the slot KV (prompt + generated tokens) was written

    # Restore the saved state into slot 0; it must round-trip exactly.
    res = server.make_request("POST", "/slots/0?action=restore", data={
        "filename": "mm_slot1.bin",
    })
    assert res.status_code == 200
    assert res.body["n_restored"] == n_saved

    # Prefix reuse is not checked with the default SWA cache.
    res = server.make_request("POST", "/completion", data={
        "prompt": "The quick brown fox jumps over the lazy dog.",
        "id_slot": 0,
        "cache_prompt": True,
    })
    assert res.status_code == 200


def test_slot_save_restore_with_image(mmproj_server):
    server = mmproj_server
    # Use the full SWA cache so the restored image prefix can be reused.
    server.swa_full = True
    server.start()

    prompt_cat = {
        "prompt_string": "What is this: <__media__>\n",
        "multimodal_data": [_get_img_base64(IMG_URL_CAT)],
    }
    res = server.make_request("POST", "/completions", data={
        "temperature": 0.0,
        "top_k": 1,
        "id_slot": 1,
        "cache_prompt": True,
        "prompt": prompt_cat,
    })
    assert res.status_code == 200
    content_cat = res.body["content"]
    prompt_n_full = res.body["timings"]["prompt_n"]
    assert res.body["timings"]["cache_n"] == 0
    assert prompt_n_full > 32  # text plus image tokens are all processed

    res = server.make_request("POST", "/slots/1?action=save", data={
        "filename": "mm_slot_image.bin",
    })
    assert res.status_code == 200
    n_saved = res.body["n_saved"]
    n_written = res.body["n_written"]
    assert n_saved > 0
    assert n_written > 0

    res = server.make_request("POST", "/slots/1?action=erase")
    assert res.status_code == 200

    res = server.make_request("POST", "/slots/0?action=restore", data={
        "filename": "mm_slot_image.bin",
    })
    assert res.status_code == 200
    assert res.body["n_restored"] == n_saved
    assert res.body["n_read"] == n_written

    # a different image must not reuse the restored image tokens; only the text prefix before the image is common
    res = server.make_request("POST", "/completions", data={
        "temperature": 0.0,
        "top_k": 1,
        "id_slot": 0,
        "cache_prompt": True,
        "prompt": {
            "prompt_string": "What is this: <__media__>\n",
            "multimodal_data": [_get_img_base64(IMG_URL_TRUCK)],
        },
    })
    assert res.status_code == 200
    cache_n = res.body["timings"]["cache_n"]
    assert cache_n < 16
    assert res.body["timings"]["prompt_n"] == prompt_n_full - cache_n

    # restore again and resend the same image: the image tokens must be reused and greedy sampling must reproduce the original content
    res = server.make_request("POST", "/slots/0?action=restore", data={
        "filename": "mm_slot_image.bin",
    })
    assert res.status_code == 200
    assert res.body["n_restored"] == n_saved

    res = server.make_request("POST", "/completions", data={
        "temperature": 0.0,
        "top_k": 1,
        "id_slot": 0,
        "cache_prompt": True,
        "prompt": prompt_cat,
    })
    assert res.status_code == 200
    assert res.body["timings"]["cache_n"] == prompt_n_full - 1
    assert res.body["timings"]["prompt_n"] == 1
    assert res.body["content"] == content_cat


def test_slot_save_restore_with_two_images(mmproj_server):
    server = mmproj_server
    server.swa_full = True
    server.n_ctx = 2048  # two images need more than the default 512 per slot
    server.start()

    prompt = {
        "prompt_string": "A: <__media__> B: <__media__>\n",
        "multimodal_data": [_get_img_base64(IMG_URL_CAT), _get_img_base64(IMG_URL_TRUCK)],
    }
    res = server.make_request("POST", "/completions", data={
        "temperature": 0.0,
        "top_k": 1,
        "id_slot": 1,
        "cache_prompt": True,
        "prompt": prompt,
    })
    assert res.status_code == 200
    prompt_n_full = res.body["timings"]["prompt_n"]
    assert prompt_n_full > 64

    res = server.make_request("POST", "/slots/1?action=save", data={
        "filename": "mm_slot_two_images.bin",
    })
    assert res.status_code == 200
    n_saved = res.body["n_saved"]

    res = server.make_request("POST", "/slots/0?action=restore", data={
        "filename": "mm_slot_two_images.bin",
    })
    assert res.status_code == 200
    assert res.body["n_restored"] == n_saved

    res = server.make_request("POST", "/completions", data={
        "temperature": 0.0,
        "top_k": 1,
        "id_slot": 0,
        "cache_prompt": True,
        "prompt": prompt,
    })
    assert res.status_code == 200
    assert res.body["timings"]["cache_n"] == prompt_n_full - 1
    assert res.body["timings"]["prompt_n"] == 1
    content = res.body["content"]

    res = server.make_request("POST", "/slots/1?action=restore", data={
        "filename": "mm_slot_two_images.bin",
    })
    assert res.status_code == 200
    assert res.body["n_restored"] == n_saved

    res = server.make_request("POST", "/completions", data={
        "temperature": 0.0,
        "top_k": 1,
        "id_slot": 0,
        "cache_prompt": True,
        "prompt": prompt,
    })
    assert res.status_code == 200
    assert res.body["timings"]["cache_n"] == prompt_n_full - 1
    assert res.body["timings"]["prompt_n"] == 1
    content = res.body["content"]

    assert res.body["content"] == content


def test_slot_save_restore_with_image_across_restart(mmproj_server):
    server = mmproj_server
    server.swa_full = True
    server.start()

    prompt_cat = {
        "prompt_string": "What is this: <__media__>\n",
        "multimodal_data": [_get_img_base64(IMG_URL_CAT)],
    }
    res = server.make_request("POST", "/completions", data={
        "temperature": 0.0,
        "top_k": 1,
        "id_slot": 0,
        "cache_prompt": True,
        "prompt": prompt_cat,
    })
    assert res.status_code == 200
    content = res.body["content"]
    prompt_n_full = res.body["timings"]["prompt_n"]

    res = server.make_request("POST", "/slots/0?action=save", data={
        "filename": "mm_slot_restart.bin",
    })
    assert res.status_code == 200
    n_saved = res.body["n_saved"]

    # restart the server with the same model and mmproj: the saved file must restore in the new process and the image KV must be reused
    server.stop()
    server.start()

    res = server.make_request("POST", "/slots/0?action=restore", data={
        "filename": "mm_slot_restart.bin",
    })
    assert res.status_code == 200
    assert res.body["n_restored"] == n_saved

    res = server.make_request("POST", "/completions", data={
        "temperature": 0.0,
        "top_k": 1,
        "id_slot": 0,
        "cache_prompt": True,
        "prompt": prompt_cat,
    })
    assert res.status_code == 200
    assert res.body["timings"]["cache_n"] == prompt_n_full - 1
    assert res.body["timings"]["prompt_n"] == 1
    assert res.body["content"] == content


def test_slot_save_restore_image_payload_larger_than_context(mmproj_server):
    server = mmproj_server
    server.swa_full = True
    server.start()

    # the slot context, as the server computed it (n_ctx split across the slots)
    res = server.make_request("GET", "/props")
    assert res.status_code == 200
    n_ctx_slot = res.body["default_generation_settings"]["n_ctx"]

    # a filler token, used to grow the prompt up to the slot context
    res = server.make_request("POST", "/tokenize", data={"content": " hello" * 8})
    assert res.status_code == 200
    assert len(res.body["tokens"]) == 8

    res = server.make_request("POST", "/completions", data={
        "temperature": 0.0,
        "top_k": 1,
        "id_slot": 0,
        "cache_prompt": True,
        "prompt": {
            "prompt_string": "What is this: <__media__>\n",
            "multimodal_data": [_get_img_base64(IMG_URL_CAT)],
        },
    })
    assert res.status_code == 200

    prompt_cat = {
        "prompt_string": "What is this: <__media__>\n" + " hello" * (n_ctx_slot - res.body["timings"]["prompt_n"] - 8),
        "multimodal_data": [_get_img_base64(IMG_URL_CAT)],
    }
    res = server.make_request("POST", "/completions", data={
        "temperature": 0.0,
        "top_k": 1,
        "id_slot": 0,
        "cache_prompt": True,
        "prompt": prompt_cat,
    })
    assert res.status_code == 200
    prompt_n_full = res.body["timings"]["cache_n"] + res.body["timings"]["prompt_n"]

    res = server.make_request("POST", "/slots/0?action=save", data={
        "filename": "mm_slot_large_payload.bin",
    })
    assert res.status_code == 200

    path = os.path.join(server.slot_save_path, "mm_slot_large_payload.bin")
    with open(path, "rb") as f:
        data = bytearray(f.read())
    payload_size = struct.unpack_from("=I", data, STATE_FILE_HEADER_SIZE - 4)[0]
    assert payload_size > n_ctx_slot  # the scenario under test: the payload does not fit in n_ctx

    # drop the image from the slot, then restore it from the file
    res = server.make_request("POST", "/completion", data={
        "prompt": "The quick brown fox",
        "id_slot": 0,
        "cache_prompt": True,
    })
    assert res.status_code == 200

    res = server.make_request("POST", "/slots/0?action=restore", data={
        "filename": "mm_slot_large_payload.bin",
    })
    assert res.status_code == 200

    res = server.make_request("POST", "/completions", data={
        "temperature": 0.0,
        "top_k": 1,
        "id_slot": 0,
        "cache_prompt": True,
        "prompt": prompt_cat,
    })
    assert res.status_code == 200
    assert res.body["timings"]["cache_n"] == prompt_n_full - 1
    assert res.body["timings"]["prompt_n"] == 1


def test_slot_restore_media_file_without_mmproj(mmproj_server):
    server = mmproj_server
    server.start()

    res = server.make_request("POST", "/completions", data={
        "temperature": 0.0,
        "top_k": 1,
        "id_slot": 0,
        "cache_prompt": True,
        "prompt": {
            "prompt_string": "What is this: <__media__>\n",
            "multimodal_data": [_get_img_base64(IMG_URL_CAT)],
        },
    })
    assert res.status_code == 200

    res = server.make_request("POST", "/slots/0?action=save", data={
        "filename": "mm_slot_no_mmproj.bin",
    })
    assert res.status_code == 200

    # restart the same model without the mmproj: restoring the media file must fail gracefully and leave the slot usable
    server.stop()
    server.no_mmproj = True
    server.start()

    res = server.make_request("POST", "/slots/0?action=restore", data={
        "filename": "mm_slot_no_mmproj.bin",
    })
    assert res.status_code == 400
    assert "Cannot restore media tokens without an mmproj" in res.body["error"]["message"]

    # A failed restore must leave the slot empty and usable.
    res = server.make_request("POST", "/completions", data={
        "temperature": 0.0,
        "top_k": 1,
        "id_slot": 1,
        "cache_prompt": True,
        "prompt": "The quick brown fox",
    })
    assert res.status_code == 200
    content = res.body["content"]

    res = server.make_request("POST", "/completions", data={
        "temperature": 0.0,
        "top_k": 1,
        "id_slot": 0,
        "cache_prompt": True,
        "prompt": "The quick brown fox",
    })
    assert res.status_code == 200
    assert res.body["timings"]["cache_n"] == 0
    assert res.body["content"] == content


@pytest.mark.slow
@pytest.mark.parametrize("automatic", [False, True])
def test_recurrent_slot_disk_cache(tmp_path, monkeypatch, automatic):
    from pathlib import Path

    model = os.environ.get("SLOT_SAVE_HTTP_MODEL")
    if not model:
        pytest.skip("SLOT_SAVE_HTTP_MODEL must point to a local FULL/recurrent GGUF")
    monkeypatch.setenv("LLAMA_ARG_SLOT_SAVE_AUTO", "1" if automatic else "0")
    srv = ServerProcess()
    srv.model_file = model
    srv.model_hf_repo = srv.model_hf_file = None
    srv.n_ctx = 4096
    srv.n_slots = 1
    srv.n_batch = srv.n_ubatch = 256
    srv.n_predict = 1
    srv.temperature = 0.0
    srv.fit = "off"
    srv.fa = "on"
    srv.cache_ram = 0
    srv.no_cache_idle_slots = True
    srv.slot_save_path = str(tmp_path)
    srv.log_path = str(tmp_path / "server.log.tmp")
    srv.server_metrics = True
    srv.server_slots = True

    def complete(tokens, **extra):
        response = srv.make_request("POST", "/completion", {
            "prompt": tokens, "n_predict": 1, "cache_prompt": True, "temperature": 0, **extra,
        })
        assert response.status_code == 200, response.body
        return response.body

    def restart():
        srv.stop()
        srv.start(timeout_seconds=180)

    def restore(path):
        if not automatic:
            response = srv.make_request("POST", "/slots/0?action=restore", {"filename": path.name})
            assert response.status_code == 200, response.body

    def prompt_metrics():
        text = requests.get(f"http://{srv.server_host}:{srv.server_port}/metrics", timeout=10).text
        values = {}
        for name in ("prompt_tokens_total", "prompt_tokens_cached_total", "prompt_seconds_total"):
            match = re.search(rf"^llamacpp:{name} ([0-9.eE+-]+)$", text, re.MULTILINE)
            assert match, text
            values[name] = float(match.group(1))
        return values

    srv.start(timeout_seconds=180)
    response = srv.make_request("POST", "/tokenize", {
        "content": "The reference code is BLUE. " * 100, "add_special": True,
    })
    tokens = response.body["tokens"]
    assert len(tokens) > 256
    first = complete(tokens)
    if automatic:
        # The metrics task is processed after the synchronous save on the slot thread.
        prompt_metrics()
        paths = list(tmp_path.glob("auto-*.bin"))
        assert len(paths) == 1
        path = paths[0]
    else:
        path = tmp_path / "manual.bin"
        response = srv.make_request("POST", "/slots/0?action=save", {"filename": path.name})
        assert response.status_code == 200, response.body
        assert response.body["n_saved"] == len(tokens)
    original_state = path.read_bytes()
    sidecar = Path(str(path) + ".logits")
    if automatic:
        manifest_length, _, magic = struct.unpack("<QQ8s", original_state[-24:])
        assert magic == b"LLROUTE1"
        assert 0 < manifest_length <= len(original_state) - 24
        manifest_start = len(original_state) - 24 - manifest_length
        manifest = json.loads(original_state[manifest_start:-24])
        assert manifest["n_tokens"] == manifest["logits_tokens"] == len(tokens)
        assert manifest["logits_offset"] == manifest["state_bytes"]
        assert manifest["logits_bytes"] == manifest["logits_vocab"] * 4 > 0
        assert manifest["logits_offset"] + manifest["logits_bytes"] == manifest_start
        original_logits = None
    else:
        assert sidecar.is_file()
        original_logits = sidecar.read_bytes()
        assert struct.unpack_from("<I", original_logits, 12)[0] == len(tokens)

    def restore_fixture():
        # Other automatic snapshots from the suffix test must not mask corruption.
        if automatic:
            for other in tmp_path.glob("auto-*.bin"):
                if other != path:
                    other.unlink()
        path.write_bytes(original_state)
        if original_logits is not None:
            sidecar.write_bytes(original_logits)

    def assert_prompt_metrics(response, expected_total):
        cached = response["timings"]["cache_n"]
        processed = response["timings"]["prompt_n"]
        assert cached + processed == expected_total, response
        values = prompt_metrics()
        assert values["prompt_tokens_cached_total"] == cached
        assert values["prompt_tokens_total"] == processed

    restart()
    restore(path)
    replay = complete(tokens)
    assert replay["timings"]["prompt_n"] == 0, replay
    assert replay["content"] == first["content"]
    metrics = requests.get(f"http://{srv.server_host}:{srv.server_port}/metrics", timeout=10).text
    predicted = re.search(r"^llamacpp:tokens_predicted_total ([0-9.]+)$", metrics, re.MULTILINE)
    assert predicted and float(predicted.group(1)) == 1, metrics
    values = prompt_metrics()
    assert values["prompt_tokens_cached_total"] == replay["timings"]["cache_n"] == len(tokens)
    assert values["prompt_tokens_total"] == 0
    assert values["prompt_seconds_total"] == 0
    print(f"disk-cache automatic={automatic}: cold prompt_n={first['timings']['prompt_n']}, replay prompt_n=0, content={replay['content']!r}")

    restart()
    restore(path)
    suffix = srv.make_request("POST", "/tokenize", {"content": " Continue: BLUE"}).body["tokens"]
    extended = complete(tokens + suffix)
    assert 0 < extended["timings"]["prompt_n"] <= len(suffix) + 1, extended
    values = prompt_metrics()
    assert values["prompt_tokens_cached_total"] == extended["timings"]["cache_n"]
    assert values["prompt_tokens_total"] == extended["timings"]["prompt_n"]
    assert values["prompt_tokens_cached_total"] + values["prompt_tokens_total"] == len(tokens + suffix)
    print(f"disk-cache automatic={automatic}: suffix prompt_n={extended['timings']['prompt_n']}")

    # Corrupt logits must not be used; a valid checkpoint may still be reused.
    srv.stop()
    restore_fixture()
    if automatic:
        altered = bytearray(original_state)
        altered[manifest["logits_offset"]] ^= 1
        path.write_bytes(altered)
    else:
        sidecar.write_bytes(original_logits[:8])
    srv.start(timeout_seconds=180)
    restore(path)
    fallback = complete(tokens)
    assert 0 < fallback["timings"]["prompt_n"] <= len(tokens), fallback
    assert_prompt_metrics(fallback, len(tokens))
    assert fallback["content"] == first["content"]
    print(f"disk-cache automatic={automatic}: corrupt-logits fallback prompt_n={fallback['timings']['prompt_n']}")

    if automatic:
        # A mismatched first native token cannot match the requested prefix.
        srv.stop()
        restore_fixture()
        altered = bytearray(original_state)
        struct.pack_into("=i", altered, STATE_FILE_HEADER_SIZE, tokens[0] ^ 1)
        path.write_bytes(altered)
        srv.start(timeout_seconds=180)
        fallback = complete(tokens)
        assert fallback["timings"]["prompt_n"] == len(tokens), fallback
        assert fallback["timings"]["cache_n"] == 0, fallback
        assert_prompt_metrics(fallback, len(tokens))
        assert fallback["content"] == first["content"]
        print(f"disk-cache automatic=True: mismatched-state fallback prompt_n={fallback['timings']['prompt_n']}")

        srv.stop()
        restore_fixture()
        altered = bytearray(original_state)
        # The footer owns the manifest length; reject an out-of-bounds length.
        struct.pack_into("<Q", altered, len(altered) - 24, 0x7fffffffffffffff)
        path.write_bytes(altered)
        srv.start(timeout_seconds=180)
        fallback = complete(tokens)
        assert fallback["timings"]["prompt_n"] == len(tokens), fallback
        assert fallback["timings"]["cache_n"] == 0, fallback
        assert_prompt_metrics(fallback, len(tokens))
        assert fallback["content"] == first["content"]
        print(f"disk-cache automatic=True: oversized-metadata fallback prompt_n={fallback['timings']['prompt_n']}")

        srv.stop()
        restore_fixture()
        altered = bytearray(original_state)
        version = b'"version":1'
        version_pos = original_state.find(version, manifest_start, len(original_state) - 24)
        assert version_pos >= 0
        altered[version_pos + len(version) - 1] = ord("2")
        assert json.loads(altered[manifest_start:-24])["version"] == 2
        # Preserve the footer and its original checksum while corrupting the manifest.
        path.write_bytes(altered)
        srv.start(timeout_seconds=180)
        fallback = complete(tokens)
        assert fallback["timings"]["prompt_n"] == len(tokens), fallback
        assert fallback["timings"]["cache_n"] == 0, fallback
        assert_prompt_metrics(fallback, len(tokens))
        assert fallback["content"] == first["content"]
        print(f"disk-cache automatic=True: corrupt-manifest fallback prompt_n={fallback['timings']['prompt_n']}")
    srv.stop()


@pytest.fixture
def swa_server(tmp_path):
    swa = ServerPreset.tinygemma3()
    swa.slot_save_path = str(tmp_path)
    swa.temperature = 0.0
    swa.cache_ram = 0
    # Keep the first prompt checkpoint before the divergence point.
    swa.n_ubatch = 32
    return swa


# the non-ASCII name checks that the appendix lands in the same file as the llama state on Windows
@pytest.mark.parametrize("filename", ["ckpt_slot1.bin", "ckpt_slot1_é.bin"])
def test_slot_restore_preserves_context_checkpoints(swa_server, filename):
    server = swa_server
    server.start()

    base = "The quick brown fox jumps over the lazy dog. " * 20

    res = server.make_request("POST", "/completion", data={
        "prompt": base + "The first ending of this story is a happy one.",
        "id_slot": 1,
        "cache_prompt": True,
    })
    assert res.status_code == 200
    n_full = res.body["timings"]["prompt_n"]

    res = server.make_request("POST", "/completion", data={
        "prompt": base + "But the second ending was different and sad.",
        "id_slot": 1,
        "cache_prompt": True,
    })
    assert res.status_code == 200
    n_live = res.body["timings"]["prompt_n"]
    assert n_live < n_full

    res = server.make_request("POST", "/slots/1?action=erase")
    assert res.status_code == 200

    res = server.make_request("POST", "/completion", data={
        "prompt": base + "The first ending of this story is a happy one.",
        "id_slot": 1,
        "cache_prompt": True,
    })
    assert res.status_code == 200

    res = server.make_request("POST", "/slots/1?action=save", data={
        "filename": filename,
    })
    assert res.status_code == 200
    assert res.body["n_saved"] > 0
    ckpt_file = os.path.join(server.slot_save_path, filename)
    assert res.body["n_written"] == os.path.getsize(ckpt_file)

    res = server.make_request("POST", "/completion", data={
        "prompt": "Unrelated text with no common prefix occupies the slot now.",
        "id_slot": 1,
        "cache_prompt": True,
    })
    assert res.status_code == 200

    res = server.make_request("POST", "/slots/1?action=restore", data={
        "filename": filename,
    })
    assert res.status_code == 200
    assert res.body["n_read"] == os.path.getsize(ckpt_file)

    res = server.make_request("POST", "/completion", data={
        "prompt": base + "But the second ending was different and sad.",
        "id_slot": 1,
        "cache_prompt": True,
    })
    assert res.status_code == 200
    assert res.body["timings"]["prompt_n"] == n_live


# checkpoint appendix: magic(4) version(4) count(4), then per checkpoint
# n_tokens(8) pos_min(4) pos_max(4) and three blobs (target, draft, speculative), each size(8) + data
def parse_ckpt_appendix(data):
    off = data.find(struct.pack("<II", 0x504b4353, 1))
    assert off > 0
    count = struct.unpack_from("<I", data, off + 8)[0]
    ckpts = []
    pos = off + 12
    for _ in range(count):
        start = pos
        pos += 16
        blobs = []
        for _ in range(3):
            n = struct.unpack_from("<Q", data, pos)[0]
            blobs.append(pos + 8)
            pos += 8 + n
        ckpts.append((start, pos, blobs[0]))
    assert pos == len(data)
    return off, ckpts


# a damaged appendix must be ignored, or its checkpoints dropped when they fail to load, without aborting the server
@pytest.mark.parametrize("damage", ["oversized_blob", "empty_target", "corrupt_state", "many_checkpoints"])
def test_slot_restore_damaged_checkpoint_appendix(swa_server, damage):
    server = swa_server
    server.start()

    base = "The quick brown fox jumps over the lazy dog. " * 20

    res = server.make_request("POST", "/completion", data={
        "prompt": base + "The first ending of this story is a happy one.",
        "id_slot": 1,
        "cache_prompt": True,
    })
    assert res.status_code == 200

    res = server.make_request("POST", "/completion", data={
        "prompt": base + "But the second ending was different and sad.",
        "id_slot": 1,
        "cache_prompt": True,
    })
    assert res.status_code == 200
    n_live = res.body["timings"]["prompt_n"]

    res = server.make_request("POST", "/slots/1?action=erase")
    assert res.status_code == 200

    res = server.make_request("POST", "/completion", data={
        "prompt": base + "The first ending of this story is a happy one.",
        "id_slot": 1,
        "cache_prompt": True,
    })
    assert res.status_code == 200

    res = server.make_request("POST", "/slots/1?action=save", data={
        "filename": "ckpt_damaged.bin",
    })
    assert res.status_code == 200

    path = os.path.join(server.slot_save_path, "ckpt_damaged.bin")
    with open(path, "rb") as f:
        data = bytearray(f.read())
    off, ckpts = parse_ckpt_appendix(data)

    if damage == "oversized_blob":
        # the first target blob declares a size that cannot be allocated, it must be rejected before allocating
        data = data[:ckpts[0][0] + 16] + struct.pack("<Q", 1 << 62)
    elif damage == "empty_target":
        # the target blobs are removed and their size set to 0, a valid save never writes an empty target state
        for start, end, tgt in reversed(ckpts):
            size = struct.unpack_from("<Q", data, tgt - 8)[0]
            data = data[:tgt - 8] + struct.pack("<Q", 0) + data[tgt + size:]
    elif damage == "corrupt_state":
        # the sizes are intact, but the target states do not load
        for _, _, tgt in ckpts:
            struct.pack_into("<I", data, tgt, 0xdeadbeef)
    else:
        # more than 1024 entries: one-byte fillers that never match go first, the real checkpoints stay last
        filler = struct.pack("<qiiQBQQ", 0, 0, 1 << 30, 1, 0, 0, 0)
        data = data[:off + 12] + filler * (1025 - len(ckpts)) + data[off + 12:]
        struct.pack_into("<I", data, off + 8, 1025)

    with open(path, "wb") as f:
        f.write(data)

    res = server.make_request("POST", "/slots/1?action=restore", data={
        "filename": "ckpt_damaged.bin",
    })
    assert res.status_code == 200
    if damage in ("oversized_blob", "empty_target"):
        assert res.body["n_read"] == off

    res = server.make_request("POST", "/completion", data={
        "prompt": base + "But the second ending was different and sad.",
        "id_slot": 1,
        "cache_prompt": True,
    })
    assert res.status_code == 200
    if damage == "many_checkpoints":
        assert res.body["timings"]["prompt_n"] == n_live
    else:
        assert res.body["timings"]["prompt_n"] > n_live


# the draft blobs of the checkpoint appendix are not covered by the main payload checks,
# so restoring into a server with another draft KV cache type must not abort
@pytest.mark.parametrize("ctkd_restore", ["f16", "q8_0"])
def test_slot_restore_checkpoints_draft_kv_type_change(swa_server, ctkd_restore):
    server = swa_server
    server.model_draft_hf_repo = "ggml-org/tinygemma3-GGUF:Q8_0"  # same file as the target, already in the HF cache
    server.spec_type = "draft-simple"
    server.ctkd = "f16"
    server.start()

    base = "The quick brown fox jumps over the lazy dog. " * 20

    res = server.make_request("POST", "/completion", data={
        "prompt": base + "The first ending of this story is a happy one.",
        "id_slot": 1,
        "cache_prompt": True,
    })
    assert res.status_code == 200

    res = server.make_request("POST", "/completion", data={
        "prompt": base + "But the second ending was different and sad.",
        "id_slot": 1,
        "cache_prompt": True,
    })
    assert res.status_code == 200
    n_live = res.body["timings"]["prompt_n"]

    res = server.make_request("POST", "/slots/1?action=erase")
    assert res.status_code == 200

    res = server.make_request("POST", "/completion", data={
        "prompt": base + "The first ending of this story is a happy one.",
        "id_slot": 1,
        "cache_prompt": True,
    })
    assert res.status_code == 200

    res = server.make_request("POST", "/slots/1?action=save", data={
        "filename": "ckpt_draft_slot1.bin",
    })
    assert res.status_code == 200

    server.stop()
    server.ctkd = ctkd_restore
    server.start()

    res = server.make_request("POST", "/slots/1?action=restore", data={
        "filename": "ckpt_draft_slot1.bin",
    })
    assert res.status_code == 200

    res = server.make_request("POST", "/completion", data={
        "prompt": base + "But the second ending was different and sad.",
        "id_slot": 1,
        "cache_prompt": True,
    })
    assert res.status_code == 200
    assert res.body["timings"]["prompt_n"] == n_live


def test_native_stream_is_explicit_and_does_not_auto_save(tmp_path):
    """The explicit native marker must not silently change the legacy path.

    This CPU server has no qualified native peer, so the explicit option is
    expected to fail closed. The default save/restore remains available and
    slot auto-save stays off.
    """
    global server
    server = ServerPreset.tinyllama2()
    server.slot_save_path = str(tmp_path)
    server.temperature = 0.0
    server.start()

    res = server.make_request("POST", "/completion", data={
        "prompt": "The quick brown fox jumps over the lazy dog.",
        "id_slot": 1,
        "cache_prompt": True,
    })
    assert res.status_code == 200

    before = sorted(os.listdir(tmp_path))
    res = server.make_request("POST", "/slots/1?action=save", data={
        "filename": "native.slot",
        "native_stream": True,
    })
    assert res.status_code != 200
    assert "native streaming" in res.body["error"]["message"]
    assert sorted(os.listdir(tmp_path)) == before
    assert not os.path.exists(os.path.join(tmp_path, "native.slot"))

    # Without the marker the legacy API is unchanged.
    res = server.make_request("POST", "/slots/1?action=save", data={
        "filename": "legacy.slot",
    })
    assert res.status_code == 200
    assert os.path.getsize(os.path.join(tmp_path, "legacy.slot")) == res.body["n_written"]

    # The explicit restore marker also fails before mutating an empty slot.
    res = server.make_request("POST", "/slots/0?action=restore", data={
        "filename": "legacy.slot",
        "native_stream": True,
    })
    assert res.status_code != 200
    assert "native streaming" in res.body["error"]["message"]

    # A preflight rejection must not clear an occupied slot. Slot 1 still has
    # the first prompt, so repeating it reuses the cached prefix.
    res = server.make_request("POST", "/slots/1?action=restore", data={
        "filename": "legacy.slot",
        "native_stream": True,
    })
    assert res.status_code != 200
    res = server.make_request("POST", "/completion", data={
        "prompt": "The quick brown fox jumps over the lazy dog.",
        "id_slot": 1,
        "cache_prompt": True,
    })
    assert res.status_code == 200
    assert res.body["timings"]["prompt_n"] == 1

    # Auto-save is off: the only new state file is the explicit legacy save.
    names = set(os.listdir(tmp_path))
    assert "native.slot" not in names
    assert "legacy.slot" in names
