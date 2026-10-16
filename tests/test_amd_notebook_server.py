import ast
import copy
import http.client
import json
import socket
import sys
import threading
from contextlib import nullcontext
from pathlib import Path
from types import SimpleNamespace

import pytest

from scripts import amd_notebook_server as adapter

KEY = "test-only-key-" + "a" * 32
MESSAGES = [
    {"role": "system", "content": "Return JSON."},
    {"role": "user", "content": "Private patient text."},
]


class CPUEngine:
    """Controlled transport engine: only lives in tests, never production."""

    context_tokens = 8192

    def __init__(self):
        self.calls = []
        self.entered = threading.Event()
        self.release = threading.Event()
        self.finished = threading.Event()
        self.block = False
        self.fail = False

    def _work(self, messages):
        self.calls.append(messages)
        self.entered.set()
        try:
            if self.block:
                assert self.release.wait(5)
            if self.fail:
                raise RuntimeError("PRIVATE exception secret /notebook/secret-path")
        finally:
            self.finished.set()

    def count_tokens(self, messages):
        self._work(messages)
        return 23

    def generate(self, messages, max_tokens):
        self._work(messages)
        return {
            "model": adapter.MODEL_ID,
            "choices": [{
                "message": {"role": "assistant", "content": '{"ok":true}'},
                "finish_reason": "stop",
            }],
            "usage": {"prompt_tokens": 23, "completion_tokens": 5},
            "inference_seconds": 0.125,
        }


@pytest.fixture
def server():
    engine = CPUEngine()
    handle = adapter._start_http_server(engine, api_key=KEY, port=0)
    yield handle, engine
    engine.release.set()
    handle.stop()


def payload(completion=False):
    body = {"model": adapter.MODEL_ID, "messages": copy.deepcopy(MESSAGES)}
    if completion:
        body.update(max_tokens=50, temperature=0)
    return body


def request(handle, method="GET", path="/health", body=None, auth=KEY, headers=None):
    connection = http.client.HTTPConnection("127.0.0.1", handle.port, timeout=2)
    request_headers = {"Content-Type": "application/json"}
    if auth is not None:
        request_headers["Authorization"] = "Bearer " + auth
    request_headers.update(headers or {})
    if isinstance(body, (dict, list)):
        body = json.dumps(body)
    try:
        connection.request(method, path, body=body, headers=request_headers)
        response = connection.getresponse()
        data = response.read()
        return response.status, json.loads(data) if data else None, dict(response.getheaders())
    finally:
        connection.close()


def assert_error(response, status, code):
    assert response[0] == status
    assert set(response[1]) == {"error"}
    assert set(response[1]["error"]) == {"code", "message"}
    assert response[1]["error"]["code"] == code
    assert response[1]["error"]["message"]


@pytest.mark.parametrize("method,path", [
    ("GET", "/health"), ("POST", "/tokenize"), ("POST", "/v1/chat/completions"),
    ("GET", "/private-path"), ("OPTIONS", "/health"), ("CUSTOM", "/health"),
    ("PUT", "/tokenize"), ("DELETE", "/v1/chat/completions"),
])
@pytest.mark.parametrize("auth", [None, "wrong-key", "é" * 32])
def test_auth_on_every_route_and_method(server, method, path, auth):
    handle, engine = server
    assert_error(request(handle, method, path, auth=auth), 401, "unauthorized")
    assert not engine.calls


def test_health_and_loopback(server):
    handle, engine = server
    assert handle._server.server_address[0] == "127.0.0.1"
    status, body, headers = request(handle)
    assert status == 200
    assert body == {
        "status": "ok", "backend": "transformers", "device": "amd_rocm",
        "model": adapter.MODEL_ID, "revision": adapter.REVISION, "context_tokens": 8192,
    }
    assert headers["Cache-Control"] == "no-store"
    assert not engine.calls


def test_tokenize_and_completion_shapes(server):
    handle, engine = server
    assert request(handle, "POST", "/tokenize", payload())[:2] == (200, {"count": 23})
    status, body, _ = request(handle, "POST", "/v1/chat/completions", payload(True))
    assert status == 200
    assert body == {
        "model": adapter.MODEL_ID,
        "choices": [{"message": {"role": "assistant", "content": '{"ok":true}'},
                     "finish_reason": "stop"}],
        "usage": {"prompt_tokens": 23, "completion_tokens": 5},
        "inference_seconds": 0.125,
    }
    assert engine.calls == [MESSAGES, MESSAGES]


@pytest.mark.parametrize("field,value", [
    ("model", "another-model"), ("model", None), ("messages", []),
    ("messages", MESSAGES[::-1]), ("messages", MESSAGES + MESSAGES),
    ("messages", [{"role": "system", "content": ""}, MESSAGES[1]]),
    ("messages", [{"role": "system", "content": "   "}, MESSAGES[1]]),
    ("messages", [{"role": "system", "content": 42}, MESSAGES[1]]),
    ("messages", [{"role": "system", "content": "x", "name": "extra"}, MESSAGES[1]]),
    ("messages", [{"role": "system", "content": "x" * 65_537}, MESSAGES[1]]),
    ("messages", [None, MESSAGES[1]]), ("messages", "not a list"),
    ("max_tokens", 0), ("max_tokens", 4097), ("max_tokens", True),
    ("max_tokens", 1.5), ("max_tokens", "20"), ("temperature", 0.1),
    ("temperature", False), ("temperature", "0"), ("stream", False),
    ("tools", []), ("response_format", {"type": "json_object"}),
])
def test_request_constraints(server, field, value):
    handle, engine = server
    body = payload(True)
    body[field] = value
    code = "model_mismatch" if field == "model" else "invalid_request"
    assert_error(request(handle, "POST", "/v1/chat/completions", body), 400, code)
    assert not engine.calls


@pytest.mark.parametrize("field", ["model", "messages", "max_tokens", "temperature"])
def test_required_fields(server, field):
    body = payload(True)
    del body[field]
    assert_error(request(server[0], "POST", "/v1/chat/completions", body), 400, "invalid_request")


@pytest.mark.parametrize("body", [
    "not-json PRIVATE", "[]", "null", '{"model":1,"model":2}', '{"temperature":NaN}',
    b"\xff", "[" * 2000 + "]" * 2000,
])
def test_invalid_json(server, body):
    assert_error(request(server[0], "POST", "/tokenize", body), 400, "invalid_request")


@pytest.mark.parametrize("headers,status,code", [
    ({"Content-Length": "262145"}, 413, "body_too_large"),
    ({"Content-Length": "-1"}, 400, "invalid_request"),
    ({"Transfer-Encoding": "chunked"}, 400, "invalid_request"),
    ({"Content-Encoding": "gzip"}, 400, "invalid_request"),
    ({"Content-Type": "text/plain"}, 415, "unsupported_media_type"),
])
def test_body_constraints(server, headers, status, code):
    assert_error(request(server[0], "POST", "/tokenize", payload(), headers=headers), status, code)


def test_tokenize_rejects_extra_fields_and_accepts_content_boundary(server):
    body = payload(True)
    assert_error(request(server[0], "POST", "/tokenize", body), 400, "invalid_request")
    body = payload()
    body["messages"][0]["content"] = "x" * adapter.MAX_CONTENT_CHARS
    assert request(server[0], "POST", "/tokenize", body)[0] == 200


def test_routes_and_methods(server):
    assert_error(request(server[0], path="/health?PRIVATE"), 404, "not_found")
    assert_error(request(server[0], "GET", "/tokenize"), 405, "method_not_allowed")
    assert_error(request(server[0], "POST", "/health"), 405, "method_not_allowed")
    assert_error(request(server[0], "CUSTOM", "/health"), 405, "method_not_allowed")
    assert request(server[0], "HEAD", auth=None)[0] == 401


@pytest.mark.parametrize("first", ["/tokenize", "/v1/chat/completions"])
def test_single_shared_lock_returns_busy_without_waiting(server, first):
    handle, engine = server
    engine.block = True
    results = []
    worker = threading.Thread(target=lambda: results.append(
        request(handle, "POST", first, payload(first.endswith("completions")))
    ))
    worker.start()
    try:
        assert engine.entered.wait(2)
        for path in ("/tokenize", "/v1/chat/completions"):
            assert_error(request(handle, "POST", path, payload(path.endswith("completions"))),
                         429, "busy")
        assert request(handle)[0] == 200
        assert len(engine.calls) == 1
    finally:
        engine.release.set()
        worker.join(3)
    assert not worker.is_alive()
    assert results[0][0] == 200
    assert request(handle, "POST", "/tokenize", payload())[0] == 200


def test_exception_privacy_and_lock_recovery(server, capsys, caplog):
    handle, engine = server
    engine.fail = True
    for path in ("/tokenize", "/v1/chat/completions"):
        response = request(handle, "POST", path, payload(path.endswith("completions")))
        assert_error(response, 503, "inference_unavailable")
        assert response[1]["error"]["message"] == "Inference unavailable."
    request(handle, path="/PRIVATE-secret-path", auth="secret-wrong-key")
    engine.fail = False
    assert request(handle, "POST", "/tokenize", payload())[0] == 200
    captured = capsys.readouterr()
    assert captured.out == captured.err == caplog.text == ""


def test_disconnect_and_stop_do_not_cancel_job(server):
    handle, engine = server
    engine.block = True
    connection = http.client.HTTPConnection("127.0.0.1", handle.port, timeout=0.05)
    connection.request("POST", "/v1/chat/completions", json.dumps(payload(True)), headers={
        "Authorization": "Bearer " + KEY, "Content-Type": "application/json",
    })
    try:
        assert engine.entered.wait(2)
        with pytest.raises(TimeoutError):
            connection.getresponse()
    finally:
        connection.close()
    assert_error(request(handle, "POST", "/tokenize", payload()), 429, "busy")
    handle.stop()
    handle.stop()
    assert handle._server.job_lock.locked()
    assert not engine.finished.is_set()
    assert handle._server.engine is engine
    with pytest.raises(OSError), socket.create_connection(("127.0.0.1", handle.port), timeout=0.2):
        pass
    engine.release.set()
    assert engine.finished.wait(2)


def raw_request(handle, data):
    with socket.create_connection(("127.0.0.1", handle.port), timeout=2) as connection:
        connection.sendall(data)
        response = http.client.HTTPResponse(connection)
        response.begin()
        return response.status, json.loads(response.read()), dict(response.getheaders())


def test_read_timeout_and_duplicate_headers(server, monkeypatch):
    monkeypatch.setattr(adapter, "READ_TIMEOUT_SECONDS", 0.1)
    handle, _ = server
    base = f"POST /tokenize HTTP/1.1\r\nAuthorization: Bearer {KEY}\r\n"
    assert_error(raw_request(handle, (base + "Content-Type: application/json\r\n"
                                     "Content-Length: 10\r\n\r\n{").encode()),
                 408, "request_timeout")
    assert_error(raw_request(handle, (base + "Content-Length: 0\r\n"
                                     "Content-Length: 1\r\n\r\n").encode()),
                 400, "invalid_request")
    assert_error(raw_request(handle, (base + f"Authorization: Bearer {KEY}\r\n\r\n").encode()),
                 401, "unauthorized")


@pytest.mark.parametrize("key", ["short", "é" * 32, "a" * 32 + "\n", " " * 32, None])
def test_invalid_keys_rejected_before_engine_creation(key):
    with pytest.raises(ValueError, match="API key"):
        adapter.start_server(None, None, api_key=key, port=0)


class FakeTensor:
    def __init__(self, values):
        self.values = values

    def __getitem__(self, item):
        row, columns = item
        return FakeTensor(self.values[row][columns])

    def tolist(self):
        return self.values


@pytest.fixture
def runtime(monkeypatch):
    """Small CPU-only dependency doubles to exercise the real engine's decisions."""
    device = SimpleNamespace(type="cuda")
    syncs = []
    torch = SimpleNamespace(
        version=SimpleNamespace(hip="6.4"),
        cuda=SimpleNamespace(is_available=lambda: True, synchronize=syncs.append),
        bfloat16="bf16", long="long", inference_mode=nullcontext,
        tensor=lambda values, **kwargs: FakeTensor(values), ones_like=lambda tensor: tensor,
    )

    class Model:
        def __init__(self):
            self.config = SimpleNamespace(
                _name_or_path=adapter.MODEL_ID, _commit_hash=adapter.REVISION,
                _attn_implementation="sdpa", max_position_embeddings=32768,
            )
            self.device = device
            self.generation_config = SimpleNamespace(eos_token_id=[99, 100])
            self.parameter = SimpleNamespace(device=device, dtype=torch.bfloat16)
            self.evaluated = False
            self.output = [42, 99]
            self.kwargs = None

        def parameters(self):
            return iter([self.parameter])

        def eval(self):
            self.evaluated = True
            return self

        def generate(self, **kwargs):
            self.kwargs = kwargs
            return FakeTensor([kwargs["input_ids"].values[0] + self.output])

    class Tokenizer:
        name_or_path = adapter.MODEL_ID
        init_kwargs = {"_commit_hash": adapter.REVISION}
        eos_token_id = 99
        pad_token_id = None

        def __init__(self):
            self.calls = []
            self.ids = [10, 11, 12, 13, 14, 15, 16]

        def apply_chat_template(self, messages, **kwargs):
            self.calls.append((messages, kwargs))
            return self.ids

        def decode(self, ids, **kwargs):
            assert kwargs == {"skip_special_tokens": True}
            return '{"valid":"even without EOS"}'

    monkeypatch.setitem(sys.modules, "torch", torch)
    monkeypatch.setitem(sys.modules, "transformers", SimpleNamespace(
        PreTrainedModel=Model, PreTrainedTokenizerBase=Tokenizer,
    ))
    return SimpleNamespace(torch=torch, model=Model(), tokenizer=Tokenizer(), syncs=syncs)


@pytest.mark.parametrize("output,reason", [([42, 99], "stop"), ([42, 100], "stop"),
                                          ([42], "length"), ([], "length")])
def test_real_engine_counts_full_template_and_eos(runtime, output, reason):
    engine = adapter._TransformersEngine(runtime.model, runtime.tokenizer, 10)
    assert runtime.model.evaluated
    assert engine.count_tokens(MESSAGES) == 7
    runtime.model.output = output
    result = engine.generate(MESSAGES, 3)
    assert result["usage"] == {"prompt_tokens": 7, "completion_tokens": len(output)}
    assert result["choices"][0]["finish_reason"] == reason
    assert result["choices"][0]["message"]["content"] == '{"valid":"even without EOS"}'
    assert isinstance(result["inference_seconds"], float)
    assert result["inference_seconds"] >= 0
    assert len(runtime.syncs) == 3
    assert runtime.model.kwargs["do_sample"] is False
    assert runtime.model.kwargs["max_new_tokens"] == 3
    assert runtime.model.kwargs["pad_token_id"] == 99
    assert runtime.tokenizer.calls == [(MESSAGES, {
        "tokenize": True, "add_generation_prompt": True, "truncation": False,
    })] * 2


def test_context_limit_uses_actual_template_and_never_truncates(runtime):
    engine = adapter._TransformersEngine(runtime.model, runtime.tokenizer, 8)
    handle = adapter._start_http_server(engine, api_key=KEY, port=0)
    try:
        body = payload(True)
        body["max_tokens"] = 2
        assert_error(request(handle, "POST", "/v1/chat/completions", body),
                     400, "context_length_exceeded")
        assert runtime.model.kwargs is None
        runtime.tokenizer.ids = list(range(12))
        assert request(handle, "POST", "/tokenize", payload())[:2] == (200, {"count": 12})
    finally:
        handle.stop()


def test_generation_failure_synchronizes_before_releasing_lock(runtime, monkeypatch):
    def fail(**kwargs):
        raise RuntimeError("PRIVATE")

    monkeypatch.setattr(runtime.model, "generate", fail)
    engine = adapter._TransformersEngine(runtime.model, runtime.tokenizer, 8192)
    with pytest.raises(RuntimeError):
        engine.generate(MESSAGES, 3)
    assert len(runtime.syncs) == 3


@pytest.mark.parametrize("change", [
    "no_hip", "no_gpu", "cpu", "wrong_model", "missing_commit", "wrong_commit",
    "wrong_tokenizer", "wrong_tokenizer_commit", "wrong_tokenizer_revision", "fp32",
    "non_sdpa", "quantized", "context_too_large", "no_parameters",
])
def test_production_start_rejects_unverified_runtime(runtime, change, monkeypatch):
    model, tokenizer = runtime.model, runtime.tokenizer
    if change == "no_hip":
        runtime.torch.version.hip = None
    elif change == "no_gpu":
        runtime.torch.cuda.is_available = lambda: False
    elif change == "cpu":
        model.device = SimpleNamespace(type="cpu")
    elif change == "wrong_model":
        model.config._name_or_path = "PRIVATE wrong model"
    elif change == "missing_commit":
        del model.config._commit_hash
    elif change == "wrong_commit":
        model.config._commit_hash = "PRIVATE wrong revision"
    elif change == "wrong_tokenizer":
        tokenizer.name_or_path = "PRIVATE tokenizer"
    elif change == "wrong_tokenizer_commit":
        tokenizer.init_kwargs = {"_commit_hash": "wrong"}
    elif change == "wrong_tokenizer_revision":
        tokenizer.init_kwargs = {"revision": "main"}
    elif change == "fp32":
        model.parameter.dtype = "fp32"
    elif change == "non_sdpa":
        model.config._attn_implementation = "eager"
    elif change == "quantized":
        model.is_quantized = True
    elif change == "context_too_large":
        model.config.max_position_embeddings = 1024
    elif change == "no_parameters":
        model.parameters = lambda: iter([])

    def must_not_listen(*args, **kwargs):
        pytest.fail("Unverified engine must not open a listener")

    monkeypatch.setattr(adapter, "_start_http_server", must_not_listen)
    with pytest.raises(ValueError, match="^A verified AMD ROCm BF16 model"):
        adapter.start_server(model, tokenizer, api_key=KEY, port=0)


def test_public_start_verifies_then_exposes_configured_context(runtime):
    handle = adapter.start_server(runtime.model, runtime.tokenizer, api_key=KEY,
                                  port=0, context_tokens=1024)
    try:
        assert request(handle)[1]["context_tokens"] == 1024
    finally:
        handle.stop()
    assert runtime.model.evaluated


def test_standalone_source_has_python310_syntax_and_no_application_import():
    source = Path(adapter.__file__).read_text(encoding="utf-8")
    tree = ast.parse(source, feature_version=(3, 10))
    imports = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imports.extend(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom):
            imports.append(node.module)
    assert not any(name.startswith(("visitbrief", "fastapi")) for name in imports)