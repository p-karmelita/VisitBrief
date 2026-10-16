"""Upload this file to the AMD notebook; no package installation or model loading.

Use start_server(model, tokenizer, api_key=secret) with the existing BF16 model.
Only 127.0.0.1 is bound; access remotely through an authenticated SSH tunnel.
The model commit must be verifiable from config._commit_hash. Missing provenance
is rejected, never advertised as verified. Tokenizer provenance is checked when
available. The existing SDPA implementation is required, not reconfigured.

Bodies are limited to 256 KiB, each message to 65,536 characters, and socket
reads to 5 seconds. Output is limited to 4,096 tokens. Tokenization reports the
full template length, even over context; generation rejects over-context input.
stop() closes the listener, NOT an active GPU job, and does not unload the model.
Client disconnects/timeouts likewise do not cancel generation or release its lock.
Do not reuse the model elsewhere (or start another server) while a job is active.
"""

import hmac
import json
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

MODEL_ID = "Qwen/Qwen2.5-7B-Instruct"
REVISION = "a09a35458c702b33eeacc393d103063234e8bc28"
MAX_BODY_BYTES = 256 * 1024
MAX_CONTENT_CHARS = 65_536
MAX_OUTPUT_TOKENS = 4096
READ_TIMEOUT_SECONDS = 5.0


class _RequestError(Exception):
    def __init__(self, status, code, message):
        super().__init__(message)
        self.status = status
        self.code = code
        self.message = message


def _invalid_request():
    return _RequestError(400, "invalid_request", "Invalid request.")


def _validate_options(api_key, port, context_tokens):
    if (
        not isinstance(api_key, str)
        or len(api_key) < 32
        or not all(33 <= ord(char) <= 126 for char in api_key)
    ):
        raise ValueError("API key must contain at least 32 visible ASCII characters.")
    if type(port) is not int or not 0 <= port <= 65535:
        raise ValueError("Invalid port.")
    if type(context_tokens) is not int or context_tokens <= 0:
        raise ValueError("Invalid context limit.")


class _TransformersEngine:
    def __init__(self, model, tokenizer, context_tokens):
        try:
            import torch
            from transformers import PreTrainedModel, PreTrainedTokenizerBase

            if (
                not isinstance(model, PreTrainedModel)
                or not isinstance(tokenizer, PreTrainedTokenizerBase)
                or not torch.version.hip
                or not torch.cuda.is_available()
                or model.config._name_or_path != MODEL_ID
                or getattr(model.config, "_commit_hash", None) != REVISION
                or getattr(model.config, "_attn_implementation", None) != "sdpa"
                or getattr(model, "is_quantized", False)
            ):
                raise ValueError
            if context_tokens > model.config.max_position_embeddings:
                raise ValueError
            tokenizer_name = getattr(tokenizer, "name_or_path", None)
            tokenizer_options = getattr(tokenizer, "init_kwargs", {})
            tokenizer_commit = tokenizer_options.get("_commit_hash")
            tokenizer_revision = tokenizer_options.get("revision")
            if (
                (tokenizer_name and tokenizer_name != MODEL_ID)
                or (tokenizer_commit and tokenizer_commit != REVISION)
                or (tokenizer_revision and tokenizer_revision != REVISION)
            ):
                raise ValueError
            self.device = model.device
            if self.device.type != "cuda":
                raise ValueError
            has_parameters = False
            for parameter in model.parameters():
                has_parameters = True
                if parameter.device != self.device or parameter.dtype != torch.bfloat16:
                    raise ValueError
            if not has_parameters:
                raise ValueError
            torch.cuda.synchronize(self.device)
            model.eval()
        except Exception:
            raise ValueError("A verified AMD ROCm BF16 model and tokenizer are required.") from None
        self.torch = torch
        self.model = model
        self.tokenizer = tokenizer
        self.context_tokens = context_tokens

    def _token_ids(self, messages):
        return self.tokenizer.apply_chat_template(
            messages, tokenize=True, add_generation_prompt=True, truncation=False
        )

    def count_tokens(self, messages):
        return len(self._token_ids(messages))

    def generate(self, messages, max_tokens):
        token_ids = self._token_ids(messages)
        prompt_tokens = len(token_ids)
        if prompt_tokens + max_tokens > self.context_tokens:
            raise _RequestError(400, "context_length_exceeded", "Context limit exceeded.")
        torch = self.torch
        eos = self.model.generation_config.eos_token_id
        if eos is None:
            eos = self.tokenizer.eos_token_id
        eos_ids = [eos] if isinstance(eos, int) else list(eos or [])
        pad = self.tokenizer.pad_token_id
        if pad is None and eos_ids:
            pad = eos_ids[0]
        with torch.inference_mode():
            input_ids = torch.tensor([token_ids], dtype=torch.long, device=self.device)
            attention_mask = torch.ones_like(input_ids)
            torch.cuda.synchronize(self.device)
            started = time.perf_counter()
            try:
                output = self.model.generate(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    max_new_tokens=max_tokens,
                    do_sample=False,
                    temperature=None,
                    top_p=None,
                    top_k=None,
                    num_beams=1,
                    num_return_sequences=1,
                    eos_token_id=eos,
                    pad_token_id=pad,
                    return_dict_in_generate=False,
                )
            finally:
                # Even a failing generation can leave queued GPU work behind.
                torch.cuda.synchronize(self.device)
            elapsed = time.perf_counter() - started
            generated = output[0, prompt_tokens:].tolist()
        content = self.tokenizer.decode(generated, skip_special_tokens=True)
        return {
            "model": MODEL_ID,
            "choices": [{
                "message": {"role": "assistant", "content": content},
                "finish_reason": "stop" if any(t in eos_ids for t in generated) else "length",
            }],
            "usage": {"prompt_tokens": prompt_tokens, "completion_tokens": len(generated)},
            "inference_seconds": elapsed,
        }


def _unique_object(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError
        result[key] = value
    return result


def _reject_constant(value):
    raise ValueError


def _validate_request(body, completion):
    fields = {"model", "messages"}
    if completion:
        fields |= {"max_tokens", "temperature"}
    if not isinstance(body, dict) or set(body) != fields:
        raise _invalid_request()
    if body["model"] != MODEL_ID:
        raise _RequestError(400, "model_mismatch", "Unsupported model.")
    messages = body["messages"]
    if not isinstance(messages, list) or len(messages) != 2:
        raise _invalid_request()
    for message, role in zip(messages, ("system", "user"), strict=True):
        if (
            not isinstance(message, dict)
            or set(message) != {"role", "content"}
            or message["role"] != role
            or not isinstance(message["content"], str)
            or not message["content"].strip()
            or len(message["content"]) > MAX_CONTENT_CHARS
        ):
            raise _invalid_request()
    if completion and (
        type(body["max_tokens"]) is not int
        or not 1 <= body["max_tokens"] <= MAX_OUTPUT_TOKENS
        or type(body["temperature"]) not in (int, float)
        or body["temperature"] != 0
    ):
        raise _invalid_request()


class _HTTPServer(ThreadingHTTPServer):
    daemon_threads = True
    block_on_close = False

    def get_request(self):
        connection, address = super().get_request()
        connection.settimeout(READ_TIMEOUT_SECONDS)
        return connection, address

    def handle_error(self, request, client_address):
        # Never let socket failures or engine exceptions print request data.
        pass


class _Handler(BaseHTTPRequestHandler):
    def log_message(self, format, *args):
        pass

    def _reply(self, status, payload):
        self.close_connection = True
        data = json.dumps(payload, ensure_ascii=True, allow_nan=False).encode("utf-8")
        try:
            self.send_response(status)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(data)))
            self.send_header("Cache-Control", "no-store")
            self.send_header("Connection", "close")
            self.end_headers()
            if self.command != "HEAD":
                self.wfile.write(data)
        except OSError:
            pass

    def _error(self, error):
        self._reply(error.status, {"error": {"code": error.code, "message": error.message}})

    def send_error(self, code, message=None, explain=None):
        self._error(_RequestError(code, "invalid_request", "Invalid request."))

    def parse_request(self):
        if not super().parse_request():
            return False
        authorization = self.headers.get_all("Authorization", [])
        if len(authorization) != 1 or not hmac.compare_digest(
            authorization[0].encode("utf-8"), self.server.authorization
        ):
            self._error(_RequestError(401, "unauthorized", "Authentication required."))
            return False
        return True

    def __getattr__(self, name):
        if name.startswith("do_"):
            return self._unsupported_method
        raise AttributeError(name)

    def _unsupported_method(self):
        self._error(_RequestError(405, "method_not_allowed", "Method not allowed."))

    def do_GET(self):
        if self.path == "/health":
            self._reply(200, {
                "status": "ok",
                "backend": "transformers",
                "device": "amd_rocm",
                "model": MODEL_ID,
                "revision": REVISION,
                "context_tokens": self.server.engine.context_tokens,
            })
        elif self.path in ("/tokenize", "/v1/chat/completions"):
            self._unsupported_method()
        else:
            self._error(_RequestError(404, "not_found", "Route not found."))

    def _read_body(self):
        lengths = self.headers.get_all("Content-Length", [])
        if (
            self.headers.get("Transfer-Encoding") is not None
            or self.headers.get("Content-Encoding") is not None
            or len(lengths) != 1
            or not lengths[0].isascii()
            or not lengths[0].isdecimal()
        ):
            raise _invalid_request()
        length = int(lengths[0])
        if length > MAX_BODY_BYTES:
            raise _RequestError(413, "body_too_large", "Request body too large.")
        if self.headers.get_content_type() != "application/json":
            raise _RequestError(415, "unsupported_media_type", "JSON body required.")
        try:
            data = self.rfile.read(length)
        except TimeoutError:
            raise _RequestError(408, "request_timeout", "Request read timed out.") from None
        if len(data) != length:
            raise _invalid_request()
        try:
            return json.loads(
                data.decode("utf-8"), object_pairs_hook=_unique_object,
                parse_constant=_reject_constant,
            )
        except (ValueError, RecursionError):
            raise _invalid_request() from None

    def do_POST(self):
        if self.path not in ("/tokenize", "/v1/chat/completions"):
            if self.path == "/health":
                self._unsupported_method()
            else:
                self._error(_RequestError(404, "not_found", "Route not found."))
            return
        try:
            completion = self.path == "/v1/chat/completions"
            body = self._read_body()
            _validate_request(body, completion)
            if not self.server.job_lock.acquire(blocking=False):
                raise _RequestError(429, "busy", "Inference worker is busy.")
            try:
                if completion:
                    result = self.server.engine.generate(body["messages"], body["max_tokens"])
                else:
                    result = {"count": self.server.engine.count_tokens(body["messages"])}
            finally:
                self.server.job_lock.release()
            self._reply(200, result)
        except _RequestError as error:
            self._error(error)
        except Exception:
            self._error(_RequestError(503, "inference_unavailable", "Inference unavailable."))


class _ServerHandle:
    def __init__(self, server, thread):
        self._server = server
        self._thread = thread
        self._stop_lock = threading.Lock()
        self._stopped = False
        self.port = server.server_port

    def stop(self):
        """Stop accepting requests; active generation continues, model stays loaded."""
        with self._stop_lock:
            if not self._stopped:
                self._server.shutdown()
                self._server.server_close()
                self._thread.join()
                self._stopped = True


def _start_http_server(engine, *, api_key, port):
    """Private transport seam; production callers must use start_server."""
    _validate_options(api_key, port, engine.context_tokens)
    server = _HTTPServer(("127.0.0.1", port), _Handler)
    server.engine = engine
    server.authorization = ("Bearer " + api_key).encode("ascii")
    server.job_lock = threading.Lock()
    thread = threading.Thread(
        target=server.serve_forever, kwargs={"poll_interval": 0.05}, daemon=True
    )
    try:
        thread.start()
    except Exception:
        server.server_close()
        raise
    return _ServerHandle(server, thread)


def start_server(model, tokenizer, *, api_key: str, port: int = 8001, context_tokens: int = 8192):
    """Verify the already-loaded AMD model and start an authenticated loopback server.

    Returns a handle with .port (also useful with port=0) and .stop(). No imports
    from the CPU application, downloads, dtype conversions, or model reloads.
    """
    _validate_options(api_key, port, context_tokens)
    engine = _TransformersEngine(model, tokenizer, context_tokens)
    return _start_http_server(engine, api_key=api_key, port=port)