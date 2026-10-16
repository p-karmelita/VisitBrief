import json

import httpx
import pytest
from pydantic import ValidationError

from visitbrief.errors import VisitBriefError
from visitbrief.inference import MODEL_REVISION, InferenceClient, messages_for
from visitbrief.settings import Settings


def configured(**kwargs):
    return Settings(inference_enabled=True, model_api_key="s" * 32, _env_file=None, **kwargs)


def completion(content='{"facts": []}', finish_reason="stop"):
    return {
        "model": "Qwen/Qwen2.5-7B-Instruct",
        "choices": [{"message": {"role": "assistant", "content": content},
                     "finish_reason": finish_reason}],
        "usage": {"prompt_tokens": 100, "completion_tokens": 20},
        "inference_seconds": 0.125,
    }


def test_private_protocol_and_full_chat_token_count():
    requests = []

    def handle(request):
        requests.append(request)
        assert request.headers["authorization"] == "Bearer " + "s" * 32
        if request.url.path == "/health":
            return httpx.Response(200, json={
                "status": "ok", "backend": "transformers", "device": "amd_rocm",
                "model": "Qwen/Qwen2.5-7B-Instruct", "revision": MODEL_REVISION,
                "context_tokens": 8192,
            })
        payload = json.loads(request.content)
        assert payload["messages"] == messages_for("fictional payload")
        if request.url.path == "/tokenize":
            return httpx.Response(200, json={"count": 100})
        assert request.url.path == "/v1/chat/completions"
        assert payload["temperature"] == 0
        return httpx.Response(200, json=completion())

    with InferenceClient(configured(), transport=httpx.MockTransport(handle)) as model:
        model.check_health()
        assert model.count_tokens("fictional payload") == 100
        output = model.extract("fictional payload")
    assert output.extraction.facts == []
    assert (output.input_tokens, output.output_tokens) == (100, 20)
    assert output.inference_seconds >= 0
    assert len(requests) == 3


@pytest.mark.parametrize("bad", ["not JSON", '{"facts": [{"extra": 1}]}', "```json\n{}\n```"])
def test_invalid_structure_retried_once_and_token_usage_counted(bad):
    replies = [completion(bad), completion()]
    with InferenceClient(configured(), transport=httpx.MockTransport(
        lambda _: httpx.Response(200, json=replies.pop(0)),
    )) as model:
        output = model.extract("fictional payload")
    assert not replies
    assert output.input_tokens == 200
    assert output.output_tokens == 40
    assert output.inference_seconds == 0.25


@pytest.mark.parametrize("failure, expected_code, attempts", [
    ("length", "model_invalid_response", 2),
    (503, "model_unavailable", 2),
    (401, "model_configuration_mismatch", 1),
    (302, "model_configuration_mismatch", 1),
    (429, "model_busy", 1),
    ("timeout", "model_timeout", 1),
    ("network", "model_unavailable", 2),
])
def test_failures_are_bounded_redacted_and_not_false_success(
    failure, expected_code, attempts, caplog,
):
    calls = []

    def handle(request):
        calls.append(request)
        if failure == "timeout":
            raise httpx.ReadTimeout("PRIVATE-TEXT", request=request)
        if failure == "network":
            raise httpx.ConnectError("PRIVATE-TEXT", request=request)
        if failure == "length":
            return httpx.Response(200, json=completion(finish_reason="length"))
        return httpx.Response(failure, text="PRIVATE-TEXT", headers={"location": "https://example.com"})

    with InferenceClient(configured(), transport=httpx.MockTransport(handle)) as model:
        with pytest.raises(VisitBriefError) as error:
            model.extract("PRIVATE-TEXT")
    assert error.value.code == expected_code
    assert len(calls) == attempts
    assert "PRIVATE-TEXT" not in str(error.value)
    assert "PRIVATE-TEXT" not in caplog.text
    assert "s" * 32 not in caplog.text


@pytest.mark.parametrize("count", [True, -1, 0, "10", None])
def test_invalid_token_counts_are_rejected(count):
    with InferenceClient(configured(), transport=httpx.MockTransport(
        lambda _: httpx.Response(200, json={"count": count}),
    )) as model:
        with pytest.raises(VisitBriefError, match="invalid"):
            model.count_tokens("test")


def test_health_does_not_accept_unverified_model_metadata():
    with InferenceClient(configured(), transport=httpx.MockTransport(
        lambda _: httpx.Response(200, json={"status": "ok"}),
    )) as model:
        with pytest.raises(VisitBriefError) as error:
            model.check_health()
    assert error.value.code == "model_configuration_mismatch"


@pytest.mark.parametrize("url", [
    "http://public.example/v1", "http://127.0.0.1:8001/v1?secret=x",
    "http://user:password@127.0.0.1:8001/v1", "http://127.0.0.1:8001/wrong",
])
def test_insecure_or_ambiguous_configuration_is_rejected(url):
    with pytest.raises(ValidationError):
        configured(model_base_url=url)


def test_enabled_requires_secret_and_context_budget():
    with pytest.raises(ValidationError):
        Settings(inference_enabled=True, _env_file=None)
    with pytest.raises(ValidationError):
        configured(model_context_tokens=1024)


def test_token_factory_uses_catalog_count_and_gateway_model_name():
    requests = []

    def handle(request):
        requests.append((request.method, request.url.path))
        assert request.headers["authorization"] == "Bearer " + "s" * 32
        if request.url.path.endswith("/v1/models"):
            return httpx.Response(200, json={"data": [{"id": "DeepSeek-V4-Flash"}]})
        body = json.loads(request.content)
        if request.url.path.endswith("/v1/messages/count_tokens"):
            assert body["model"] == "DeepSeek-V4-Flash"
            assert body["messages"][0]["content"] == "fictional payload"
            return httpx.Response(200, json={"input_tokens": 4})
        assert body["response_format"] == {"type": "json_object"}
        assert body["reasoning_effort"] == "none"
        assert "inference_seconds" not in body
        return httpx.Response(200, json={
            "model": "self-dploy/DeepSeek-V4-Flash",
            "choices": [{"message": {"role": "assistant", "content": '{"facts": []}'},
                         "finish_reason": "stop"}],
            "usage": {"prompt_tokens": 19, "completion_tokens": 6},
        })

    settings = configured(
        model_backend="token_factory",
        model_base_url="https://developer.amd.com.cn/radeon/api/v1",
        model_id="DeepSeek-V4-Flash",
        reasoning_effort="none",
        model_context_tokens=32768,
        model_max_output_tokens=4096,
    )
    with InferenceClient(settings, transport=httpx.MockTransport(handle)) as model:
        model.check_health()
        # Remote counters can under-report. The character floor still has to win.
        assert model.count_tokens("fictional payload") > 4
        output = model.extract("fictional payload")
    assert output.extraction.facts == []
    assert output.inference_seconds >= 0
    # An empty fact list is retried once on Token Factory.
    assert [path.rpartition("/v1/")[2] for _, path in requests] == [
        "models", "messages/count_tokens", "chat/completions", "chat/completions",
    ]


def test_token_factory_retries_rate_limit_then_accepts(monkeypatch):
    monkeypatch.setattr("visitbrief.inference.time.sleep", lambda _: None)
    calls = {"n": 0}

    def handle(request):
        calls["n"] += 1
        if calls["n"] == 1:
            return httpx.Response(
                429, headers={"retry-after": "1"}, json={"error": {"message": "slow"}},
            )
        return httpx.Response(200, json={
            "model": "DeepSeek-V4-Flash",
            "choices": [{"message": {"content": json.dumps({"facts": [{
                "category": "event", "value": "Example.", "event_date": None,
                "temporality": "unknown",
                "evidence": [{"source_id": "dev-01:note:p1:b1", "quote": "Example."}],
            }]})}, "finish_reason": "stop"}],
            "usage": {"prompt_tokens": 10, "completion_tokens": 40},
        })

    settings = configured(
        model_backend="token_factory",
        model_base_url="https://developer.amd.com.cn/radeon/api/v1",
        model_id="DeepSeek-V4-Flash",
        model_context_tokens=32768,
        model_max_output_tokens=4096,
    )
    with InferenceClient(settings, transport=httpx.MockTransport(handle)) as model:
        output = model.extract("fictional payload")
    assert calls["n"] == 2
    assert len(output.extraction.facts) == 1


def test_token_factory_rejects_a_model_missing_from_the_catalog():
    settings = configured(
        model_backend="token_factory",
        model_base_url="https://developer.amd.com.cn/radeon/api/v1",
        model_id="DeepSeek-V4-Flash",
        model_context_tokens=32768,
        model_max_output_tokens=4096,
    )
    with InferenceClient(settings, transport=httpx.MockTransport(
        lambda _: httpx.Response(200, json={"data": [{"id": "Other"}]}),
    )) as model:
        with pytest.raises(VisitBriefError) as error:
            model.check_health()
    assert error.value.code == "model_configuration_mismatch"


def test_blank_reasoning_effort_is_unset():
    settings = Settings(
        inference_enabled=False, reasoning_effort="", _env_file=None,
    )
    assert settings.reasoning_effort is None


def test_oversized_response_is_rejected():
    with InferenceClient(configured(), transport=httpx.MockTransport(
        lambda _: httpx.Response(200, content=b" " * (1024 * 1024 + 1)),
    )) as model:
        with pytest.raises(VisitBriefError) as error:
            model.extract("test")
    assert error.value.code == "model_invalid_response"