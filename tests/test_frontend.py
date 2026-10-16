import json
from pathlib import Path
from typing import TypedDict

import httpx
import pytest
from streamlit.testing.v1 import AppTest

from frontend import client

APP = Path(__file__).resolve().parents[1] / "frontend" / "app.py"
QUOTE = "Historical medication: <script>alert('unsafe')</script>"


class BackendState(TypedDict):
    available: bool
    status: str
    error: int | None
    posts: list[str]


@pytest.fixture
def backend(monkeypatch):
    state: BackendState = {
        "available": True, "status": "ready_for_review", "error": None, "posts": [],
    }
    catalog = [{
        "case_id": case_id, "title": f"Fictional {case_id}", "description": "Test case",
        "split": "development", "fictional": True,
    } for case_id in ("dev-01", "dev-02")]

    def handle(request):
        if request.url.path == "/health":
            return httpx.Response(200, json={
                "status": "ok", "inference_available": state["available"],
            })
        if request.url.path == "/v1/cases":
            return httpx.Response(200, json=catalog)
        if request.method == "POST":
            payload = json.loads(request.content)
            assert set(payload) == {"case_id"}
            case_id = payload["case_id"]
            state["posts"].append(case_id)
            if error_status := state["error"]:
                return httpx.Response(error_status, json={"error": {
                    "code": "inference_busy", "message": "PRIVATE ERROR BODY",
                }})
            return httpx.Response(200, json={
                "case_id": case_id, "status": state["status"], "facts": [{
                    "id": "fact-01", "category": "medication", "value": QUOTE,
                    "event_date": "2025-01-01", "temporality": "historical",
                    "evidence": [{"source_id": f"{case_id}:note:p2:b1", "quote": QUOTE}],
                }, {
                    "id": "fact-02", "category": "problem", "value": "Undated record",
                    "event_date": None, "temporality": "unknown",
                    "evidence": [{
                        "source_id": f"{case_id}:note:p1:b1", "quote": "Undated record",
                    }],
                }],
                "timeline": [{"event_date": "2025-01-01", "fact_ids": ["fact-01"]}],
                "review_items": [{"reason": "missing_allergy", "fact_ids": []}],
                "metrics": {"total_seconds": 2, "inference_seconds": 1,
                            "input_tokens": 100, "output_tokens": 20},
                "requires_clinician_review": True,
            })
        case_id = request.url.path.rsplit("/", 1)[1]
        summary = next(case for case in catalog if case["case_id"] == case_id)
        pages = []
        for number, text in enumerate(("Undated record", f"Before {QUOTE} after"), start=1):
            pages.append({
                "document_id": "note", "page_number": number, "raw_text": text, "text": text,
                "blocks": [{"source_id": f"{case_id}:note:p{number}:b1", "document_id": "note",
                            "page_number": number, "start": 0, "end": len(text),
                            "raw_start": 0, "raw_end": len(text), "text": text}],
            })
        return httpx.Response(200, json={
            **summary, "documents": [{"document_id": "note", "title": "Visit note",
                                      "document_date": "2025-02-01"}],
            "source_status": "available", "pages": pages,
        })

    original = httpx.Client
    monkeypatch.setattr("frontend.client.httpx.Client", lambda **kwargs: original(
        **kwargs, transport=httpx.MockTransport(handle),
    ))
    monkeypatch.setenv("VISITBRIEF_API_URL", "http://test-api")
    return state


def generate(app):
    next(button for button in app.button if button.label == "Generate brief").click().run()
    assert not app.exception


def test_ui_source_browsing_without_model(backend):
    backend["available"] = False
    app = AppTest.from_file(APP).run()
    assert not app.exception
    assert next(button for button in app.button if button.label == "Generate brief").disabled
    assert any("Source browsing only" in warning.value for warning in app.warning)
    assert any("Undated record" in text.value for text in app.text)
    assert not app.text_input
    assert not app.get("file_uploader")
    assert backend["posts"] == []


@pytest.mark.parametrize("status", ["ready_for_review", "incomplete"])
def test_ui_brief_sources_timeline_and_switching(backend, status):
    backend["status"] = status
    app = AppTest.from_file(APP).run()
    generate(app)
    assert backend["posts"] == ["dev-01"]
    assert [tab.label for tab in app.tabs] == ["Facts", "Timeline", "To review"]
    assert any("Historical" in caption.value for caption in app.caption)
    assert any("Date not recorded" in caption.value for caption in app.caption)
    assert any("2025-01-01" in header.value for header in app.subheader)
    assert any("Missing allergy" in warning.value for warning in app.warning)
    assert any("Visit note · Page 2" in text.value for text in app.text)
    html = app.get("html")[0].proto.body
    assert "<mark>Historical medication: &lt;script&gt;" in html
    assert "<script>" not in html
    if status == "incomplete":
        assert any("Incomplete extraction" in warning.value for warning in app.warning)
    app.selectbox(key="selected_fact").select("fact-02").run()
    assert not app.exception
    assert any("Visit note · Page 1" in text.value for text in app.text)
    app.selectbox(key="case_id").select("dev-02").run()
    assert not app.exception
    assert "brief" not in app.session_state
    assert "selected_fact" not in app.session_state
    assert app.session_state["detail"].case_id == "dev-02"
    assert not app.tabs
    assert any("No brief generated" in info.value for info in app.info)


def test_failed_regeneration_clears_previous_result(backend):
    app = AppTest.from_file(APP).run()
    generate(app)
    backend["error"] = 429
    generate(app)
    assert "brief" not in app.session_state
    assert not app.tabs
    assert any("Another case" in error.value for error in app.error)
    assert all("PRIVATE" not in error.value for error in app.error)
    app.selectbox(key="case_id").select("dev-02").run()
    assert "generation_error" not in app.session_state


def test_sessions_do_not_share_results(backend):
    first = AppTest.from_file(APP).run()
    generate(first)
    second = AppTest.from_file(APP).run()
    assert "brief" in first.session_state
    assert "brief" not in second.session_state
    assert not second.tabs


@pytest.mark.parametrize("failure, message", [
    (httpx.ConnectError("SECRET"), "Cannot reach the backend"),
    (httpx.ReadTimeout("SECRET"), "Processing may still be running"),
    (httpx.Response(200, text="SECRET"), "invalid response"),
    (httpx.Response(503, text="SECRET"), "HTTP 503"),
    (httpx.Response(200, json={"status": "ok"}), "invalid response"),
])
def test_client_errors_are_safe(monkeypatch, failure, message):
    def handle(_):
        if isinstance(failure, Exception):
            raise failure
        return failure

    original = httpx.Client
    monkeypatch.setattr("frontend.client.httpx.Client", lambda **kwargs: original(
        **kwargs, transport=httpx.MockTransport(handle),
    ))
    with pytest.raises(client.APIError) as error:
        client.generate("dev-01")
    assert message in str(error.value)
    assert "SECRET" not in str(error.value)


def test_client_rejects_wrong_case(backend, monkeypatch):
    wrong_case = client.generate("dev-02")
    monkeypatch.setattr(client, "request", lambda *args, **kwargs: wrong_case)
    with pytest.raises(client.APIError, match="different case"):
        client.generate("dev-01")


def test_ui_backend_offline_is_visible(monkeypatch):
    def offline():
        raise client.APIError("Cannot reach the backend.")

    monkeypatch.setattr(client, "health", offline)
    monkeypatch.setattr(client, "cases", offline)
    app = AppTest.from_file(APP).run()
    assert not app.exception
    assert any("Cannot reach" in error.value for error in app.error)
    assert not app.selectbox