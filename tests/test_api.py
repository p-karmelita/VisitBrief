import pytest
from fastapi.testclient import TestClient

from visitbrief.api import create_app
from visitbrief.schemas import CaseDetail, CaseSummary
from visitbrief.settings import Settings


@pytest.fixture
def client():
    with TestClient(create_app(Settings(_env_file=None))) as test_client:
        yield test_client


def test_health_does_not_claim_model_readiness(client):
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json() == {"status": "ok", "inference_available": False}


def test_catalog_has_twenty_fictional_cases_with_fixed_split(client):
    response = client.get("/v1/cases")
    assert response.status_code == 200
    cases = [CaseSummary.model_validate(case) for case in response.json()]
    assert len(cases) == 20
    assert len({case.case_id for case in cases}) == 20
    assert sum(case.split == "development" for case in cases) == 12
    assert sum(case.split == "held_out" for case in cases) == 8
    assert all(case.fictional for case in cases)
    assert "filename" not in response.text


def test_details_are_isolated_and_exclude_internal_paths(client):
    cases = client.get("/v1/cases").json()
    for case in cases:
        response = client.get(f"/v1/cases/{case['case_id']}")
        assert response.status_code == 200
        detail = CaseDetail.model_validate(response.json())
        assert detail.case_id == case["case_id"]
        assert 1 <= len(detail.documents) <= 3
        assert detail.source_status == "available"
        assert detail.pages
        for page in detail.pages:
            assert page.document_id in {document.document_id for document in detail.documents}
            assert page.blocks
            for block in page.blocks:
                assert block.source_id.startswith(f"{case['case_id']}:")
                assert page.text[block.start:block.end] == block.text
        assert "filename" not in response.text
        assert str(Settings(_env_file=None).cases_dir) not in response.text


@pytest.mark.parametrize("case_id", ["missing-case", "manifest.json", "%2E%2E%2Fmain.py", "DEV-01"])
def test_unknown_or_path_like_case_ids_are_not_read(client, case_id):
    response = client.get(f"/v1/cases/{case_id}")
    assert response.status_code == 404
    assert response.json()["error"]["code"] in {"case_not_found", "not_found"}
    assert "main.py" not in response.text


def test_no_upload_prompt_or_fake_brief_endpoint(client):
    response = client.post("/v1/briefs", json={"case_id": "dev-01"})
    assert response.status_code == 503
    assert response.json()["error"]["code"] == "inference_disabled"
    assert client.post("/v1/cases", json={"prompt": "private text"}).status_code == 405
    assert client.post(
        "/v1/briefs", json={"case_id": "dev-01", "prompt": "text"},
    ).status_code == 422
    paths = client.get("/openapi.json").json()["paths"]
    assert set(paths) == {"/health", "/v1/cases", "/v1/cases/{case_id}", "/v1/briefs"}


def test_errors_do_not_echo_requests_and_responses_are_not_cached(client, caplog):
    marker = "private-content-sentinel"
    response = client.get(f"/v1/cases/{marker}")
    assert response.status_code == 404
    assert marker not in response.text
    assert marker not in caplog.text
    assert response.headers["cache-control"] == "no-store"
    assert client.get("/v1/cases/dev-01").headers["cache-control"] == "no-store"


def test_bad_catalog_produces_a_redacted_stable_error(tmp_path):
    (tmp_path / "catalog.json").write_text("PRIVATE invalid JSON", encoding="utf-8")
    with TestClient(create_app(Settings(cases_dir=tmp_path, _env_file=None))) as client:
        response = client.get("/v1/cases")
    assert response.status_code == 503
    assert response.json()["error"]["code"] == "case_data_unavailable"
    assert "PRIVATE" not in response.text