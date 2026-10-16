import os

import httpx
from pydantic import TypeAdapter, ValidationError

from visitbrief.schemas import BriefResult, CaseDetail, CaseSummary, HealthResponse

ERRORS = {
    "inference_disabled": "Model inference is disabled. You can still browse source documents.",
    "inference_busy": "Another case is being processed. Please try again later.",
    "model_unavailable": "The AMD model is unavailable. No brief was generated.",
    "model_timeout": "The model timed out. No brief was generated.",
    "model_busy": "The shared AMD endpoint is busy. Wait a moment and try again.",
}


class APIError(Exception):
    pass


def request(path, schema, *, case_id=None):
    base_url = os.environ.get("VISITBRIEF_API_URL", "http://127.0.0.1:8000").rstrip("/")
    try:
        with httpx.Client(
            base_url=base_url, timeout=httpx.Timeout(600 if case_id else 15, connect=5),
            trust_env=False,
        ) as client:
            response = (
                client.post(path, json={"case_id": case_id}) if case_id
                else client.get(path)
            )
        if response.is_error:
            try:
                code = response.json()["error"]["code"]
            except (ValueError, KeyError, TypeError):
                code = None
            message = ERRORS.get(code) if isinstance(code, str) else None
            raise APIError(message or f"Backend request failed (HTTP {response.status_code}).")
        return TypeAdapter(schema).validate_json(response.content)
    except httpx.TimeoutException:
        raise APIError(
            "The backend timed out. Processing may still be running; wait before retrying."
        ) from None
    except httpx.HTTPError:
        raise APIError("Cannot reach the backend. Check that the API is running.") from None
    except (ValidationError, ValueError):
        raise APIError("The backend returned an invalid response.") from None


def health():
    return request("/health", HealthResponse)


def cases():
    return request("/v1/cases", list[CaseSummary])


def detail(case_id):
    result = request(f"/v1/cases/{case_id}", CaseDetail)
    if result.case_id != case_id:
        raise APIError("The backend returned a different case.")
    return result


def generate(case_id):
    result = request("/v1/briefs", BriefResult, case_id=case_id)
    if result.case_id != case_id:
        raise APIError("The backend returned a different case.")
    return result