import pytest
from pydantic import ValidationError

from visitbrief.schemas import BriefRequest, BriefResult, Evidence, Fact
from visitbrief.settings import Settings


def fact_data():
    return {
        "id": "fact-01",
        "category": "medication",
        "value": "Historical medication recorded in the document.",
        "event_date": None,
        "temporality": "historical",
        "evidence": [{"source_id": "dev-01:visit-note:p1:b1", "quote": "Historical medication"}],
    }


def test_fact_preserves_unknown_date_and_history():
    fact = Fact.model_validate(fact_data())
    assert fact.event_date is None
    assert fact.temporality == "historical"


@pytest.mark.parametrize(
    ("field", "value"),
    [("evidence", []), ("value", "   "), ("category", "diagnosis"),
     ("event_date", "03/04/2025"), ("event_date", "2025-02-30")],
)
def test_invalid_fact_structure_is_rejected(field, value):
    data = fact_data()
    data[field] = value
    with pytest.raises(ValidationError):
        Fact.model_validate(data)


def test_evidence_requires_page_scoped_id_and_quote():
    with pytest.raises(ValidationError):
        Evidence(source_id="visit-note", quote="text")
    with pytest.raises(ValidationError):
        Evidence(source_id="dev-01:visit-note:p1:b1", quote="")


def test_request_does_not_accept_paths_or_prompts():
    with pytest.raises(ValidationError):
        BriefRequest(case_id="../secret")
    with pytest.raises(ValidationError):
        BriefRequest(case_id="dev-01", prompt="Custom instructions")


def test_brief_never_has_clinical_approval_status():
    assert BriefResult.model_json_schema()["properties"]["status"]["enum"] == [
        "ready_for_review", "incomplete"
    ]
    review_flag = BriefResult.model_json_schema()["properties"]["requires_clinician_review"]
    assert review_flag["const"] is True


def test_secret_is_redacted_and_limits_cannot_exceed_mvp():
    settings = Settings(model_api_key="secret-sentinel", _env_file=None)
    assert "secret-sentinel" not in repr(settings)
    for field, value in [("max_documents", 4), ("max_file_bytes", 10485761), ("max_pages", 21)]:
        with pytest.raises(ValidationError):
            Settings(**{field: value}, _env_file=None)