from datetime import date
from enum import StrEnum
from typing import Annotated, Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

Identifier = Annotated[str, Field(pattern=r"^[a-z0-9]+(?:-[a-z0-9]+)*$", max_length=64)]
NonEmptyText = Annotated[str, Field(min_length=1, max_length=10000, pattern=r"\S")]
SourceIdentifier = Annotated[
    str,
    Field(pattern=r"^[a-z0-9-]+:[a-z0-9-]+:p[1-9][0-9]*:b[1-9][0-9]*$", max_length=160),
]


class Contract(BaseModel):
    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)


class ErrorDetail(Contract):
    code: str
    message: str


class ErrorResponse(Contract):
    error: ErrorDetail


class HealthResponse(Contract):
    status: Literal["ok"] = "ok"
    inference_available: bool = False


class DocumentInfo(Contract):
    document_id: Identifier
    title: NonEmptyText
    document_date: date | None = None


class ManifestDocument(DocumentInfo):
    filename: Annotated[str, Field(pattern=r"^[a-zA-Z0-9][a-zA-Z0-9_-]*\.pdf$")]


class CaseSummary(Contract):
    case_id: Identifier
    title: NonEmptyText
    description: NonEmptyText
    split: Literal["development", "held_out"]
    fictional: Literal[True] = True


class CaseManifest(CaseSummary):
    documents: Annotated[list[ManifestDocument], Field(min_length=1, max_length=3)]

    @model_validator(mode="after")
    def unique_documents(self) -> "CaseManifest":
        ids = [document.document_id for document in self.documents]
        filenames = [document.filename for document in self.documents]
        if len(ids) != len(set(ids)) or len(filenames) != len(set(filenames)):
            raise ValueError("Duplicate document identifiers or filenames")
        return self


class SourceBlock(Contract):
    source_id: SourceIdentifier
    document_id: Identifier
    page_number: int = Field(ge=1)
    start: int = Field(ge=0)
    end: int = Field(gt=0)
    raw_start: int = Field(ge=0)
    raw_end: int = Field(gt=0)
    text: NonEmptyText


class SourcePage(Contract):
    model_config = ConfigDict(extra="forbid", str_strip_whitespace=False)

    document_id: Identifier
    page_number: int = Field(ge=1)
    raw_text: str
    text: str
    blocks: list[SourceBlock]


class CaseDetail(CaseSummary):
    documents: list[DocumentInfo]
    source_status: Literal["not_processed", "available"]
    pages: list[SourcePage]


class Evidence(Contract):
    source_id: SourceIdentifier
    quote: NonEmptyText


class FactCategory(StrEnum):
    PROBLEM = "problem"
    MEDICATION = "medication"
    ALLERGY = "allergy"
    EVENT = "event"
    RECOMMENDATION = "recommendation"


class Fact(Contract):
    id: Identifier
    category: FactCategory
    value: NonEmptyText
    event_date: date | None = None
    temporality: Literal["historical", "recorded", "unknown"] = "unknown"
    evidence: Annotated[list[Evidence], Field(min_length=1)]


class ReviewItem(Contract):
    reason: NonEmptyText
    fact_ids: list[Identifier] = Field(default_factory=list)


class TimelineEntry(Contract):
    event_date: date
    fact_ids: Annotated[list[Identifier], Field(min_length=1)]


class GateEvent(Contract):
    """One model candidate after the quote check.

    `text` is copied only when it already occurs in a source block. Rejected
    wording that is not in a block is counted and not repeated.
    """

    outcome: Literal["kept", "repaired", "dropped"]
    reason: Literal[
        "verbatim_quote",
        "paraphrase_replaced_by_quote",
        "unsupported_evidence_removed",
        "quote_not_in_block",
        "untrusted_instruction",
        "date_not_in_quote",
        "temporality_corrected",
    ]
    category: FactCategory
    text: NonEmptyText | None = None


class BriefMetrics(Contract):
    total_seconds: float = Field(ge=0, allow_inf_nan=False)
    inference_seconds: float = Field(ge=0, allow_inf_nan=False)
    input_tokens: int = Field(ge=0)
    output_tokens: int = Field(ge=0)


class BriefRequest(Contract):
    case_id: Identifier


class BriefResult(Contract):
    case_id: Identifier
    status: Literal["ready_for_review", "incomplete"]
    facts: list[Fact]
    timeline: list[TimelineEntry]
    review_items: list[ReviewItem]
    gate: list[GateEvent] = Field(default_factory=list)
    metrics: BriefMetrics
    requires_clinician_review: Literal[True] = True