from threading import Lock

from fastapi import FastAPI, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from starlette.exceptions import HTTPException

from visitbrief.catalog import CaseRepository
from visitbrief.errors import VisitBriefError
from visitbrief.inference import InferenceClient
from visitbrief.ingestion import ingest_case
from visitbrief.pipeline import create_brief
from visitbrief.schemas import (
    BriefRequest,
    BriefResult,
    CaseDetail,
    CaseSummary,
    ErrorResponse,
    HealthResponse,
)
from visitbrief.settings import Settings


def error_response(code: str, message: str, status_code: int) -> JSONResponse:
    return JSONResponse(
        status_code=status_code,
        content={"error": {"code": code, "message": message}},
        headers={"Cache-Control": "no-store"},
    )


def create_app(settings: Settings | None = None) -> FastAPI:
    settings = settings or Settings()
    repository = CaseRepository(settings.cases_dir)
    inference_lock = Lock()
    app = FastAPI(
        title="VisitBrief",
        version="0.1.0",
        description=(
            "Fictional documentation only. Research demonstrator, not a clinical tool. "
            "Draft results require source review; matching quotes do not prove medical truth."
        ),
        responses={
            404: {"model": ErrorResponse},
            422: {"model": ErrorResponse},
            429: {"model": ErrorResponse},
            503: {"model": ErrorResponse},
            504: {"model": ErrorResponse},
        },
    )

    @app.middleware("http")
    async def no_store(request: Request, call_next):
        response = await call_next(request)
        response.headers["Cache-Control"] = "no-store"
        response.headers["X-Content-Type-Options"] = "nosniff"
        return response

    @app.exception_handler(VisitBriefError)
    async def application_error(request: Request, exc: VisitBriefError):
        return error_response(exc.code, exc.message, exc.status_code)

    @app.exception_handler(RequestValidationError)
    async def validation_error(request: Request, exc: RequestValidationError):
        return error_response("invalid_request", "The request is invalid.", 422)

    @app.exception_handler(HTTPException)
    async def http_error(request: Request, exc: HTTPException):
        code = {404: "not_found", 405: "method_not_allowed"}.get(exc.status_code, "request_failed")
        return error_response(code, "The request cannot be served.", exc.status_code)

    @app.get("/health", response_model=HealthResponse)
    def health() -> HealthResponse:
        if not settings.inference_enabled:
            return HealthResponse()
        try:
            probe_timeout = 15 if settings.model_backend == "token_factory" else 3
            probe_settings = settings.model_copy(update={"model_timeout_seconds": probe_timeout})
            with InferenceClient(probe_settings) as model:
                model.check_health()
            return HealthResponse(inference_available=True)
        except VisitBriefError:
            return HealthResponse()

    @app.get("/v1/cases", response_model=list[CaseSummary])
    def list_cases() -> list[CaseSummary]:
        return repository.list_cases()

    @app.get("/v1/cases/{case_id}", response_model=CaseDetail)
    def get_case(case_id: str) -> CaseDetail:
        stored = repository.get(case_id)
        data = stored.manifest.model_dump(exclude={"documents"})
        data["documents"] = [
            document.model_dump(exclude={"filename"}) for document in stored.manifest.documents
        ]
        return CaseDetail(**data, source_status="available", pages=ingest_case(stored, settings))

    @app.post("/v1/briefs", response_model=BriefResult)
    def brief(request: BriefRequest) -> BriefResult:
        stored = repository.get(request.case_id)
        if not inference_lock.acquire(blocking=False):
            raise VisitBriefError("inference_busy", "Another case is being processed.", 429)
        try:
            with InferenceClient(settings) as model:
                return create_brief(stored, settings, model)
        finally:
            inference_lock.release()

    return app