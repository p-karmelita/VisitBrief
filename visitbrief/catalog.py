from dataclasses import dataclass
from pathlib import Path

from pydantic import TypeAdapter, ValidationError

from visitbrief.errors import VisitBriefError
from visitbrief.schemas import CaseManifest, CaseSummary, Identifier


@dataclass(frozen=True)
class CaseFiles:
    manifest: CaseManifest
    documents: dict[str, Path]


def unavailable() -> VisitBriefError:
    return VisitBriefError("case_data_unavailable", "Fictional case data is unavailable.", 503)


class CaseRepository:
    def __init__(self, root: Path):
        self.root = root.resolve()

    @staticmethod
    def _read(path: Path, limit: int) -> bytes:
        try:
            with path.open("rb") as source:
                data = source.read(limit + 1)
            if len(data) > limit:
                raise unavailable()
            return data
        except OSError:
            raise unavailable() from None

    @staticmethod
    def _contained_file(directory: Path, filename: str) -> Path:
        try:
            path = (directory / filename).resolve(strict=True)
            if path.parent != directory or not path.is_file():
                raise unavailable()
            return path
        except (OSError, RuntimeError):
            raise unavailable() from None

    def _case_ids(self) -> list[str]:
        path = self._contained_file(self.root, "catalog.json")
        try:
            case_ids = TypeAdapter(list[Identifier]).validate_json(self._read(path, 16384))
        except ValidationError:
            raise unavailable() from None
        if len(case_ids) != len(set(case_ids)) or len(case_ids) > 100:
            raise unavailable()
        return case_ids

    def list_cases(self) -> list[CaseSummary]:
        return [
            CaseSummary.model_validate(self._load(case_id).manifest.model_dump(exclude={"documents"}))
            for case_id in self._case_ids()
        ]

    def get(self, case_id: str) -> CaseFiles:
        if case_id not in self._case_ids():
            raise VisitBriefError("case_not_found", "Case not found.", 404)
        return self._load(case_id)

    def _load(self, case_id: str) -> CaseFiles:
        try:
            directory = (self.root / case_id).resolve(strict=True)
            if directory.parent != self.root or directory.name != case_id:
                raise unavailable()
            path = self._contained_file(directory, "manifest.json")
            manifest = CaseManifest.model_validate_json(self._read(path, 65536))
            if manifest.case_id != case_id:
                raise unavailable()
            documents = {
                document.document_id: self._contained_file(directory, document.filename)
                for document in manifest.documents
            }
            return CaseFiles(manifest=manifest, documents=documents)
        except (OSError, RuntimeError, ValidationError):
            raise unavailable() from None