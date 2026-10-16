import json

import pytest

from visitbrief.catalog import CaseRepository
from visitbrief.errors import VisitBriefError


@pytest.fixture
def case_root(tmp_path):
    (tmp_path / "catalog.json").write_text('["dev-01"]', encoding="utf-8")
    directory = tmp_path / "dev-01"
    directory.mkdir()
    manifest = {
        "case_id": "dev-01", "title": "Fictional", "description": "Test only",
        "fictional": True, "split": "development",
        "documents": [{"document_id": "note", "title": "Note", "filename": "note.pdf"}],
    }
    (directory / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    (directory / "note.pdf").write_bytes(b"PDF content is not read by the catalog")
    return tmp_path


def test_repository_resolves_only_manifest_documents(case_root):
    stored = CaseRepository(case_root).get("dev-01")
    assert stored.manifest.case_id == "dev-01"
    assert stored.documents == {"note": case_root / "dev-01" / "note.pdf"}
    with pytest.raises(VisitBriefError, match="Case not found"):
        CaseRepository(case_root).get("../dev-01")


@pytest.mark.parametrize("filename", ["../other.pdf", "/tmp/other.pdf", "other/note.pdf"])
def test_path_traversal_in_manifest_is_rejected(case_root, filename):
    path = case_root / "dev-01" / "manifest.json"
    manifest = json.loads(path.read_text())
    manifest["documents"][0]["filename"] = filename
    path.write_text(json.dumps(manifest), encoding="utf-8")
    with pytest.raises(VisitBriefError) as error:
        CaseRepository(case_root).get("dev-01")
    assert error.value.code == "case_data_unavailable"


def test_unlisted_directory_cannot_be_opened(case_root):
    (case_root / "catalog.json").write_text("[]", encoding="utf-8")
    with pytest.raises(VisitBriefError) as error:
        CaseRepository(case_root).get("dev-01")
    assert error.value.code == "case_not_found"


def test_symlink_to_another_case_is_rejected(case_root):
    other = case_root / "dev-02"
    other.mkdir()
    (other / "note.pdf").write_bytes(b"Another fictional case")
    document = case_root / "dev-01" / "note.pdf"
    document.unlink()
    document.symlink_to(other / "note.pdf")
    with pytest.raises(VisitBriefError) as error:
        CaseRepository(case_root).get("dev-01")
    assert error.value.code == "case_data_unavailable"


def test_manifest_case_id_cannot_impersonate_another_case(case_root):
    path = case_root / "dev-01" / "manifest.json"
    manifest = json.loads(path.read_text())
    manifest["case_id"] = "dev-02"
    path.write_text(json.dumps(manifest), encoding="utf-8")
    with pytest.raises(VisitBriefError):
        CaseRepository(case_root).get("dev-01")


def test_duplicate_catalog_ids_are_rejected(case_root):
    (case_root / "catalog.json").write_text('["dev-01", "dev-01"]', encoding="utf-8")
    with pytest.raises(VisitBriefError):
        CaseRepository(case_root).list_cases()