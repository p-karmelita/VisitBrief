import json
from collections import Counter

from pypdf import PdfReader

from visitbrief.catalog import CaseRepository
from visitbrief.schemas import Fact, ReviewItem
from visitbrief.settings import PROJECT_ROOT, Settings


def test_fixtures_have_real_text_pdfs_and_source_grounded_annotations():
    repository = CaseRepository(PROJECT_ROOT / "fixtures" / "cases")
    expected_root = PROJECT_ROOT / "fixtures" / "expected"
    cases = repository.list_cases()
    assert Counter(case.split for case in cases) == {"development": 12, "held_out": 8}
    expected_files = {path.stem for path in expected_root.iterdir() if path.suffix == ".json"}
    assert expected_files == {case.case_id for case in cases}
    all_source_ids = set()
    for case in cases:
        stored = repository.get(case.case_id)
        pages = {}
        for document_id, path in stored.documents.items():
            reader = PdfReader(path, strict=True)
            assert not reader.is_encrypted
            assert path.stat().st_size <= Settings(_env_file=None).max_file_bytes
            for number, page in enumerate(reader.pages, start=1):
                text = " ".join(page.extract_text().split())
                assert "FICTIONAL" in text
                assert case.case_id in text
                assert len(text) < 2000
                source_id = f"{case.case_id}:{document_id}:p{number}:b1"
                assert source_id not in all_source_ids
                all_source_ids.add(source_id)
                pages[source_id] = text
        assert 1 <= len(pages) <= 20
        expected = json.loads((expected_root / f"{case.case_id}.json").read_text())
        assert expected["case_id"] == case.case_id
        assert expected["split"] == case.split
        facts = [Fact.model_validate(fact) for fact in expected["facts"]]
        assert facts
        fact_ids = {fact.id for fact in facts}
        assert len(fact_ids) == len(facts)
        for fact in facts:
            for evidence in fact.evidence:
                assert evidence.source_id in pages
                assert evidence.quote in pages[evidence.source_id]
        for item in expected["review_items"]:
            review = ReviewItem.model_validate(item)
            assert set(review.fact_ids) <= fact_ids