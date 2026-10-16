import importlib.util
from pathlib import Path

_SPEC = importlib.util.spec_from_file_location(
    "evaluate", Path(__file__).parents[1] / "scripts" / "evaluate.py",
)
evaluate = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(evaluate)


def test_score_matches_quote_and_ignores_extra_supported_fact():
    expected = {
        "facts": [{
            "category": "medication",
            "event_date": "2025-01-16",
            "temporality": "recorded",
            "evidence": [{
                "source_id": "dev-01:visit-note:p1:b1",
                "quote": "Current medication: amlodipine 5 mg.",
            }],
        }],
        "review_items": [{"reason": "missing_allergy_documentation", "fact_ids": []}],
    }
    predicted = {
        "case_id": "dev-01",
        "status": "incomplete",
        "facts": [
            {
                "category": "medication",
                "event_date": "2025-01-16",
                "temporality": "recorded",
                "evidence": [{
                    "source_id": "dev-01:visit-note:p1:b1",
                    "quote": "Current medication: amlodipine 5 mg.",
                }],
            },
            {
                "category": "event",
                "event_date": None,
                "temporality": "unknown",
                "evidence": [{
                    "source_id": "dev-01:visit-note:p1:b1",
                    "quote": "An extra recorded sentence.",
                }],
            },
        ],
        "review_items": [{"reason": "missing_allergy_documentation", "fact_ids": []}],
        "metrics": {
            "total_seconds": 1, "inference_seconds": 0.5,
            "input_tokens": 10, "output_tokens": 4,
        },
    }
    row = evaluate.score_case(predicted, expected)
    summary = evaluate.summarize([row])
    assert row["matched_facts"] == 1
    assert row["extra_facts"] == 1
    assert summary["annotated_fact_recall"] == 1
    assert summary["extra_supported_facts"] == 1
    assert summary["annotated_review_recall"] == 1


def test_wrong_category_does_not_match_the_same_quote():
    expected = {
        "facts": [{
            "category": "problem",
            "event_date": None,
            "temporality": "unknown",
            "evidence": [{
                "source_id": "dev-03:urgent-note:p1:b1",
                "quote": "No fracture is seen.",
            }],
        }],
        "review_items": [],
    }
    predicted = {
        "case_id": "dev-03",
        "status": "ready_for_review",
        "facts": [{
            "category": "event",
            "event_date": None,
            "temporality": "unknown",
            "evidence": [{
                "source_id": "dev-03:urgent-note:p1:b1",
                "quote": "No fracture is seen.",
            }],
        }],
        "review_items": [],
        "metrics": {
            "total_seconds": 1, "inference_seconds": 1,
            "input_tokens": 1, "output_tokens": 1,
        },
    }
    assert evaluate.score_case(predicted, expected)["matched_facts"] == 0
