"""Compare live VisitBrief responses with the curated fixture annotations.

The API must already be running. This script never sends annotations to the
model and never treats a stored answer as a prediction. Held-out cases are
scored separately; do not use that score to change the prompt or heuristics.

Annotations are a curated set, not every true sentence in a document. Extra
source-backed facts are counted, not treated as clinical errors.
"""

import argparse
import json
import os
import sys
import urllib.error
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
CASES = ROOT / "fixtures" / "cases"
EXPECTED = ROOT / "fixtures" / "expected"


def request(base: str, path: str, payload: dict | None = None) -> dict:
    data = None if payload is None else json.dumps(payload).encode()
    call = urllib.request.Request(
        base.rstrip("/") + path, data=data,
        headers={"Content-Type": "application/json"} if data else {},
        method="POST" if data else "GET",
    )
    try:
        with urllib.request.urlopen(call, timeout=600) as response:
            return json.load(response)
    except urllib.error.HTTPError as exc:
        body = exc.read().decode("utf-8", "replace")
        raise SystemExit(f"{path} failed with HTTP {exc.code}: {body[:300]}") from None


def quotes(fact: dict) -> set[tuple[str, str]]:
    return {(item["source_id"], item["quote"]) for item in fact["evidence"]}


def score_case(predicted: dict, expected: dict) -> dict:
    expected_facts = expected["facts"]
    predicted_facts = predicted["facts"]
    used: set[int] = set()
    matched = 0
    date_hits = temporality_hits = 0
    for reference in expected_facts:
        reference_quotes = quotes(reference)
        found = None
        for index, candidate in enumerate(predicted_facts):
            if index in used or candidate["category"] != reference["category"]:
                continue
            if reference_quotes & quotes(candidate):
                found = index
                break
        if found is None:
            continue
        used.add(found)
        matched += 1
        candidate = predicted_facts[found]
        date_hits += candidate["event_date"] == reference["event_date"]
        temporality_hits += candidate["temporality"] == reference["temporality"]
    expected_reasons = {item["reason"] for item in expected["review_items"]}
    predicted_reasons = {item["reason"] for item in predicted["review_items"]}
    return {
        "case_id": predicted["case_id"],
        "status": predicted["status"],
        "expected_facts": len(expected_facts),
        "predicted_facts": len(predicted_facts),
        "matched_facts": matched,
        "extra_facts": len(predicted_facts) - len(used),
        "date_hits": date_hits,
        "temporality_hits": temporality_hits,
        "expected_reviews": sorted(expected_reasons),
        "predicted_reviews": sorted(predicted_reasons),
        "matched_reviews": sorted(expected_reasons & predicted_reasons),
        "metrics": predicted["metrics"],
    }


def ratio(numerator: int, denominator: int) -> float | None:
    if denominator == 0:
        return None
    return round(numerator / denominator, 4)


def summarize(rows: list[dict]) -> dict:
    expected = sum(row["expected_facts"] for row in rows)
    matched = sum(row["matched_facts"] for row in rows)
    predicted = sum(row["predicted_facts"] for row in rows)
    extra = sum(row["extra_facts"] for row in rows)
    reviews = sum(len(row["expected_reviews"]) for row in rows)
    review_hits = sum(len(row["matched_reviews"]) for row in rows)
    return {
        "cases": len(rows),
        "annotated_fact_recall": ratio(matched, expected),
        "matched_facts": matched,
        "annotated_facts": expected,
        "extra_supported_facts": extra,
        "displayed_facts": predicted,
        "date_agreement_on_matches": ratio(sum(row["date_hits"] for row in rows), matched),
        "temporality_agreement_on_matches": ratio(
            sum(row["temporality_hits"] for row in rows), matched,
        ),
        "annotated_review_recall": ratio(review_hits, reviews),
        "incomplete_cases": [row["case_id"] for row in rows if row["status"] != "ready_for_review"],
        "inference_seconds": round(sum(row["metrics"]["inference_seconds"] for row in rows), 3),
        "input_tokens": sum(row["metrics"]["input_tokens"] for row in rows),
        "output_tokens": sum(row["metrics"]["output_tokens"] for row in rows),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--split", choices=("development", "held_out", "all"), default="all")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    base = os.environ.get("VISITBRIEF_API_URL", "http://127.0.0.1:8000")
    catalog = json.loads((CASES / "catalog.json").read_text(encoding="utf-8"))
    rows = []
    for case_id in catalog:
        expected = json.loads((EXPECTED / f"{case_id}.json").read_text(encoding="utf-8"))
        if args.split != "all" and expected["split"] != args.split:
            continue
        print(f"scoring {case_id}", flush=True)
        rows.append(score_case(request(base, "/v1/briefs", {"case_id": case_id}), expected))
    grouped: dict[str, list[dict]] = {}
    for row in rows:
        split = json.loads((EXPECTED / f"{row['case_id']}.json").read_text())["split"]
        grouped.setdefault(split, []).append(row)
    report = {
        "model_endpoint": base,
        "splits": {split: summarize(items) for split, items in grouped.items()},
        "cases": rows,
    }
    text = json.dumps(report, indent=2)
    if args.output:
        args.output.write_text(text + "\n", encoding="utf-8")
    print(text)
    if any(row["status"] not in {"ready_for_review", "incomplete"} for row in rows):
        sys.exit(1)


if __name__ == "__main__":
    main()
