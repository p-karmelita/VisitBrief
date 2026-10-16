import re
from collections import defaultdict
from dataclasses import dataclass
from datetime import date

from visitbrief.inference import ExtractedFact
from visitbrief.schemas import Fact, ReviewItem, SourceBlock, TimelineEntry


@dataclass(frozen=True)
class ValidatedFacts:
    facts: list[Fact]
    incomplete: bool


_UNTRUSTED_INSTRUCTION = re.compile(
    r"ignore (?:all )?previous instructions|invent a diagnosis",
    re.IGNORECASE,
)


def _supported_evidence(candidate: ExtractedFact, sources: dict[str, str]) -> ExtractedFact | None:
    """Keep verbatim quotes. A rewritten value is replaced by the quote it cites."""
    evidence = [
        item for item in candidate.evidence
        if item.source_id in sources and item.quote in sources[item.source_id]
        and not _UNTRUSTED_INSTRUCTION.search(item.quote)
    ]
    if not evidence:
        return None
    if any(candidate.value == item.quote for item in evidence):
        return candidate.model_copy(update={"evidence": evidence})
    # One cited sentence is enough to display. The paraphrase is not shown.
    if len(evidence) == 1:
        return candidate.model_copy(update={"value": evidence[0].quote, "evidence": evidence})
    return None


def validate_facts(candidates: list[ExtractedFact], blocks: list[SourceBlock]) -> ValidatedFacts:
    sources = {block.source_id: block.text for block in blocks}
    facts = []
    incomplete = False
    for candidate in candidates:
        repaired = _supported_evidence(candidate, sources)
        if repaired is None:
            incomplete = True
            continue
        if repaired.value != candidate.value or repaired.evidence != candidate.evidence:
            incomplete = True
        candidate = repaired
        evidence = candidate.evidence
        event_date = None
        if candidate.event_date is not None:
            try:
                event_date = date.fromisoformat(candidate.event_date)
            except ValueError:
                incomplete = True
            if event_date is not None and not any(
                candidate.event_date in item.quote or (
                    re.search(r"\b(today|current|this encounter)\b", item.quote, re.IGNORECASE)
                    and f"Document date: {candidate.event_date}" in sources[item.source_id]
                ) for item in evidence
            ):
                event_date = None
                incomplete = True
        temporality = candidate.temporality
        if candidate.category == "medication" and re.search(
            r"\b(historical|no longer|stopped|ended|completed|was prescribed)\b",
            candidate.value, re.IGNORECASE,
        ) and temporality != "historical":
            temporality = "historical"
            incomplete = True
        facts.append(Fact(
            id=f"fact-{len(facts) + 1:02d}", category=candidate.category,
            value=candidate.value, event_date=event_date, temporality=temporality,
            evidence=evidence,
        ))
    return ValidatedFacts(facts, incomplete)


def assemble_facts(
    facts: list[Fact], blocks: list[SourceBlock], *, incomplete: bool,
) -> tuple[list[Fact], list[TimelineEntry], list[ReviewItem], bool]:
    merged = {}
    for fact in facts:
        key = (fact.category, fact.value, fact.event_date, fact.temporality)
        if key not in merged:
            merged[key] = fact.model_copy(
                update={"id": f"fact-{len(merged) + 1:02d}", "evidence": []},
            )
        target = merged[key]
        for evidence in fact.evidence:
            if evidence not in target.evidence:
                target.evidence.append(evidence)
    facts = list(merged.values())
    reviews = []
    dates = defaultdict(list)
    undated = []
    doses = defaultdict(list)
    for fact in facts:
        if fact.event_date is None:
            undated.append(fact.id)
        else:
            dates[fact.event_date].append(fact.id)
        if fact.category == "medication" and fact.temporality != "historical":
            match = re.search(
                r"\bcurrent medication:\s*([a-z][a-z -]*?)\s+"
                r"(\d+(?:\.\d+)?\s+(?:mcg|mg|g|units))\b", fact.value, re.IGNORECASE,
            )
            if match:
                for evidence in fact.evidence:
                    block = next(item for item in blocks if item.source_id == evidence.source_id)
                    document_date = re.search(r"Document date: (\d{4}-\d{2}-\d{2})", block.text)
                    if document_date:
                        doses[(match[1].casefold(), document_date[1])].append(
                            (fact.id, match[2], block.document_id)
                        )
    for entries in doses.values():
        if len({item[1] for item in entries}) > 1 and len({item[2] for item in entries}) > 1:
            reviews.append(ReviewItem(
                reason="conflicting_medication_doses",
                fact_ids=list(dict.fromkeys(item[0] for item in entries)),
            ))
    if not any(fact.category == "allergy" for fact in facts):
        reviews.append(ReviewItem(reason="missing_allergy_documentation"))
    if undated:
        reviews.append(ReviewItem(reason="undated_information", fact_ids=undated))
    if any(re.search(
        r"\b(incomplete|missing page|page missing|text truncated|document truncated)\b",
        block.text, re.IGNORECASE,
    ) for block in blocks):
        reviews.append(ReviewItem(reason="incomplete_document"))
        incomplete = True
    if any(_UNTRUSTED_INSTRUCTION.search(block.text) for block in blocks):
        reviews.append(ReviewItem(reason="untrusted_instruction_in_source"))
    if not facts:
        reviews.append(ReviewItem(reason="no_supported_facts"))
        incomplete = True
    if incomplete:
        reviews.append(ReviewItem(reason="incomplete_extraction"))
    timeline = [TimelineEntry(event_date=day, fact_ids=dates[day]) for day in sorted(dates)]
    return facts, timeline, reviews, incomplete