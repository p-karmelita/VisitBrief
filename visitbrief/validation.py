import re
from collections import defaultdict
from dataclasses import dataclass
from datetime import date

from visitbrief.inference import ExtractedFact
from visitbrief.schemas import Fact, GateEvent, ReviewItem, SourceBlock, TimelineEntry


@dataclass(frozen=True)
class ValidatedFacts:
    facts: list[Fact]
    incomplete: bool
    gate: tuple[GateEvent, ...] = ()


_UNTRUSTED_INSTRUCTION = re.compile(
    r"ignore (?:all )?previous instructions|invent a diagnosis",
    re.IGNORECASE,
)


def _gate(outcome: str, reason: str, candidate: ExtractedFact, text: str | None) -> GateEvent:
    return GateEvent(outcome=outcome, reason=reason, category=candidate.category, text=text)


def _classify_support(
    candidate: ExtractedFact, sources: dict[str, str],
) -> tuple[ExtractedFact | None, str, str | None]:
    """Return the displayable fact, the gate reason, and source text safe to repeat."""
    trusted = []
    untrusted = []
    for item in candidate.evidence:
        block = sources.get(item.source_id)
        if block is None or item.quote not in block:
            continue
        if _UNTRUSTED_INSTRUCTION.search(item.quote):
            untrusted.append(item.quote)
        else:
            trusted.append(item)
    if not trusted:
        if untrusted:
            return None, "untrusted_instruction", untrusted[0]
        return None, "quote_not_in_block", None
    if any(candidate.value == item.quote for item in trusted):
        updated = candidate.model_copy(update={"evidence": trusted})
        reason = (
            "verbatim_quote" if updated.evidence == candidate.evidence
            else "unsupported_evidence_removed"
        )
        return updated, reason, updated.value
    # Several cited sentences can support one rewritten value. Show the first
    # sentence and keep the others as evidence. The rewrite itself is not shown.
    shown = trusted[0].quote
    return (
        candidate.model_copy(update={"value": shown, "evidence": trusted}),
        "paraphrase_replaced_by_quote",
        shown,
    )


def validate_facts(candidates: list[ExtractedFact], blocks: list[SourceBlock]) -> ValidatedFacts:
    sources = {block.source_id: block.text for block in blocks}
    facts = []
    notes = []
    incomplete = False
    for original in candidates:
        repaired, reason, safe_text = _classify_support(original, sources)
        if repaired is None:
            incomplete = True
            notes.append(_gate("dropped", reason, original, safe_text))
            continue
        outcome = "kept" if reason == "verbatim_quote" else "repaired"
        if repaired.value != original.value or repaired.evidence != original.evidence:
            incomplete = True
        notes.append(_gate(outcome, reason, repaired, safe_text))
        evidence = repaired.evidence
        event_date = None
        if original.event_date is not None:
            try:
                parsed = date.fromisoformat(original.event_date)
            except ValueError:
                parsed = None
            supported = parsed is not None and any(
                original.event_date in item.quote or (
                    re.search(r"\b(today|current|this encounter)\b", item.quote, re.IGNORECASE)
                    and f"Document date: {original.event_date}" in sources[item.source_id]
                ) for item in evidence
            )
            if supported:
                event_date = parsed
            else:
                incomplete = True
                notes.append(_gate("repaired", "date_not_in_quote", repaired, repaired.value))
        temporality = original.temporality
        if repaired.category == "medication" and re.search(
            r"\b(historical|no longer|stopped|ended|completed|was prescribed)\b",
            repaired.value, re.IGNORECASE,
        ) and temporality != "historical":
            temporality = "historical"
            incomplete = True
            notes.append(_gate("repaired", "temporality_corrected", repaired, repaired.value))
        facts.append(Fact(
            id=f"fact-{len(facts) + 1:02d}", category=repaired.category,
            value=repaired.value, event_date=event_date, temporality=temporality,
            evidence=evidence,
        ))
    return ValidatedFacts(facts, incomplete, tuple(notes))


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