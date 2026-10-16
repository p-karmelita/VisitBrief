import time

from visitbrief.catalog import CaseFiles
from visitbrief.inference import InferenceClient
from visitbrief.ingestion import ingest_case, pack_blocks
from visitbrief.schemas import BriefMetrics, BriefResult
from visitbrief.settings import Settings
from visitbrief.validation import assemble_facts, validate_facts


def create_brief(stored: CaseFiles, settings: Settings, client: InferenceClient) -> BriefResult:
    started = time.perf_counter()
    pages = ingest_case(stored, settings)
    blocks = [block for page in pages for block in page.blocks]
    client.check_health()
    batches = pack_blocks(
        blocks, token_count=client.count_tokens,
        context_tokens=settings.model_context_tokens,
        reserved_tokens=settings.model_max_output_tokens,
    )
    facts = []
    gate = []
    incomplete = False
    input_tokens = output_tokens = 0
    inference_seconds = 0.0
    for batch in batches:
        output = client.extract(batch.payload)
        validated = validate_facts(output.extraction.facts, list(batch.blocks))
        facts.extend(validated.facts)
        gate.extend(validated.gate)
        incomplete |= validated.incomplete or not validated.facts
        input_tokens += output.input_tokens
        output_tokens += output.output_tokens
        inference_seconds += output.inference_seconds
    facts, timeline, reviews, incomplete = assemble_facts(facts, blocks, incomplete=incomplete)
    return BriefResult(
        case_id=stored.manifest.case_id, status="incomplete" if incomplete else "ready_for_review",
        facts=facts, timeline=timeline, review_items=reviews, gate=gate,
        metrics=BriefMetrics(
            total_seconds=time.perf_counter() - started, inference_seconds=inference_seconds,
            input_tokens=input_tokens, output_tokens=output_tokens,
        ),
    )