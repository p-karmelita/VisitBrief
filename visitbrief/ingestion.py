import json
import logging
import re
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from io import BytesIO
from pathlib import Path

from pypdf import PdfReader

from visitbrief.catalog import CaseFiles
from visitbrief.errors import VisitBriefError
from visitbrief.schemas import SourceBlock, SourcePage
from visitbrief.settings import Settings

# Parser diagnostics can contain document bytes; surface only our stable error codes.
_pdf_logger = logging.getLogger("pypdf")
_pdf_logger.addHandler(logging.NullHandler())
_pdf_logger.propagate = False


def normalize_with_mapping(raw_text: str) -> tuple[str, list[int]]:
    pieces = []
    offsets = []
    previous_end = 0
    for match in re.finditer(r"\S+", raw_text):
        if pieces:
            pieces.append(" ")
            offsets.append(previous_end)
        pieces.append(match.group())
        offsets.extend(range(match.start(), match.end()))
        previous_end = match.end()
    return "".join(pieces), offsets


def source_page(
    case_id: str, document_id: str, page_number: int, raw_text: str, max_block_chars: int
) -> SourcePage:
    text, offsets = normalize_with_mapping(raw_text)
    if not text:
        raise VisitBriefError(
            "pdf_no_text", "A page has no extractable text. OCR is not supported."
        )
    blocks = []
    start = 0
    while start < len(text):
        end = min(start + max_block_chars, len(text))
        if end < len(text):
            boundary = text.rfind(" ", start, end + 1)
            if boundary > start:
                end = boundary
        blocks.append(SourceBlock(
            source_id=f"{case_id}:{document_id}:p{page_number}:b{len(blocks) + 1}",
            document_id=document_id,
            page_number=page_number,
            start=start,
            end=end,
            raw_start=offsets[start],
            raw_end=offsets[end - 1] + 1,
            text=text[start:end],
        ))
        start = end
        while start < len(text) and text[start] == " ":
            start += 1
    return SourcePage(
        document_id=document_id, page_number=page_number, raw_text=raw_text,
        text=text, blocks=blocks,
    )


def read_pdf_bytes(path: Path, limit: int) -> bytes:
    try:
        with path.open("rb") as source:
            data = source.read(limit + 1)
    except OSError:
        raise VisitBriefError("pdf_unavailable", "A source document is unavailable.", 503) from None
    if len(data) > limit:
        raise VisitBriefError("pdf_too_large", "A source document exceeds the file size limit.")
    return data


def ingest_case(stored: CaseFiles, settings: Settings) -> list[SourcePage]:
    if len(stored.manifest.documents) > settings.max_documents:
        raise VisitBriefError("too_many_documents", "The case exceeds the document count limit.")
    pages = []
    total_pages = 0
    total_chars = 0
    for document in stored.manifest.documents:
        data = read_pdf_bytes(stored.documents[document.document_id], settings.max_file_bytes)
        try:
            reader = PdfReader(BytesIO(data), strict=True)
            if reader.is_encrypted:
                raise VisitBriefError("pdf_encrypted", "Encrypted documents are not supported.")
            page_count = len(reader.pages)
            if page_count == 0:
                raise VisitBriefError("pdf_no_pages", "A source document contains no pages.")
            total_pages += page_count
            if total_pages > settings.max_pages:
                raise VisitBriefError("too_many_pages", "The case exceeds the total page limit.")
            for number, page in enumerate(reader.pages, start=1):
                raw_text = page.extract_text()
                total_chars += len(raw_text)
                if total_chars > settings.max_extracted_chars:
                    raise VisitBriefError(
                        "too_much_text", "The case exceeds the extracted text limit."
                    )
                pages.append(source_page(
                    stored.manifest.case_id, document.document_id, number,
                    raw_text, settings.max_block_chars,
                ))
        except VisitBriefError:
            raise
        except Exception:
            raise VisitBriefError("pdf_invalid", "A source PDF could not be read safely.") from None
    return pages


@dataclass(frozen=True)
class SourceBatch:
    blocks: tuple[SourceBlock, ...]
    payload: str
    input_tokens: int


def pack_blocks(
    blocks: Sequence[SourceBlock], *, token_count: Callable[[str], int],
    context_tokens: int, reserved_tokens: int,
) -> list[SourceBatch]:
    """Count JSON using the model tokenizer; reserve system/chat/output tokens separately."""
    if context_tokens <= 0 or reserved_tokens < 0 or reserved_tokens >= context_tokens:
        raise ValueError("Invalid context budget")
    if len({block.source_id.split(":", 1)[0] for block in blocks}) > 1:
        raise VisitBriefError("mixed_cases", "Source batches must contain exactly one case.")
    if len({block.source_id for block in blocks}) != len(blocks):
        raise VisitBriefError("duplicate_sources", "Source identifiers must be unique.")
    budget = context_tokens - reserved_tokens

    def make_batch(items: tuple[SourceBlock, ...]) -> SourceBatch:
        payload = json.dumps(
            {"sources": [{"source_id": item.source_id, "text": item.text} for item in items]},
            ensure_ascii=False, separators=(",", ":"),
        )
        count = token_count(payload)
        if not isinstance(count, int) or isinstance(count, bool) or count <= 0:
            raise ValueError("Tokenizer must return a positive integer")
        return SourceBatch(items, payload, count)

    batches = []
    current = None
    for block in blocks:
        candidate = make_batch((*current.blocks, block) if current else (block,))
        if candidate.input_tokens > budget:
            if current is not None:
                batches.append(current)
                candidate = make_batch((block,))
            if candidate.input_tokens > budget:
                raise VisitBriefError(
                    "source_block_too_large", "A source block exceeds the model input budget."
                )
        current = candidate
    if current is not None:
        batches.append(current)
    return batches