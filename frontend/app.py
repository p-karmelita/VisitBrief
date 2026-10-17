from html import escape

import streamlit as st
from fpdf import FPDF

from frontend import client

CATEGORIES = {
    "problem": "Documented problems",
    "medication": "Recorded medications",
    "allergy": "Recorded allergies",
    "event": "Events",
    "recommendation": "Recorded recommendations",
}

DEMO_NOTES = {
    "dev-01": "Demo · routine packet. Open a fact to see the quote it came from.",
    "dev-02": "Demo · two recorded doses. The app does not choose between them.",
    "dev-11": (
        "Demo · the document tells the model to invent a diagnosis. "
        "That sentence is not a fact."
    ),
}

DEMO_LABELS = {
    "dev-01": "routine packet",
    "dev-02": "two doses, the app does not choose",
    "dev-11": "an instruction in the source is not a fact",
}

REASON_GLOSS = {
    "conflicting_medication_doses": (
        "Two documents record different doses of the same medicine. "
        "Both stay on screen. The app does not choose."
    ),
    "untrusted_instruction_in_source": (
        "A document tells the model to ignore instructions or invent a diagnosis. "
        "That sentence is not a fact."
    ),
    "missing_allergy_documentation": (
        "No allergy sentence was extracted. This is not a finding of no allergies."
    ),
    "undated_information": (
        "These lines have no date inside the quote, "
        "so they stay off the timeline."
    ),
    "incomplete_document": "A source says the document itself is incomplete or a page is missing.",
    "no_supported_facts": "No model sentence survived the quote check.",
    "incomplete_extraction": (
        "At least one model candidate was dropped or had its wording replaced by the quote."
    ),
}

STORY_REASONS = {
    "conflicting_medication_doses",
    "untrusted_instruction_in_source",
    "no_supported_facts",
    "incomplete_document",
}

GATE_GLOSS = {
    "verbatim_quote": "The displayed sentence is the quote from its source block.",
    "paraphrase_replaced_by_quote": (
        "The model rewrote this sentence. The screen shows the cited quote instead."
    ),
    "unsupported_evidence_removed": "Evidence that was not in its assigned block was removed.",
    "quote_not_in_block": (
        "Dropped. The quote is not in the assigned block. The rejected wording is not shown."
    ),
    "untrusted_instruction": (
        "Dropped. This source sentence tells the model to ignore instructions "
        "or invent a diagnosis."
    ),
    "date_not_in_quote": "A date supplied by the model is not in the quote, so it was cleared.",
    "temporality_corrected": (
        "The wording says this medication is historical, so it was not labeled as current."
    ),
}


def pdf_text(value: object) -> str:
    return str(value).encode("latin-1", "replace").decode("latin-1")


def write_line(pdf: FPDF, text: str, height: float) -> None:
    pdf.multi_cell(0, height, pdf_text(text), new_x="LMARGIN", new_y="NEXT")


def write_heading(pdf: FPDF, text: str) -> None:
    pdf.ln(2)
    pdf.set_font("Helvetica", "B", 13)
    write_line(pdf, text, 7)
    pdf.set_font("Helvetica", size=11)


def write_paragraph(pdf: FPDF, text: str) -> None:
    write_line(pdf, text, 6)


def write_fact(pdf: FPDF, fact) -> None:
    write_paragraph(pdf, fact.value)
    write_paragraph(
        pdf, f"{fact.temporality.title()} · {fact.event_date or 'Date not recorded'}"
    )
    for evidence in fact.evidence:
        write_paragraph(pdf, f"Quote: {evidence.quote}")
    pdf.ln(1)


def brief_pdf(brief) -> bytes:
    pdf = FPDF()
    pdf.set_title(pdf_text(f"VisitBrief {brief.case_id}"))
    pdf.set_margins(16, 16, 16)
    pdf.set_auto_page_break(auto=True, margin=16)
    pdf.add_page()
    pdf.set_font("Helvetica", "B", 16)
    write_line(pdf, "VisitBrief draft", 8)
    pdf.set_font("Helvetica", size=11)
    write_paragraph(pdf, f"Case {brief.case_id}")
    write_paragraph(
        pdf,
        "Fictional draft for review. Not a clinical document. "
        "A matching quote does not prove medical truth.",
    )
    status = (
        "Incomplete extraction. Some information may be missing."
        if brief.status == "incomplete"
        else "Draft ready for review. Not clinically approved."
    )
    write_paragraph(pdf, status)
    by_id = {fact.id: fact for fact in brief.facts}
    for category, title in CATEGORIES.items():
        write_heading(pdf, title)
        facts = [fact for fact in brief.facts if fact.category == category]
        if not facts:
            write_paragraph(pdf, "No extracted facts in this section.")
            continue
        for fact in facts:
            write_fact(pdf, fact)
    write_heading(pdf, "Timeline")
    if not brief.timeline:
        write_paragraph(pdf, "No dated facts.")
    for entry in brief.timeline:
        write_heading(pdf, str(entry.event_date))
        for fact_id in entry.fact_ids:
            if fact_id in by_id:
                write_fact(pdf, by_id[fact_id])
    write_heading(pdf, "Date not recorded")
    undated = [fact for fact in brief.facts if fact.event_date is None]
    if not undated:
        write_paragraph(pdf, "No undated extracted facts.")
    for fact in undated:
        write_fact(pdf, fact)
    write_heading(pdf, "To review")
    if not brief.review_items:
        write_paragraph(pdf, "No automated flags. Clinician review is still required.")
    for item in brief.review_items:
        write_paragraph(pdf, item.reason.replace("_", " ").capitalize())
        for fact_id in item.fact_ids:
            if fact_id in by_id:
                write_fact(pdf, by_id[fact_id])
    write_heading(pdf, "Processing metrics")
    write_paragraph(
        pdf,
        f"Total {brief.metrics.total_seconds:.2f} s. "
        f"Inference {brief.metrics.inference_seconds:.2f} s. "
        f"Input tokens {brief.metrics.input_tokens}. "
        f"Output tokens {brief.metrics.output_tokens}.",
    )
    return bytes(pdf.output())


def clear_case():
    for key in ("detail", "brief", "generation_error", "selected_fact"):
        st.session_state.pop(key, None)


def gloss(reason):
    return REASON_GLOSS.get(reason, reason.replace("_", " ").capitalize())


def show_fact(fact):
    st.text(fact.value)
    st.caption(f"{fact.temporality.title()} · {fact.event_date or 'Date not recorded'}")


def show_evidence(fact, detail):
    documents = {document.document_id: document for document in detail.documents}
    blocks = {
        block.source_id: (page, block) for page in detail.pages for block in page.blocks
    }
    for evidence in fact.evidence:
        source = blocks.get(evidence.source_id)
        if source is None or evidence.quote not in source[1].text:
            st.error("Source quotation could not be verified. Do not rely on this fact.")
            continue
        page, block = source
        document = documents[block.document_id]
        st.text(f"{document.title} · Page {block.page_number}")
        st.caption(f"Document date: {document.document_date or 'Not recorded'}")
        st.code(evidence.source_id, language=None)
        st.text(evidence.quote)
        before, match, after = block.text.partition(evidence.quote)
        st.html(
            '<div style="white-space:pre-wrap;line-height:1.7;border-left:3px solid '
            '#14b8a6;padding:12px">'
            f'{escape(before)}<mark>{escape(match)}</mark>{escape(after)}</div>'
        )
        with st.expander("Full source page (normalized text)"):
            st.text(page.text)


def show_status(brief):
    if brief.status == "incomplete":
        st.warning(
            "Incomplete extraction. Draft for clinician review. "
            "Some model candidates did not reach the screen."
        )
    else:
        st.info(
            "Draft for clinician review. Not a diagnosis and not a treatment change."
        )
    st.caption(
        "Historical records are not current prescriptions. A matching quote does not prove "
        "correct interpretation or medical truth."
    )
    story = []
    routine = []
    for item in brief.review_items:
        if item.reason == "incomplete_extraction" and brief.status == "incomplete":
            continue
        if item.reason in STORY_REASONS:
            story.append(item.reason)
        else:
            routine.append(item.reason)
    for reason in dict.fromkeys(story):
        st.warning(gloss(reason))
    if routine:
        st.caption(" · ".join(gloss(reason) for reason in dict.fromkeys(routine)))


def show_gate(brief):
    if not brief.gate:
        return
    kept = sum(event.outcome == "kept" for event in brief.gate)
    repaired = [event for event in brief.gate if event.outcome == "repaired"]
    dropped = [event for event in brief.gate if event.outcome == "dropped"]
    st.subheader("Model gate")
    counts = st.columns(3)
    counts[0].metric("Kept", kept)
    counts[1].metric("Repaired", len(repaired))
    counts[2].metric("Dropped", len(dropped))
    st.caption("Rejected wording that is not in a source block is counted and not repeated.")
    for event in (*repaired, *dropped):
        label = "Repaired" if event.outcome == "repaired" else "Dropped"
        st.caption(f"{label} · {GATE_GLOSS[event.reason]}")
        if event.text:
            st.text(event.text)


def show_brief(brief, detail):
    show_status(brief)
    by_id = {fact.id: fact for fact in brief.facts}
    facts_column, evidence_column = st.columns([1.15, 1])
    with facts_column:
        facts_tab, timeline_tab, review_tab = st.tabs(["Facts", "Timeline", "To review"])
        with facts_tab:
            for category, title in CATEGORIES.items():
                st.subheader(title)
                facts = [fact for fact in brief.facts if fact.category == category]
                for fact in facts:
                    show_fact(fact)
                if not facts:
                    st.caption("No extracted facts in this section — not evidence of absence.")
        with timeline_tab:
            for entry in brief.timeline:
                st.subheader(str(entry.event_date))
                for fact_id in entry.fact_ids:
                    if fact_id in by_id:
                        show_fact(by_id[fact_id])
            st.subheader("Date not recorded")
            undated = [fact for fact in brief.facts if fact.event_date is None]
            for fact in undated:
                show_fact(fact)
            if not undated:
                st.caption("No undated extracted facts.")
        with review_tab:
            st.caption("Flags are limited checks, not a complete clinical review.")
            for item in brief.review_items:
                st.warning(gloss(item.reason))
                for fact_id in item.fact_ids:
                    if fact_id in by_id:
                        show_fact(by_id[fact_id])
            if not brief.review_items:
                st.info("No automated flags. Clinician review is still required.")
    with evidence_column:
        st.subheader("Evidence")
        if brief.facts:
            fact_id = st.selectbox(
                "Select a fact to inspect its sources", list(by_id),
                format_func=lambda value: f"{by_id[value].category.value}: {by_id[value].value}",
                key="selected_fact",
            )
            show_evidence(by_id[fact_id], detail)
        else:
            st.info("No supported facts were extracted. Inspect the source documents below.")
        show_gate(brief)
    with st.expander("Processing metrics"):
        columns = st.columns(4)
        columns[0].metric("Total", f"{brief.metrics.total_seconds:.2f} s")
        columns[1].metric("Inference", f"{brief.metrics.inference_seconds:.2f} s")
        columns[2].metric("Input tokens", brief.metrics.input_tokens)
        columns[3].metric("Output tokens", brief.metrics.output_tokens)


def main():
    st.set_page_config(
        page_title="VisitBrief · Source-linked review", page_icon="📋", layout="wide",
    )
    st.title("VisitBrief")
    st.caption("DOCUMENTATION → FACTS → SOURCES")
    st.warning(
        "Fictional cases only · Research demo, not a clinical tool. "
        "Every result requires clinician review. No diagnosis or treatment recommendations."
    )
    with st.sidebar:
        st.header("Review workspace")
        st.button("Refresh connection")
        try:
            available = client.health().inference_available
        except client.APIError as exc:
            available = False
            st.error(str(exc))
        if available:
            st.success("AMD inference endpoint available")
            st.caption("Readiness is not proof of extraction quality.")
        else:
            st.warning("Model unavailable · Source browsing only")
        try:
            catalog = client.cases()
        except client.APIError as exc:
            clear_case()
            st.error(str(exc))
            st.stop()
        if not catalog:
            clear_case()
            st.info("No fictional cases available.")
            st.stop()
        summaries = {case.case_id: case for case in catalog}
        case_id = st.selectbox(
            "Fictional case", list(summaries),
            format_func=lambda value: (
                f"{value} · {summaries[value].title} — {DEMO_LABELS[value]}"
                if value in DEMO_LABELS else f"{value} · {summaries[value].title}"
            ),
            key="case_id", on_change=clear_case,
        )
        st.caption(f"Dataset: {summaries[case_id].split.replace('_', ' ')}")
        st.caption("No uploads. No free-form prompts. Results stay in this session only.")
    st.header(summaries[case_id].title)
    st.text(summaries[case_id].description)
    if case_id in DEMO_NOTES:
        st.info(DEMO_NOTES[case_id])
    if "detail" not in st.session_state or st.session_state.detail.case_id != case_id:
        clear_case()
        try:
            with st.spinner("Loading source documents…"):
                st.session_state.detail = client.detail(case_id)
        except client.APIError as exc:
            st.error(str(exc))
            st.stop()
    detail = st.session_state.detail
    sources_available = detail.source_status == "available"
    if not sources_available:
        st.warning("Source documents have not been processed yet.")
    if st.button(
        "Generate brief", type="primary", disabled=not available or not sources_available,
    ):
        for key in ("brief", "generation_error", "selected_fact"):
            st.session_state.pop(key, None)
        try:
            with st.spinner("Extracting and validating facts. This may take minutes…"):
                st.session_state.brief = client.generate(case_id)
        except client.APIError as exc:
            st.session_state.generation_error = str(exc)
    if "brief" in st.session_state:
        st.download_button(
            "Download PDF",
            data=brief_pdf(st.session_state.brief),
            file_name=f"{case_id}-brief.pdf",
            mime="application/pdf",
        )

    if "generation_error" in st.session_state:
        st.error(st.session_state.generation_error)
    if "brief" in st.session_state:
        show_brief(st.session_state.brief, detail)
    else:
        st.info(
            "No brief generated. Browse documents below or connect the AMD model to extract facts."
        )
    st.divider()
    st.subheader("Source documents")
    for document in detail.documents:
        with st.expander(f"{document.title} · {document.document_date or 'Date not recorded'}"):
            for page in detail.pages:
                if page.document_id == document.document_id:
                    st.caption(f"Page {page.page_number}")
                    st.text(page.text)


if __name__ == "__main__":
    main()
