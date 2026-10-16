from html import escape

import streamlit as st

from frontend import client

CATEGORIES = {
    "problem": "Documented problems",
    "medication": "Recorded medications",
    "allergy": "Recorded allergies",
    "event": "Events",
    "recommendation": "Recorded recommendations",
}


def clear_case():
    for key in ("detail", "brief", "generation_error", "selected_fact"):
        st.session_state.pop(key, None)


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


def show_brief(brief, detail):
    if brief.status == "incomplete":
        st.warning("Incomplete extraction — some information may be missing. Review all sources.")
    else:
        st.info("Draft ready for review — not clinically approved.")
    st.caption(
        "Historical records are not current prescriptions. A matching quote does not prove "
        "correct interpretation or medical truth."
    )
    facts_tab, timeline_tab, review_tab = st.tabs(["Facts", "Timeline", "To review"])
    with facts_tab:
        for category, title in CATEGORIES.items():
            st.subheader(title)
            facts = [fact for fact in brief.facts if fact.category == category]
            for fact in facts:
                show_fact(fact)
            if not facts:
                st.caption("No extracted facts in this section — not evidence of absence.")
    by_id = {fact.id: fact for fact in brief.facts}
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
            st.warning(item.reason.replace("_", " ").capitalize())
            for fact_id in item.fact_ids:
                if fact_id in by_id:
                    show_fact(by_id[fact_id])
        if not brief.review_items:
            st.info("No automated flags. Clinician review is still required.")
    st.divider()
    st.subheader("Evidence explorer")
    if brief.facts:
        fact_id = st.selectbox(
            "Select a fact to inspect its sources", list(by_id),
            format_func=lambda value: f"{by_id[value].category.value}: {by_id[value].value}",
            key="selected_fact",
        )
        show_evidence(by_id[fact_id], detail)
    else:
        st.info("No supported facts were extracted. Inspect the source documents below.")
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
            format_func=lambda value: f"{value} · {summaries[value].title}",
            key="case_id", on_change=clear_case,
        )
        st.caption(f"Dataset: {summaries[case_id].split.replace('_', ' ')}")
        st.caption("No uploads. No free-form prompts. Results stay in this session only.")
    st.header(summaries[case_id].title)
    st.text(summaries[case_id].description)
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