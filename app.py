from __future__ import annotations

import os
from typing import Any

import requests
import streamlit as st


st.set_page_config(
    page_title="VoughtCrop",
    page_icon=":material/clinical_notes:",
    layout="wide",
    initial_sidebar_state="expanded",
)

API_ROOT = os.getenv("KNOWLEDGE_API_URL", "http://127.0.0.1:8000").rstrip("/")
ROLE_LABELS = {
    "clinician": "Clinician",
    "operations": "Operations",
    "billing_admin": "Billing administrator",
    "patient": "Patient",
}
EXAMPLES = [
    "Can staff reuse a transfer sling with visible contamination after wiping it?",
    "What should staff check before an assisted transfer?",
    "What initial rest and return-to-activity advice does the concussion guidance give?",
    "What conservative care does the indexed Achilles source describe?",
    "Does Medication Beta require prior authorization?",
    "What dose of amoxicillin is approved for a child?",
]


@st.cache_data(ttl=5, max_entries=1)
def fetch_health() -> dict[str, Any]:
    response = requests.get(f"{API_ROOT}/health", timeout=2)
    response.raise_for_status()
    return response.json()


@st.cache_data(ttl=30, max_entries=8)
def fetch_catalog(role: str) -> list[dict[str, Any]]:
    response = requests.get(f"{API_ROOT}/api/v1/catalog", params={"user_role": role}, timeout=3)
    response.raise_for_status()
    return response.json()


def citation_source_map(result: dict[str, Any]) -> dict[str, dict[str, Any]]:
    sources = {source["id"]: source for source in result.get("sources", [])}
    for conflict in result.get("conflicts", []):
        for passage in conflict.get("passages", []):
            sources.setdefault(passage["id"], {
                "id": passage["id"],
                "doc_id": passage["doc_id"],
                "content": passage["content"],
                "metadata": {
                    "title": passage["title"],
                    "status": passage["status"],
                    "effective_date": passage.get("effective_date"),
                    "source_type": "Conflicting source",
                },
                "freshness_status": passage["status"],
                "score": 0,
            })
    return sources


def render_citation(citation_id: str, source: dict[str, Any]) -> None:
    metadata = source.get("metadata", {})
    title = metadata.get("title", source.get("doc_id", "Source passage"))
    state = source.get("freshness_status", metadata.get("status", "source"))
    with st.expander(f"{citation_id}  |  {title}  |  {state}", icon=":material/article:"):
        details = [source.get("doc_id", "Unknown document"), metadata.get("source_type", "Source")]
        if metadata.get("section"):
            details.append(metadata["section"])
        if metadata.get("row_number"):
            details.append(f"table row {metadata['row_number']}")
        if metadata.get("record_number"):
            details.append(f"record {metadata['record_number']}")
        if metadata.get("effective_date"):
            details.append(f"effective {metadata['effective_date']}")
        st.caption(" | ".join(str(value) for value in details if value))
        source_url = metadata.get("source_url") or metadata.get("citation_url")
        if source_url:
            st.link_button("Open source page", source_url, icon=":material/open_in_new:", width="content")
        st.write(source["content"])
        st.caption(f"Passage ID: {citation_id}")


def render_result(result: dict[str, Any]) -> None:
    decision = result.get("decision", "refused")
    sources = citation_source_map(result)
    status_labels = {
        "answered": ("Evidence found", "green"),
        "conflict": ("Conflict detected", "orange"),
        "refused": ("Insufficient evidence", "red"),
        "identifier_blocked": ("Identifier blocked", "red"),
    }
    label, color = status_labels.get(decision, ("Review required", "gray"))
    st.badge(label, color=color)
    public_search_status = result.get("public_search_status", "not_requested")
    if public_search_status == "answered":
        st.info("Quoted from PubMed abstracts, not organization-approved policy. Review dates and differences before acting.", icon=":material/science:")
    elif public_search_status == "no_support":
        st.info("PubMed was searched, but its abstracts did not support a citation-complete answer; local evidence is retained.", icon=":material/search_off:")
    elif public_search_status in {"no_results", "no_abstracts"}:
        st.info("The public search returned no usable guideline or review abstracts.", icon=":material/search_off:")
    elif public_search_status in {"unavailable", "response_too_large"}:
        st.warning("PubMed fallback did not return usable evidence; no external evidence was added.", icon=":material/warning:")
    elif public_search_status in {"invalid_query", "no_search_terms"}:
        st.caption("No usable de-identified clinical search terms were available for PubMed.")
    elif public_search_status == "unsupported_role":
        st.caption("Public PubMed fallback is limited to the clinician role.")
    elif public_search_status == "local_or_high_risk_scope":
        st.caption("PubMed fallback is not used for organization policy, formulary, approval, operational, or dosing questions.")
    elif public_search_status == "skipped_conflict":
        st.caption("Public search was skipped because accessible organization sources conflict.")
    elif public_search_status == "identifier_blocked":
        st.caption("External search was blocked because the query matched a direct-identifier pattern.")
    elif decision == "refused":
        st.caption("No external public search was requested.")

    if decision == "conflict":
        for conflict in result.get("conflicts", []):
            st.warning(conflict["summary"], icon=":material/compare_arrows:")
            for passage in conflict.get("passages", []):
                citation_id = passage["id"]
                render_citation(citation_id, sources[citation_id])
    elif result.get("claims"):
        answer_text = result.get("answer", "")
        if answer_text.startswith("A matching source is past its scheduled review date"):
            st.warning(answer_text.split("\n\n", 1)[0], icon=":material/warning:")
        st.markdown("**Retrieved statements**")
        for claim in result["claims"]:
            st.markdown(claim["text"])
            for citation_id in claim.get("citations", []):
                source = sources.get(citation_id)
                if source:
                    render_citation(citation_id, source)
    else:
        st.write(result.get("answer", "Insufficient evidence retrieved."))

    missing = result.get("missing_evidence", [])
    if missing and decision in {"refused", "identifier_blocked"}:
        st.caption("Missing: " + ", ".join(missing))

    if result.get("sources"):
        with st.expander(f"Retrieval audit | {len(result['sources'])} passages", icon=":material/manage_search:"):
            st.caption(f"{result.get('retrieval_mode', 'Hybrid retrieval')} | {result.get('processing_ms', 0):.0f} ms")
            for source in result["sources"]:
                meta = source["metadata"]
                st.markdown(f"**{source['doc_id']} | {meta.get('title', 'Source')}**")
                st.caption(f"{source['id']} | {meta.get('source_type', 'Source')} | {source.get('freshness_status', 'unknown')} | score {source.get('score', 0):.3f}")
                st.write(source["content"])

    privacy_status = "blocked input" if result.get("pii_blocked") else f"{result.get('corpus_redactions', 0)} redactions at ingestion"
    st.caption(
        f"{len(result.get('sources', []))} accessible passages | {result.get('retrieval_mode', 'local retrieval')} | "
        f"{result.get('processing_ms', 0):.0f} ms | identifier scan "
        f"{privacy_status}"
    )


for key, default in (("chat_history", []), ("pending_question", ""), ("evaluation_report", None)):
    if key not in st.session_state:
        st.session_state[key] = default

st.title(":material/clinical_notes: VoughtCrop")
st.caption("Evidence workspace for clinical guidance, operations, formulary rules, and source conflicts")

with st.sidebar:
    st.subheader("Access context")
    selected_role = st.selectbox(
        "Simulated user role",
        list(ROLE_LABELS),
        format_func=lambda role: ROLE_LABELS[role],
        key="selected_role",
    )
    if st.session_state.get("active_role") != selected_role:
        st.session_state.chat_history = []
        st.session_state.evaluation_report = None
        st.session_state.active_role = selected_role
    st.caption("Demo role is passed to the API retrieval filter. Connect production roles to verified SSO claims.")

    try:
        health = fetch_health()
        st.success("Local API ready", icon=":material/check_circle:")
        public_count = health.get("public_source_count", 0)
        if public_count:
            st.caption(f"{public_count} validated public source(s) are loaded with synthetic organization records. Answers quote evidence; no hosted answer model is called.")
        else:
            st.caption("The local synthetic corpus is loaded. No hosted answer model is called.")
    except requests.RequestException:
        health = None
        st.error("Local API is offline. Start the backend on port 8000.", icon=":material/wifi_off:")
    st.caption("VoughtCrop | local workspace")

if health:
    metric_docs, metric_passages, metric_search = st.columns(3)
    metric_docs.metric("Indexed documents", health["document_count"])
    metric_passages.metric("Citable passages", health["passage_count"])
    metric_search.metric("Retrieval", "Hybrid")
    st.caption(f"{health['corpus_name']} | {health['retrieval_mode']} | {health['answer_mode']}")

view = st.segmented_control(
    "Workspace view",
    ["Ask", "Corpus", "Quality"],
    default="Ask",
    label_visibility="collapsed",
    key="workspace_view",
)

if view == "Ask":
    st.subheader("Ask the knowledge base", divider="gray")
    st.caption("Answers quote source passages. Unsupported questions are refused; contradictory versions are shown together.")

    if not st.session_state.chat_history:
        st.markdown("**Try a question**")
        suggestion_columns = st.columns(2)
        for index, example in enumerate(EXAMPLES):
            if suggestion_columns[index % 2].button(example, key=f"example_{index}", width="stretch"):
                st.session_state.pending_question = example
                st.rerun()

    include_public_search = False
    if selected_role == "clinician":
        include_public_search = st.checkbox(
            "PubMed fallback",
            key="include_public_search",
            help="If local evidence is insufficient or stale, send general clinical search terms to NCBI PubMed. Never include patient names or details.",
        )

    for message in st.session_state.chat_history:
        with st.chat_message(message["role"]):
            if message["role"] == "user":
                st.write(message["content"])
            else:
                render_result(message["response"])

    typed_question = st.chat_input("Ask about a policy, guideline, device, formulary, or operational record")
    question = typed_question or st.session_state.pending_question
    st.session_state.pending_question = ""

    if question:
        if not health:
            st.error("The knowledge API is offline. Start the backend, then try again.")
        else:
            try:
                with st.spinner("Searching accessible evidence"):
                    response = requests.post(
                        f"{API_ROOT}/api/v1/query",
                        json={
                            "query": question,
                            "user_role": selected_role,
                            "include_public_search": include_public_search,
                        },
                        timeout=45,
                    )
                    response.raise_for_status()
                    result = response.json()
                user_text = "Question blocked because it contained a direct identifier." if result.get("pii_blocked") else question
                st.session_state.chat_history.extend([
                    {"role": "user", "content": user_text},
                    {"role": "assistant", "response": result},
                ])
                st.rerun()
            except requests.RequestException:
                st.error("The knowledge API could not complete this request. Check that the backend is running.")

elif view == "Corpus":
    st.subheader("Accessible source catalog", divider="gray")
    st.caption(f"Showing documents available to the {ROLE_LABELS[selected_role].lower()} role. Filtering is enforced by the API.")
    try:
        catalog = fetch_catalog(selected_role)
        if not catalog:
            st.info("No documents are available to this role.")
        for document in catalog:
            title = f"{document['title']} | {document['doc_id']}"
            status = document.get("freshness_status", document.get("status", "unknown"))
            with st.container(border=True):
                st.markdown(f"**{title}**")
                st.caption(
                    f"{document['source_type']} | {document['publisher']} | v{document['version']} | "
                    f"effective {document['effective_date']} | review {document['review_date']}"
                )
                detail, count = st.columns([3, 1])
                detail.badge(status, color="green" if status == "current" else "orange" if status == "review due" else "gray")
                count.metric("Passages", document["chunk_count"])
                if document.get("source_url"):
                    st.link_button("Open source page", document["source_url"], icon=":material/open_in_new:", width="content")
                if document.get("superseded_by"):
                    st.caption(f"Superseded by {document['superseded_by']}")
    except requests.RequestException:
        st.error("The source catalog is unavailable while the API is offline.")

else:
    st.subheader("Retrieval quality", divider="gray")
    st.caption("Deterministic benchmark over a small, authored question set and the corpus currently loaded by the API.")
    if st.button("Run evaluation", type="primary", icon=":material/play_arrow:"):
        try:
            with st.spinner("Running retrieval, refusal, citation, conflict, and privacy checks"):
                response = requests.post(f"{API_ROOT}/api/v1/evaluate", timeout=30)
                response.raise_for_status()
                st.session_state.evaluation_report = response.json()
        except requests.RequestException:
            st.error("Evaluation requires the local API. Start the backend, then run it again.")

    report = st.session_state.evaluation_report
    if report:
        metrics = report["metrics"]
        first, second, third = st.columns(3)
        first.metric("Grounded claims", f"{metrics['grounded_claims_pct']:.1f}%", f"{metrics['claims_checked']} claims checked")
        second.metric("Retrieval recall@6", f"{metrics['retrieval_recall_at_6_pct']:.1f}%")
        third.metric("Refusal accuracy", f"{metrics['refusal_accuracy_pct']:.1f}%")
        fourth, fifth, sixth = st.columns(3)
        fourth.metric("Conflict detection", f"{metrics['conflict_detection_pct']:.1f}%")
        fifth.metric("ACL leaks", metrics["acl_leaks"], delta_color="inverse")
        sixth.metric("Identifier leaks", metrics["identifier_leaks"], delta_color="inverse")
        st.caption(f"{metrics['questions']} questions | {metrics['unsupported_claims']} unsupported claims | {report['retrieval_mode']}")
        st.dataframe(report["cases"], hide_index=True)
