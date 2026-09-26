"""Construct citation-complete responses only from retrieved source passages."""

from __future__ import annotations

import re
from collections import defaultdict
from typing import Any

from ingest import contains_identifier, redact_identifiers
from retrieve import _SYNONYMS, tokenize


def _missing_message(terms: list[str]) -> str:
    if terms:
        return "No accessible current passage matched these requested details: " + ", ".join(terms) + "."
    return "No accessible current passage provided enough detail to answer this question."


def _extract_claims(query: str, sources: list[dict[str, Any]]) -> list[dict[str, Any]]:
    query_terms = set(tokenize(query))
    expanded_query_terms = set(query_terms)
    for term in query_terms:
        for expansion in _SYNONYMS.get(term, []):
            expanded_query_terms.update(tokenize(expansion))
    ranked_claims: list[tuple[float, int, int, str, str, str]] = []
    table_groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for source in sources:
        if source["metadata"].get("table"):
            table_groups[source["doc_id"]].append(source)
    focused_table_ids: set[str] = set()
    for rows in table_groups.values():
        matching_rows = []
        for source in rows:
            item_label = source["content"].split("|", 1)[0].partition(":")[2]
            item_terms = set(tokenize(item_label)) - {"medication", "example", "medicine", "equipment", "transfer", "sling", "model"}
            if item_terms & expanded_query_terms:
                matching_rows.append(source["id"])
        if matching_rows:
            focused_table_ids.update(matching_rows)

    for source in sources:
        metadata = source["metadata"]
        if metadata.get("status") != "active":
            continue
        if metadata.get("table") and focused_table_ids and source["id"] not in focused_table_ids:
            continue
        content = source["content"].strip()
        if metadata.get("table") or "record_number" in metadata or "row_number" in metadata:
            candidates = [content]
        else:
            candidates = [part.strip() for part in re.split(r"(?<=[.!?])\s+|\n+", content) if part.strip()]

        ranked: list[tuple[float, int, int, str]] = []
        for candidate in candidates:
            terms = set(tokenize(candidate))
            direct_overlap = len(query_terms & terms)
            expanded_overlap = len(expanded_query_terms & terms)
            weighted_overlap = direct_overlap * 3 + max(expanded_overlap - direct_overlap, 0)
            if weighted_overlap:
                ranked.append((weighted_overlap / max(len(query_terms), 1), direct_overlap, expanded_overlap, candidate))
        ranked.sort(key=lambda row: (row[0], row[1], row[2]), reverse=True)
        limit = 1 if metadata.get("table") or "record_number" in metadata or "row_number" in metadata else 2
        for relevance, direct_overlap, expanded_overlap, statement in ranked[:limit]:
            ranked_claims.append((relevance, direct_overlap, expanded_overlap, source["id"], statement, metadata.get("status", "active")))

    ranked_claims.sort(key=lambda item: (item[0], item[1], item[2]), reverse=True)
    claims: list[dict[str, Any]] = []
    claim_by_text: dict[str, dict[str, Any]] = {}
    for _, _, _, source_id, statement, status in ranked_claims:
        normalized = re.sub(r"\W+", " ", statement.lower()).strip()
        existing = claim_by_text.get(normalized)
        if existing:
            if source_id not in existing["citations"]:
                existing["citations"].append(source_id)
            continue
        claim = {"text": statement, "citations": [source_id], "source_status": status}
        claims.append(claim)
        claim_by_text[normalized] = claim
        if len(claims) == 5:
            break
    return claims


def generate_clinical_response(
    query: str,
    retrieved_docs: list[dict[str, Any]],
    conflicts: list[dict[str, Any]] | None = None,
    missing_evidence: list[str] | None = None,
    evidence_coverage: float = 1.0,
    matched_query_terms: int = 2,
    unmatched_specific_terms: list[str] | None = None,
) -> dict[str, Any]:
    """Return exact extractive claims, citations, decision state, and evidence gaps."""
    conflicts = conflicts or []
    missing_evidence = missing_evidence or []
    unmatched_specific_terms = unmatched_specific_terms or []
    if contains_identifier(query):
        return {
            "answer": "I can't process questions containing direct personal identifiers. Remove the identifier and ask about the general policy or process.",
            "claims": [], "decision": "identifier_blocked", "missing_evidence": ["A de-identified question"],
        }

    current_sources = [source for source in retrieved_docs if source["metadata"].get("status") == "active"]
    claims = _extract_claims(query, current_sources)

    if not claims or (not conflicts and (evidence_coverage < 0.3 or matched_query_terms < 2 or unmatched_specific_terms)):
        if unmatched_specific_terms:
            missing_evidence = list(dict.fromkeys(unmatched_specific_terms + missing_evidence))
        if conflicts:
            claims = []
        else:
            stale_sources = [source for source in retrieved_docs if source["metadata"].get("status") in {"superseded", "retired"}]
            if stale_sources:
                missing = ["A current, in-force source"]
                response = "Insufficient evidence retrieved. The matching passage is superseded or retired, and no current source supports an answer."
            else:
                missing = missing_evidence or ["A passage that directly addresses the question"]
                response = "Insufficient evidence retrieved. " + _missing_message(missing)
            safe, _ = redact_identifiers(response)
            return {"answer": safe, "claims": [], "decision": "refused", "missing_evidence": missing}

    claim_lines = [f"- {claim['text']} [{', '.join(claim['citations'])}]" for claim in claims]
    if conflicts:
        conflict = conflicts[0]
        claims = [
            {"text": passage["content"], "citations": [passage["id"]], "source_status": passage["status"]}
            for passage in conflict["passages"]
        ]
        conflict_lines = []
        for passage in conflict["passages"]:
            label = "current" if passage["status"] == "active" else passage["status"]
            conflict_lines.append(f"- {label.title()} source ({passage['doc_id']}, effective {passage.get('effective_date', 'date unavailable')}): {passage['content']} [{passage['id']}]")
        answer = "Conflicting instructions were retrieved. The source status and effective date matter; escalate to the named policy owner before acting.\n\n" + "\n".join(conflict_lines)
        decision = "conflict"
    else:
        answer = "Retrieved evidence (verbatim passages):\n\n" + "\n".join(claim_lines)
        decision = "answered"
        overdue = [source for source in current_sources if source.get("freshness_status") == "review due"]
        if overdue:
            answer = "A matching source is past its scheduled review date. Confirm that it remains in force with the policy owner.\n\n" + answer

    safe_answer, _ = redact_identifiers(answer)
    for claim in claims:
        claim["text"], _ = redact_identifiers(claim["text"])
    return {
        "answer": safe_answer,
        "claims": claims,
        "decision": decision,
        "missing_evidence": missing_evidence,
    }
