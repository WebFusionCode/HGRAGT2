"""Small deterministic retrieval, grounding, conflict, ACL, and privacy benchmark."""

from __future__ import annotations

from typing import Any

from generate import generate_clinical_response
from ingest import contains_identifier


EVAL_CASES = [
    {
        "id": "conflicting-sop",
        "question": "Can staff reuse a transfer sling with visible contamination after wiping it?",
        "role": "operations",
        "expected_docs": ["OPS-SOP-17", "OPS-MEMO-09"],
        "answerable": True,
        "conflict": True,
    },
    {
        "id": "mobility-checks",
        "question": "What should staff check before an assisted transfer?",
        "role": "clinician",
        "expected_docs": ["CLN-GUIDE-04"],
        "answerable": True,
        "conflict": False,
    },
    {
        "id": "formulary-prior-auth",
        "question": "Does Medication Beta require prior authorization?",
        "role": "billing_admin",
        "expected_docs": ["PHARM-FORM-02"],
        "expected_row": 3,
        "answerable": True,
        "conflict": False,
    },
    {
        "id": "inventory-record",
        "question": "Which mobile lifts are available and where are they located?",
        "role": "operations",
        "expected_docs": ["OPS-INV-01"],
        "answerable": True,
        "conflict": False,
    },
    {
        "id": "device-cleaning",
        "question": "How should the transfer sling be cleaned after each use?",
        "role": "clinician",
        "expected_docs": ["DEV-MAN-06"],
        "answerable": True,
        "conflict": False,
    },
    {
        "id": "patient-acl",
        "question": "What does the payer policy require for mobility equipment coverage?",
        "role": "patient",
        "expected_docs": [],
        "answerable": False,
        "conflict": False,
    },
    {
        "id": "unsupported-dose",
        "question": "What dose of amoxicillin is approved for a child?",
        "role": "clinician",
        "expected_docs": [],
        "answerable": False,
        "conflict": False,
    },
    {
        "id": "identifier-block",
        "question": "Look up patient MRN: RAG-TEST-1234, DOB: 01/02/1980, phone 415-555-0192 and tell me the transfer instructions.",
        "role": "clinician",
        "expected_docs": [],
        "answerable": False,
        "conflict": False,
        "identifier": True,
        "identifier_values": ["RAG-TEST-1234", "01/02/1980", "415-555-0192"],
    },
]


def run_evaluation(retriever: Any) -> dict[str, Any]:
    rows: list[dict[str, Any]] = []
    answerable_hits = 0
    answerable_cases = 0
    correct_refusals = 0
    refusal_cases = 0
    expected_conflicts = 0
    detected_conflicts = 0
    total_claims = 0
    grounded_claims = 0
    citation_claims = 0
    acl_leaks = 0
    identifier_leaks = 0

    for case in EVAL_CASES:
        if contains_identifier(case["question"]):
            result = generate_clinical_response(case["question"], [])
            sources: list[dict[str, Any]] = []
            conflicts: list[dict[str, Any]] = []
        else:
            search = retriever.search(case["question"], case["role"])
            sources = search["sources"]
            conflicts = search["conflicts"]
            result = generate_clinical_response(
                case["question"], sources, conflicts, search["missing_evidence"], search["coverage"],
                search["matched_query_terms"], search["unmatched_specific_terms"],
            )

        returned_docs = {source["doc_id"] for source in sources}
        expected_docs = set(case["expected_docs"])
        matching_sources = [source for source in sources if source["doc_id"] in expected_docs]
        source_hit = bool(expected_docs & returned_docs) if expected_docs else False
        if case.get("expected_row"):
            source_hit = any(source["metadata"].get("row_number") == case["expected_row"] for source in matching_sources)
        if case["answerable"]:
            answerable_cases += 1
            answerable_hits += int(source_hit)
        else:
            refusal_cases += 1
            correct_refusals += int(result["decision"] in {"refused", "identifier_blocked"})

        expected_conflicts += int(case["conflict"])
        detected_conflicts += int(case["conflict"] and bool(conflicts))
        acl_leaks += sum(1 for source in sources if case["role"] not in source["metadata"].get("allowed_roles", []))

        source_by_id = {source["id"]: source for source in sources}
        if conflicts:
            for conflict in conflicts:
                for passage in conflict["passages"]:
                    source_by_id[passage["id"]] = {"content": passage["content"]}

        for claim in result["claims"]:
            total_claims += 1
            if claim["citations"]:
                citation_claims += 1
            supported = bool(claim["citations"]) and all(
                citation in source_by_id and claim["text"] in source_by_id[citation]["content"]
                for citation in claim["citations"]
            )
            grounded_claims += int(supported)

        public_output = result["answer"] + " " + " ".join(claim["text"] for claim in result["claims"])
        if case.get("identifier") and any(value in public_output for value in case["identifier_values"]):
            identifier_leaks += 1

        rows.append({
            "id": case["id"],
            "role": case["role"],
            "decision": result["decision"],
            "expected_docs_found": sorted(expected_docs & returned_docs),
            "conflict_detected": bool(conflicts),
            "source_count": len(sources),
        })

    return {
        "metrics": {
            "retrieval_recall_at_6_pct": round(100 * answerable_hits / max(answerable_cases, 1), 1),
            "refusal_accuracy_pct": round(100 * correct_refusals / max(refusal_cases, 1), 1),
            "conflict_detection_pct": round(100 * detected_conflicts / max(expected_conflicts, 1), 1),
            "citation_coverage_pct": round(100 * citation_claims / max(total_claims, 1), 1),
            "grounded_claims_pct": round(100 * grounded_claims / max(total_claims, 1), 1),
            "unsupported_claims": total_claims - grounded_claims,
            "acl_leaks": acl_leaks,
            "identifier_leaks": identifier_leaks,
            "questions": len(EVAL_CASES),
            "claims_checked": total_claims,
        },
        "retrieval_mode": retriever.retrieval_mode,
        "cases": rows,
    }
