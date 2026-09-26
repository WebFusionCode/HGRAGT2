"""Opt-in, query-time retrieval of guideline and review abstracts from PubMed."""

from __future__ import annotations

import hashlib
import os
import re
import xml.etree.ElementTree as ET
from datetime import date, datetime
from typing import Any

import requests

from ingest import contains_identifier
from retrieve import _SYNONYMS, tokenize


EUTILS = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils"
MAX_RECORDS = 6
MAX_RESPONSE_BYTES = 2_000_000
PUBLICATION_FILTER = (
    '("Guideline"[Publication Type] OR "Practice Guideline"[Publication Type] '
    'OR "Systematic Review"[Publication Type] OR "Meta-Analysis"[Publication Type] '
    'OR "Review"[Publication Type])'
)
SEARCH_NOISE = {
    "advised", "after", "according", "brief", "clinical", "current", "days",
    "duration", "few", "first", "following", "guidance", "guideline", "guidelines",
    "hours", "initial", "later", "many", "months", "most", "period", "possible",
    "potential", "recommended", "recommendation", "short", "should", "suspected",
    "time", "weeks", "years",
}
SEARCH_SYNONYMS = {
    "dose": ["dosage", "dosing"],
    "treatment": ["management", "therapy", "intervention"],
    "treatments": ["management", "therapy", "interventions"],
    "return": ["return to activity", "gradual return", "resumption"],
    "rest": ["relative rest", "physical rest", "cognitive rest"],
}
LOCAL_OR_HIGH_RISK_TERMS = {
    "approved", "approval", "coverage", "dosing", "dose", "dosage", "formulary",
    "inventory", "payer", "policy", "prior authorization", "workflow",
}


def build_search_term(query: str) -> str:
    """Keep external searches focused on clinical terms, not raw user prose."""
    original_terms = [
        token for token in tokenize(query)
        if len(token) >= 3 and not token.isdigit() and token not in SEARCH_NOISE
    ]
    groups: list[str] = []
    for token in original_terms[:6]:
        variants = [token, *SEARCH_SYNONYMS.get(token, []), *_SYNONYMS.get(token, [])]
        phrases = list(dict.fromkeys(
            " ".join(word for word in tokenize(variant) if not word.isdigit())
            for variant in variants
        ))
        phrases = [phrase for phrase in phrases if phrase]
        if phrases:
            groups.append("(" + " OR ".join(f'"{phrase}"[Title/Abstract]' for phrase in phrases) + ")")
    if not groups:
        return ""
    return f"({' AND '.join(groups)}) AND {PUBLICATION_FILTER}"


def is_local_or_high_risk_query(query: str) -> bool:
    lowered = re.sub(r"\s+", " ", query.lower())
    if any(term in lowered for term in LOCAL_OR_HIGH_RISK_TERMS):
        return True
    tokens = set(tokenize(query))
    return bool(tokens & {"hospital", "organization", "organizational", "operational", "our", "staff", "sop"})


def _element_text(element: ET.Element | None) -> str:
    if element is None:
        return ""
    return re.sub(r"\s+", " ", " ".join(element.itertext())).strip()


def _published_date(article: ET.Element) -> str | None:
    candidates = [
        article.find(".//ArticleDate"),
        article.find(".//JournalIssue/PubDate"),
        article.find(".//PubMedPubDate[@PubStatus='pubmed']"),
    ]
    for node in candidates:
        if node is None:
            continue
        year = _element_text(node.find("Year"))
        if not year:
            medline_date = _element_text(node.find("MedlineDate"))
            match = re.search(r"\b(19|20)\d{2}\b", medline_date)
            year = match.group(0) if match else ""
        if not year.isdigit() or len(year) != 4:
            continue
        month = _element_text(node.find("Month"))
        day = _element_text(node.find("Day"))
        try:
            month_number = int(month) if month.isdigit() else datetime.strptime(month[:3], "%b").month
            day_number = int(day) if day.isdigit() else 1
            return date(int(year), month_number, day_number).isoformat()
        except (ValueError, TypeError):
            return f"{year}-01-01"
    return None


def parse_pubmed_xml(payload: bytes) -> list[dict[str, Any]]:
    root = ET.fromstring(payload)
    records = []
    for article in root.findall(".//PubmedArticle"):
        pmid = _element_text(article.find(".//PMID"))
        if not pmid.isdigit():
            continue
        article_node = article.find(".//Article")
        title = _element_text(article_node.find("ArticleTitle") if article_node is not None else None)
        abstract_node = article.find(".//Abstract")
        if not title or abstract_node is None:
            continue

        abstract_lines = []
        for section in abstract_node.findall("AbstractText"):
            text = _element_text(section)
            if text:
                label = section.attrib.get("Label") or section.attrib.get("NlmCategory")
                abstract_lines.append(f"{label}: {text}" if label else text)
        abstract = "\n".join(abstract_lines)
        if len(abstract) < 120:
            continue

        journal = _element_text(article.find(".//Journal/Title"))
        publication_types = [
            _element_text(node)
            for node in article.findall(".//PublicationTypeList/PublicationType")
            if _element_text(node)
        ]
        published = _published_date(article)
        records.append({
            "pmid": pmid,
            "title": title,
            "abstract": abstract,
            "journal": journal,
            "publication_types": publication_types,
            "published_date": published,
        })
    return records


def _passages(text: str) -> list[str]:
    cleaned = re.sub(r"\s+", " ", text).strip()
    return [cleaned] if cleaned else []


def records_to_chunks(records: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    documents: list[dict[str, Any]] = []
    chunks: list[dict[str, Any]] = []
    for record in records:
        pmid = record["pmid"]
        doc_id = f"PUBMED-{pmid}"
        source_url = f"https://pubmed.ncbi.nlm.nih.gov/{pmid}/"
        documents.append({
            "doc_id": doc_id,
            "title": record["title"],
            "source_type": "PubMed guideline/review abstract",
            "publisher": record["journal"] or "PubMed / NCBI",
            "version": pmid,
            "effective_date": record["published_date"] or "Not stated",
            "published_date": record["published_date"],
            "review_date": "Not stated",
            "status": "active",
            "allowed_roles": ["clinician"],
            "conflict_keys": [],
            "stance": "",
            "source_url": source_url,
            "publication_types": record["publication_types"],
        })
        for index, passage in enumerate(_passages(record["abstract"]), start=1):
            digest = hashlib.sha1(f"{pmid}|{index}|{passage}".encode("utf-8")).hexdigest()[:12]
            chunks.append({
                "id": f"PMID-{pmid}-{digest}",
                "doc_id": doc_id,
                "content": passage,
                "metadata": {
                    "doc_id": doc_id,
                    "title": record["title"],
                    "source_type": "PubMed guideline/review abstract",
                    "section": "Abstract",
                    "status": "active",
                    "published_date": record["published_date"],
                    "effective_date": record["published_date"],
                    "review_date": "Not stated",
                    "source_url": source_url,
                    "allowed_roles": ["clinician"],
                    "publication_types": record["publication_types"],
                },
            })
    return documents, chunks


def search_pubmed_abstracts(query: str, user_role: str) -> tuple[list[dict[str, Any]], str]:
    """Search only after explicit user opt-in; return abstract passages and status."""
    if user_role != "clinician":
        return [], "unsupported_role"
    if not query.strip() or len(query) > 1_000:
        return [], "invalid_query"
    if contains_identifier(query):
        return [], "identifier_blocked"
    if is_local_or_high_risk_query(query):
        return [], "local_or_high_risk_scope"

    term = build_search_term(query)
    if not term:
        return [], "no_search_terms"

    session = requests.Session()
    session.headers.update({"User-Agent": "VoughtCrop/1.0 (opt-in clinical evidence search)"})
    common = {"tool": "VoughtCrop"}
    contact = os.getenv("NCBI_EMAIL", "").strip()
    if contact:
        common["email"] = contact
    api_key = os.getenv("NCBI_API_KEY", "").strip()
    if api_key:
        common["api_key"] = api_key
    try:
        search_response = session.get(
            f"{EUTILS}/esearch.fcgi",
            params={**common, "db": "pubmed", "term": term, "retmode": "json", "retmax": MAX_RECORDS},
            timeout=(3, 10),
        )
        search_response.raise_for_status()
        identifiers = search_response.json().get("esearchresult", {}).get("idlist", [])
        identifiers = [value for value in identifiers[:MAX_RECORDS] if str(value).isdigit()]
        if not identifiers:
            return [], "no_results"

        fetch_response = session.get(
            f"{EUTILS}/efetch.fcgi",
            params={**common, "db": "pubmed", "id": ",".join(identifiers), "retmode": "xml"},
            timeout=(3, 15),
        )
        fetch_response.raise_for_status()
        if len(fetch_response.content) > MAX_RESPONSE_BYTES:
            return [], "response_too_large"
        records = parse_pubmed_xml(fetch_response.content)
        _, chunks = records_to_chunks(records)
        return chunks, "results" if chunks else "no_abstracts"
    finally:
        session.close()
