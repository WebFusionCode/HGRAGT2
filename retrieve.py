"""Access-filtered lexical and vector retrieval with freshness-aware reranking."""

from __future__ import annotations

import math
import os
import re
from collections import Counter, defaultdict
from datetime import date
from typing import Any

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


_TOKEN_RE = re.compile(r"[a-z0-9]+", re.I)
_STOP_WORDS = {
    "a", "an", "and", "are", "as", "at", "be", "been", "before", "can", "could", "do", "does", "for", "from", "give", "how", "i", "in", "is", "it", "me", "of", "on", "or", "please", "should", "tell", "that", "the", "their", "them", "this", "to", "use", "was", "what", "when", "where", "which", "who", "with", "would",
}
_SYNONYMS = {
    "assisted": ["assistance", "help"],
    "activities": ["activity", "physical activity", "daily activities", "exercise"],
    "activity": ["activities", "physical activity", "daily activities", "exercise"],
    "checks": ["check", "inspection"],
    "concussion": ["mild traumatic brain injury", "mTBI", "head injury"],
    "dirty": ["contamination", "visible soil"],
    "soiled": ["contamination", "visible soil"],
    "contaminated": ["contamination", "visible soil"],
    "clean": ["cleaning", "disinfection"],
    "cleaned": ["cleaning", "disinfection"],
    "wipe": ["wipe", "cleaning"],
    "reuse": ["reuse", "return to service"],
    "sling": ["transfer sling"],
    "coverage": ["covered", "payer", "prior authorization"],
    "authorization": ["prior authorization"],
    "require": ["required", "requirement", "yes"],
    "inventory": ["asset", "equipment"],
    "quantity": ["count", "available"],
    "count": ["quantity", "available"],
    "located": ["location"],
    "limited": ["limit", "restrict", "avoid"],
    "limit": ["limited", "restrict", "avoid"],
    "lifts": ["lift"],
    "initial": ["first", "early", "beginning"],
    "period": ["duration", "hours", "days"],
    "possible": ["suspected", "potential"],
    "recommended": ["recommend", "recommendation", "advised", "suggest"],
    "rest": ["relative rest", "physical and cognitive rest", "reduced activity"],
}


def _query_term_covered(token: str, matched_tokens: set[str]) -> bool:
    variants = {token}
    if token.endswith("ies") and len(token) > 4:
        variants.add(token[:-3] + "y")
    if token.endswith("s") and len(token) > 4:
        variants.add(token[:-1])
    if token.endswith("ed") and len(token) > 4:
        variants.add(token[:-2])
    if token.endswith("ing") and len(token) > 5:
        variants.add(token[:-3])
    variants.update(tokenize(" ".join(_SYNONYMS.get(token, []))))
    return bool(variants & matched_tokens)


def tokenize(text: str) -> list[str]:
    return [token.lower() for token in _TOKEN_RE.findall(text) if token.lower() not in _STOP_WORDS and len(token) > 1]


def _bm25(query_tokens: list[str], docs: list[list[str]]) -> list[float]:
    if not query_tokens or not docs:
        return [0.0] * len(docs)
    lengths = [len(doc) for doc in docs]
    avg_length = sum(lengths) / max(len(lengths), 1)
    doc_freq = Counter(token for token in set(query_tokens) for doc in docs if token in set(doc))
    scores = []
    for tokens, length in zip(docs, lengths):
        counts = Counter(tokens)
        score = 0.0
        for term in set(query_tokens):
            frequency = counts[term]
            if not frequency:
                continue
            idf = math.log(1 + (len(docs) - doc_freq[term] + 0.5) / (doc_freq[term] + 0.5))
            denominator = frequency + 1.5 * (1 - 0.75 + 0.75 * length / max(avg_length, 1))
            score += idf * frequency * 2.5 / denominator
        scores.append(score)
    return scores


class HybridRetriever:
    """Build role-specific candidate sets before any retrieval signal is computed."""

    def __init__(self, chunks: list[dict[str, Any]], documents: list[dict[str, Any]] | None = None):
        self.chunks = chunks
        self.documents = documents or []
        self.neural_model = None
        self.neural_vectors = None
        self.neural_error: str | None = None
        self._load_optional_neural_encoder()

    def _load_optional_neural_encoder(self) -> None:
        if os.getenv("ENABLE_NEURAL_EMBEDDINGS", "0").lower() not in {"1", "true", "yes"}:
            return
        try:
            from sentence_transformers import SentenceTransformer

            model_name = os.getenv("SENTENCE_TRANSFORMER_MODEL", "all-MiniLM-L6-v2")
            self.neural_model = SentenceTransformer(model_name, device="cpu")
            self.neural_vectors = self.neural_model.encode([chunk["content"] for chunk in self.chunks], normalize_embeddings=True)
        except Exception as exc:
            self.neural_error = type(exc).__name__
            self.neural_model = None
            self.neural_vectors = None

    @property
    def retrieval_mode(self) -> str:
        if self.neural_model is not None:
            return "BM25 + neural vectors + RRF rerank"
        return "BM25 + TF-IDF vectors + RRF rerank"

    def _visible_chunks(self, role: str) -> list[dict[str, Any]]:
        return [chunk for chunk in self.chunks if role in chunk["metadata"].get("allowed_roles", [])]

    def search(self, query: str, user_role: str, top_k: int = 6) -> dict[str, Any]:
        visible = self._visible_chunks(user_role)
        if not visible or not query.strip():
            return {"sources": [], "conflicts": [], "coverage": 0.0, "matched_query_terms": 0, "missing_evidence": [], "retrieval_mode": self.retrieval_mode}

        query_tokens = tokenize(query)
        expanded_tokens = list(query_tokens)
        for token in query_tokens:
            for expansion in _SYNONYMS.get(token, []):
                expanded_tokens.extend(tokenize(expansion))
        expanded_tokens = list(dict.fromkeys(expanded_tokens))
        doc_tokens = [tokenize(chunk["content"]) for chunk in visible]
        bm25_scores = _bm25(expanded_tokens, doc_tokens)

        if self.neural_model is not None and self.neural_vectors is not None:
            import numpy as np

            query_vector = self.neural_model.encode([query], normalize_embeddings=True)[0]
            source_positions = {chunk["id"]: index for index, chunk in enumerate(self.chunks)}
            dense_scores = [float(np.dot(query_vector, self.neural_vectors[source_positions[chunk["id"]]])) for chunk in visible]
        else:
            vectorizer = TfidfVectorizer(ngram_range=(1, 2), sublinear_tf=True, strip_accents="unicode")
            try:
                matrix = vectorizer.fit_transform([chunk["content"] for chunk in visible] + [query])
                dense_scores = cosine_similarity(matrix[-1], matrix[:-1]).ravel().tolist()
            except ValueError:
                dense_scores = [0.0] * len(visible)

        bm25_order = sorted(range(len(visible)), key=lambda index: bm25_scores[index], reverse=True)
        dense_order = sorted(range(len(visible)), key=lambda index: dense_scores[index], reverse=True)
        bm25_ranks = {index: rank for rank, index in enumerate(bm25_order)}
        dense_ranks = {index: rank for rank, index in enumerate(dense_order)}
        rrf = {index: 1 / (60 + bm25_ranks[index] + 1) + 1 / (60 + dense_ranks[index] + 1) for index in range(len(visible))}
        max_bm25 = max(bm25_scores, default=0.0) or 1.0
        max_rrf = max(rrf.values(), default=1.0) or 1.0
        ranked: list[dict[str, Any]] = []

        for index, chunk in enumerate(visible):
            if bm25_scores[index] <= 0 and dense_scores[index] <= 0:
                continue
            metadata = chunk["metadata"]
            status = metadata.get("status", "active")
            recency_adjustment = 0.035 if status == "active" else (-0.08 if status == "superseded" else -0.2)
            score = (
                0.45 * (bm25_scores[index] / max_bm25)
                + 0.35 * max(dense_scores[index], 0.0)
                + 0.20 * (rrf[index] / max_rrf)
                + recency_adjustment
            )
            ranked.append({
                **chunk,
                "score": round(max(score, 0.0), 4),
                "bm25_score": round(bm25_scores[index], 4),
                "vector_score": round(float(dense_scores[index]), 4),
                "freshness_status": self._freshness_status(metadata),
            })

        ranked.sort(key=lambda item: item["score"], reverse=True)
        top_sources = [item for item in ranked if item["score"] >= 0.08][:top_k]
        matched_tokens = set(token for item in top_sources[:top_k] for token in tokenize(item["content"]))
        relevant_query_tokens = [token for token in query_tokens if len(token) > 2]
        matched_query_terms = sum(_query_term_covered(token, matched_tokens) for token in relevant_query_tokens)
        coverage = matched_query_terms / max(len(relevant_query_tokens), 1)
        all_accessible_tokens = {token for doc in doc_tokens for token in doc}
        missing = [token for token in relevant_query_tokens if not _query_term_covered(token, all_accessible_tokens)]
        unmatched_specific_terms = [
            token for token in relevant_query_tokens
            if len(token) >= 7 and not _query_term_covered(token, matched_tokens)
        ]
        conflicts = self._detect_conflicts(top_sources)

        return {
            "sources": top_sources,
            "conflicts": conflicts,
            "coverage": round(coverage, 3),
            "matched_query_terms": matched_query_terms,
            "unmatched_specific_terms": unmatched_specific_terms,
            "missing_evidence": list(dict.fromkeys(missing))[:8],
            "retrieval_mode": self.retrieval_mode,
        }

    @staticmethod
    def _freshness_status(metadata: dict[str, Any]) -> str:
        if metadata.get("status") != "active":
            return metadata.get("status", "unknown")
        review_date = metadata.get("review_date")
        if review_date:
            try:
                if date.fromisoformat(review_date) < date.today():
                    return "review due"
            except ValueError:
                pass
        published_date = metadata.get("published_date")
        if published_date:
            try:
                published = date.fromisoformat(published_date[:10])
                if (date.today() - published).days > 730:
                    return "review due"
            except (TypeError, ValueError):
                pass
        return "current"

    @staticmethod
    def _detect_conflicts(sources: list[dict[str, Any]]) -> list[dict[str, Any]]:
        groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
        for source in sources:
            for key in source["metadata"].get("conflict_keys", []):
                if source["metadata"].get("stance"):
                    groups[key].append(source)
        conflicts = []
        for key, passages in groups.items():
            stances = {item["metadata"].get("stance") for item in passages}
            if len(stances) < 2:
                continue
            passages.sort(key=lambda item: (item["metadata"].get("status") == "active", item["metadata"].get("effective_date", "")), reverse=True)
            conflicts.append({
                "topic": key.replace("_", " "),
                "summary": "Retrieved sources state incompatible instructions. The older source is shown as historical context; follow the current policy owner for direction.",
                "passages": [
                    {"id": item["id"], "doc_id": item["doc_id"], "title": item["metadata"].get("title", "Source"), "status": item["metadata"].get("status"), "effective_date": item["metadata"].get("effective_date"), "stance": item["metadata"].get("stance"), "content": item["content"]}
                    for item in passages
                ],
            })
        return conflicts

    def catalog(self, user_role: str) -> list[dict[str, Any]]:
        counts: Counter[str] = Counter(chunk["doc_id"] for chunk in self._visible_chunks(user_role))
        docs = [doc for doc in self.documents if user_role in doc.get("allowed_roles", [])]
        return [
            {**doc, "chunk_count": counts.get(doc["doc_id"], 0), "freshness_status": self._freshness_status(doc)}
            for doc in docs
        ]
