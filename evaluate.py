"""Run the local, reproducible retrieval safety benchmark."""

from __future__ import annotations

import json

from evaluation import run_evaluation
from ingest import load_corpus
from retrieve import HybridRetriever


def main() -> None:
    corpus = load_corpus()
    retriever = HybridRetriever(corpus["chunks"], corpus["documents"])
    report = run_evaluation(retriever)
    print(f"Mode: {report['retrieval_mode']}")
    for key, value in report["metrics"].items():
        print(f"{key}: {value}")
    print("\nCase results")
    for case in report["cases"]:
        found = ", ".join(case["expected_docs_found"]) or "none"
        print(f"{case['id']:<24} {case['decision']:<18} sources={case['source_count']} expected={found} conflict={case['conflict_detected']}")
    print("\nJSON report")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
