from __future__ import annotations

import unittest
from unittest.mock import patch

from pubmed_search import parse_pubmed_xml, records_to_chunks, search_pubmed_abstracts


PUBMED_XML = b"""<PubmedArticleSet>
  <PubmedArticle>
    <MedlineCitation>
      <PMID>12345678</PMID>
      <Article>
        <Journal><Title>Test Journal</Title><JournalIssue><PubDate>
          <Year>2024</Year><Month>Feb</Month><Day>2</Day>
        </PubDate></JournalIssue></Journal>
        <ArticleTitle>Asthma treatment guideline example</ArticleTitle>
        <Abstract>
          <AbstractText Label="BACKGROUND">This sufficiently long abstract describes a clinical practice guideline and its target population in cautious, clearly stated terms.</AbstractText>
          <AbstractText Label="CONCLUSIONS">The evidence supports this recommendation, while noting limitations and the need to follow current organizational policy.</AbstractText>
        </Abstract>
        <PublicationTypeList><PublicationType>Practice Guideline</PublicationType></PublicationTypeList>
      </Article>
    </MedlineCitation>
  </PubmedArticle>
</PubmedArticleSet>"""


class FakeResponse:
    def __init__(self, payload: bytes | None = None, data: dict | None = None):
        self.content = payload or b""
        self._data = data or {}

    def raise_for_status(self) -> None:
        return None

    def json(self) -> dict:
        return self._data


class FakeSession:
    def __init__(self):
        self.headers = {}
        self.urls: list[str] = []
        self.closed = False

    def get(self, url: str, **kwargs):
        self.urls.append(url)
        if "esearch.fcgi" in url:
            return FakeResponse(data={"esearchresult": {"idlist": ["12345678"]}})
        return FakeResponse(payload=PUBMED_XML)

    def close(self) -> None:
        self.closed = True


class PubMedSearchTests(unittest.TestCase):
    def test_xml_parser_preserves_abstract_citation_and_date(self):
        records = parse_pubmed_xml(PUBMED_XML)
        self.assertEqual(records[0]["published_date"], "2024-02-02")
        self.assertIn("BACKGROUND:", records[0]["abstract"])
        self.assertEqual(records[0]["publication_types"], ["Practice Guideline"])

    def test_search_returns_clinician_scoped_citable_passage(self):
        fake_session = FakeSession()
        with patch("pubmed_search.requests.Session", return_value=fake_session):
            chunks, status = search_pubmed_abstracts("What is the asthma treatment guideline?", "clinician")

        self.assertEqual(status, "results")
        self.assertEqual(len(chunks), 1)
        self.assertEqual(chunks[0]["metadata"]["allowed_roles"], ["clinician"])
        self.assertEqual(chunks[0]["metadata"]["published_date"], "2024-02-02")
        self.assertEqual(chunks[0]["metadata"]["source_url"], "https://pubmed.ncbi.nlm.nih.gov/12345678/")
        self.assertEqual(len(fake_session.urls), 2)
        self.assertTrue(fake_session.closed)

    def test_role_and_identifier_gates_avoid_network(self):
        with patch("pubmed_search.requests.Session") as session_factory:
            self.assertEqual(search_pubmed_abstracts("asthma treatment", "patient"), ([], "unsupported_role"))
            self.assertEqual(
                search_pubmed_abstracts("MRN: RAG-TEST-1234 asthma treatment", "clinician"),
                ([], "identifier_blocked"),
            )
            self.assertEqual(
                search_pubmed_abstracts("Is this medication approved on our formulary?", "clinician"),
                ([], "local_or_high_risk_scope"),
            )
        session_factory.assert_not_called()

    def test_abstract_passage_is_not_cut_mid_sentence(self):
        abstract = "A complete sentence. " + ("Evidence text " * 180) + "ends with a complete conclusion."
        _, chunks = records_to_chunks([{
            "pmid": "87654321",
            "title": "Long abstract",
            "abstract": abstract,
            "journal": "Test Journal",
            "publication_types": ["Review"],
            "published_date": "2024-01-01",
        }])
        self.assertEqual(len(chunks), 1)
        self.assertTrue(chunks[0]["content"].endswith("complete conclusion."))


if __name__ == "__main__":
    unittest.main()
