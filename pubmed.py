"""
pubmed.py — PubMed search via NCBI E-utilities.
No external dependencies beyond the standard library.
"""

import json
import logging
import urllib.parse
import urllib.request
import xml.etree.ElementTree as ET
from datetime import date
from typing import Any

# TODO: replace with a real email address for NCBI compliance
NCBI_EMAIL = "your-email@example.com"
NCBI_TOOL = "prof-ai-assistant"

ESEARCH_URL = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esearch.fcgi"
EFETCH_URL  = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/efetch.fcgi"

_cache: dict[str, list[dict]] = {}
logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# HTTP helpers
# ---------------------------------------------------------------------------

def _base_params(**kwargs: Any) -> dict:
    return {"tool": NCBI_TOOL, "email": NCBI_EMAIL, **kwargs}


def _get(url: str, params: dict) -> bytes:
    full_url = url + "?" + urllib.parse.urlencode(params)
    with urllib.request.urlopen(full_url, timeout=15) as resp:
        return resp.read()


# ---------------------------------------------------------------------------
# E-utilities calls
# ---------------------------------------------------------------------------

def _esearch(query: str, max_results: int, min_date: str, max_date: str) -> list[str]:
    params = _base_params(
        db="pubmed",
        term=query,
        retmax=max_results,
        sort="date",
        mindate=min_date,
        maxdate=max_date,
        datetype="pdat",
        retmode="json",
    )
    data = json.loads(_get(ESEARCH_URL, params))
    return data.get("esearchresult", {}).get("idlist", [])


def _efetch(pmids: list[str]) -> ET.Element:
    params = _base_params(
        db="pubmed",
        id=",".join(pmids),
        rettype="abstract",
        retmode="xml",
    )
    return ET.fromstring(_get(EFETCH_URL, params))


# ---------------------------------------------------------------------------
# XML parsing
# ---------------------------------------------------------------------------

def _parse_article(article_el: ET.Element) -> dict | None:
    medline = article_el.find("MedlineCitation")
    if medline is None:
        return None

    pmid_el = medline.find("PMID")
    pmid = (pmid_el.text or "").strip() if pmid_el is not None else ""
    if not pmid:
        return None

    art = medline.find("Article")
    if art is None:
        return None

    # Title
    title_el = art.find("ArticleTitle")
    title = (title_el.text or "").strip() if title_el is not None else ""

    # Abstract — may have multiple labelled sections (BACKGROUND, METHODS, …)
    abstract_parts: list[str] = []
    abstract_el = art.find("Abstract")
    if abstract_el is not None:
        for text_el in abstract_el.findall("AbstractText"):
            label = text_el.get("Label")
            body = text_el.text or ""
            abstract_parts.append(f"{label}: {body}" if label else body)
    abstract = " ".join(abstract_parts).strip()

    # Authors (up to all; caller can truncate for display)
    authors: list[str] = []
    author_list = art.find("AuthorList")
    if author_list is not None:
        for author in author_list.findall("Author"):
            last = author.findtext("LastName", "")
            fore = author.findtext("ForeName", "")
            if last:
                authors.append(f"{last} {fore}".strip())

    # Journal + publication date
    journal = ""
    pub_date = ""
    journal_el = art.find("Journal")
    if journal_el is not None:
        journal = journal_el.findtext("Title", "")
        issue = journal_el.find("JournalIssue")
        if issue is not None:
            pd = issue.find("PubDate")
            if pd is not None:
                med = pd.findtext("MedlineDate", "")
                if med:
                    pub_date = med
                else:
                    parts = filter(None, [
                        pd.findtext("Year", ""),
                        pd.findtext("Month", ""),
                        pd.findtext("Day", ""),
                    ])
                    pub_date = " ".join(parts)

    return {
        "pmid":     pmid,
        "title":    title,
        "abstract": abstract,
        "authors":  authors,
        "journal":  journal,
        "pub_date": pub_date,
        "url":      f"https://pubmed.ncbi.nlm.nih.gov/{pmid}/",
    }


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def search_pubmed(
    query: str,
    max_results: int = 5,
    recency_years: int = 2,
) -> list[dict]:
    """Search PubMed and return a list of article dicts.

    Results are cached in memory for the lifetime of the process.
    Returns [] on network or parse errors (logged to stderr).
    """
    cache_key = f"{query}|{max_results}|{recency_years}"
    if cache_key in _cache:
        return _cache[cache_key]

    today = date.today()
    try:
        min_date = today.replace(year=today.year - recency_years).strftime("%Y/%m/%d")
    except ValueError:
        # Feb 29 edge case on non-leap years
        min_date = today.replace(year=today.year - recency_years, day=28).strftime("%Y/%m/%d")
    max_date = today.strftime("%Y/%m/%d")

    try:
        pmids = _esearch(query, max_results, min_date, max_date)
        if not pmids:
            _cache[cache_key] = []
            return []

        root = _efetch(pmids)
        results = [
            parsed
            for article_el in root.findall("PubmedArticle")
            if (parsed := _parse_article(article_el)) is not None
        ]
    except Exception:
        logger.exception("PubMed search failed for query %r", query)
        return []

    _cache[cache_key] = results
    return results


# ---------------------------------------------------------------------------
# Standalone test
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import sys

    logging.basicConfig(level=logging.INFO)
    q = " ".join(sys.argv[1:]) if len(sys.argv) > 1 else "hippocampus memory consolidation"
    print(f"Searching PubMed for: {q!r}\n")
    articles = search_pubmed(q, max_results=3)

    if not articles:
        print("No results.")
    for a in articles:
        author_str = ", ".join(a["authors"][:3])
        if len(a["authors"]) > 3:
            author_str += " et al."
        print(f"[{a['pmid']}] {a['title']}")
        print(f"  {a['journal']} | {a['pub_date']}")
        print(f"  {author_str}")
        print(f"  {a['abstract'][:200]}{'...' if len(a['abstract']) > 200 else ''}")
        print(f"  {a['url']}\n")
