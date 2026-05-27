"""
pubmed.py — PubMed search via NCBI E-utilities.
No external dependencies beyond the standard library (except ollama for query reformulation).
"""

import json
import logging
import re
import urllib.parse
import urllib.request
import xml.etree.ElementTree as ET
from datetime import date, timedelta
from typing import Any

import ollama

# TODO: replace with a real email address for NCBI compliance
NCBI_EMAIL = "your-email@example.com"
NCBI_TOOL = "prof-ai-assistant"
REFORMULATE_MODEL = "qwen3:14b"
_THINK_RE = re.compile(r"<think>.*?</think>", re.DOTALL)

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

def reformulate_for_pubmed(
    user_question: str,
    conversation_history: str,
    selected_doc_titles: list[str],
) -> str | None:
    """Rewrite a natural-language question into a PubMed-optimised search query.

    Returns None if the model produces an unusable result (caller should skip PubMed).
    """
    titles_str = ", ".join(selected_doc_titles) if selected_doc_titles else "none"
    prompt = (
        "You convert a researcher's question into a PubMed search query. "
        "Return ONLY the search query string. No explanation. No quotes. "
        "Do NOT include date filters - those are added by Python separately.\n\n"
        "GUIDELINES:\n"
        "- Use 2-3 concept groups joined by AND. More than that returns too few results.\n"
        "- For broad exploratory questions like 'what's new', 'what would extend this', "
        "or 'what recent research', prefer BROADER queries. The main topic plus a recency "
        "hint like (novel OR emerging OR recent) is often enough.\n"
        "- For specific mechanistic questions, you can be more targeted.\n"
        "- Do NOT invent constraints. Do not add age groups, populations, geographic "
        "limits, or other filters that weren't in the question or the source materials.\n"
        "- Use OR within a concept group to catch synonyms. MeSH terms are good when obvious.\n\n"
        f"Selected source materials: {titles_str}\n"
        f"Recent conversation: {conversation_history}\n"
        f"Current question: {user_question}\n\n"
        "EXAMPLES:\n\n"
        "Question: 'What recent research extends the dopamine hypothesis?'\n"
        "Good: (dopamine hypothesis OR dopaminergic dysfunction) AND (schizophrenia OR psychosis) AND (novel OR recent)\n\n"
        "Question: 'Are there new genetic findings for schizophrenia?'\n"
        "Good: schizophrenia AND (genetic OR genomic OR GWAS) AND (novel OR emerging)\n\n"
        "Question: 'What research would make these schizophrenia slides more interesting?'\n"
        "Good: schizophrenia AND (novel OR emerging OR breakthrough) AND (mechanism OR treatment OR pathophysiology)\n\n"
        "BAD (too narrow, invented constraints): schizophrenia AND adolescent AND treatment AND antipsychotics AND clinical trial\n\n"
        "Search query:"
    )
    try:
        response = ollama.chat(
            model=REFORMULATE_MODEL,
            messages=[{"role": "user", "content": prompt}],
        )
        raw = _THINK_RE.sub("", response["message"]["content"]).strip().strip("\"'")
        if len(raw) < 5:
            logger.warning("PubMed reformulator returned too-short result: %r", raw)
            return None
        print(f"[PubMed] reformulated query: {raw}", flush=True)
        return raw
    except Exception:
        logger.exception("PubMed reformulator failed for question %r", user_question)
        return None


def search_pubmed(
    query: str,
    max_results: int = 5,
    recency_years: int = 2,
) -> list[dict]:
    """Search PubMed and return a list of article dicts.

    Results are cached in memory for the lifetime of the process.
    Returns [] on network or parse errors (logged to stderr).
    """
    today = date.today()
    min_date = (today - timedelta(days=recency_years * 365)).strftime("%Y/%m/%d")
    max_date = today.strftime("%Y/%m/%d")

    cache_key = f"{query}|{max_results}|{min_date}"
    if cache_key in _cache:
        return _cache[cache_key]

    print(f"[PubMed] date range: {min_date} to {max_date}", flush=True)

    try:
        pmids = _esearch(query, max_results, min_date, max_date)
        print(f"[PubMed] esearch returned {len(pmids)} PMIDs: {pmids}", flush=True)
        if not pmids:
            _cache[cache_key] = []
            return []

        root = _efetch(pmids)
        results = [
            parsed
            for article_el in root.findall("PubmedArticle")
            if (parsed := _parse_article(article_el)) is not None
        ]
        print(f"[PubMed] efetched {len(results)} abstracts", flush=True)
        for a in results:
            print(f"  - [{a['pmid']}] {a['title'][:80]}... ({a['pub_date']})", flush=True)
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
