"""
openalex.py — OpenAlex API search (open, no auth required).
Setting OPENALEX_MAILTO to an email address opts into OpenAlex's "polite pool" —
better rate limits, and they can contact you if there's an API issue. Recommended.
"""

import json
import logging
import urllib.parse
import urllib.request
from datetime import date

API_BASE = "https://api.openalex.org"
RESULTS_LIMIT = 20
RECENCY_YEARS = 2

# Set to an email address to join the polite pool (recommended, no auth required).
OPENALEX_MAILTO = ""

_SELECT_FIELDS = (
    "id,display_name,abstract_inverted_index,authorships,"
    "publication_year,primary_location,cited_by_count,doi"
)
_cache: dict[str, list[dict]] = {}
logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# HTTP helper
# ---------------------------------------------------------------------------

def _get(url: str, params: dict) -> bytes:
    if OPENALEX_MAILTO:
        params = {**params, "mailto": OPENALEX_MAILTO}
    full_url = url + "?" + urllib.parse.urlencode(params)
    with urllib.request.urlopen(full_url, timeout=15) as resp:
        return resp.read()


# ---------------------------------------------------------------------------
# Abstract reconstruction
# ---------------------------------------------------------------------------

def _reconstruct_abstract(inverted_index: dict | None) -> str:
    """Reconstruct plain text from OpenAlex's inverted index format.

    The index maps each word to a list of positions it appears at.
    We reverse the mapping, sort by position, and join.
    """
    if not inverted_index:
        return ""
    position_word: dict[int, str] = {}
    for word, positions in inverted_index.items():
        for pos in positions:
            position_word[pos] = word
    if not position_word:
        return ""
    return " ".join(position_word[i] for i in sorted(position_word))


# ---------------------------------------------------------------------------
# Parsing
# ---------------------------------------------------------------------------

def _parse_work(work: dict) -> dict:
    # Strip URL prefix from OpenAlex ID: "https://openalex.org/W123" → "W123"
    raw_id = work.get("id", "")
    work_id = raw_id.rsplit("/", 1)[-1] if raw_id else ""

    # Strip DOI URL prefix: "https://doi.org/10.1234/..." → "10.1234/..."
    raw_doi = work.get("doi") or ""
    doi = raw_doi.replace("https://doi.org/", "").strip() or None

    authors = [
        auth["author"]["display_name"]
        for auth in (work.get("authorships") or [])
        if (auth.get("author") or {}).get("display_name")
    ]

    venue = ""
    primary_loc = work.get("primary_location") or {}
    source_info = primary_loc.get("source") or {}
    if source_info:
        venue = source_info.get("display_name", "")

    abstract = _reconstruct_abstract(work.get("abstract_inverted_index"))

    return {
        "id": work_id,
        "title": (work.get("display_name") or "").strip(),
        "abstract": abstract.strip(),
        "authors": authors,
        "year": work.get("publication_year"),
        "venue": venue.strip(),
        "citation_count": work.get("cited_by_count"),
        "doi": doi,
        "source": "openalex",
    }


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def search_openalex(
    query: str,
    max_results: int = 5,
    recency_years: int = RECENCY_YEARS,
) -> list[dict]:
    """Search OpenAlex and return a list of work dicts.

    Returns [] on error (logged).
    """
    today = date.today()
    # publication_year:>2022 means 2023 and later
    start_year = today.year - recency_years
    year_filter = f"publication_year:>{start_year - 1}"

    cache_key = f"{query}|{max_results}|{start_year}"
    if cache_key in _cache:
        return _cache[cache_key]

    print(f"[OpenAlex] searching: {query!r} filter={year_filter}", flush=True)
    try:
        params = {
            "search": query,
            "filter": year_filter,
            "per-page": min(max_results, RESULTS_LIMIT),
            "select": _SELECT_FIELDS,
        }
        data = json.loads(_get(f"{API_BASE}/works", params))
        works = data.get("results", [])
        results = [_parse_work(w) for w in works if w.get("id")]
        results = [r for r in results if r["title"]]
        print(f"[OpenAlex] {len(results)} result(s)", flush=True)
        for r in results:
            print(f"  - [{r['id']}] {r['title'][:80]} ({r['year']})", flush=True)
    except Exception:
        logger.exception("[OpenAlex] search failed for query %r", query)
        return []

    _cache[cache_key] = results
    return results


# ---------------------------------------------------------------------------
# Standalone test
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import sys
    logging.basicConfig(level=logging.INFO)
    q = " ".join(sys.argv[1:]) if len(sys.argv) > 1 else "agenda-setting theory media effects"
    print(f"Searching OpenAlex for: {q!r}\n")
    results = search_openalex(q, max_results=3)
    if not results:
        print("No results.")
    for r in results:
        author_str = ", ".join(r["authors"][:3]) + (" et al." if len(r["authors"]) > 3 else "")
        print(f"[{r['id']}] {r['title']}")
        print(f"  {r['venue']} | {r['year']} | citations: {r['citation_count']}")
        print(f"  {author_str}")
        print(f"  DOI: {r['doi']}")
        print(f"  {r['abstract'][:200]}{'...' if len(r['abstract']) > 200 else ''}\n")
