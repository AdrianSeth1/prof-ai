"""
semantic_scholar.py — Semantic Scholar Graph API search.
No auth required for basic usage. For higher rate limits, set SEMANTIC_SCHOLAR_API_KEY
to your key from https://www.semanticscholar.org/product/api — it will be sent as
the x-api-key header automatically.
"""

import json
import logging
import time
import urllib.error
import urllib.parse
import urllib.request
from datetime import date

API_BASE = "https://api.semanticscholar.org/graph/v1"
RESULTS_LIMIT = 20
RECENCY_YEARS = 2

# Optional: set to your API key string for higher rate limits. Empty = unauthenticated.
SEMANTIC_SCHOLAR_API_KEY = ""

_FIELDS = "title,abstract,authors,year,venue,externalIds,citationCount"
_cache: dict[str, list[dict]] = {}
logger = logging.getLogger(__name__)

# Exponential backoff delays (seconds) for 429 responses. 4 retries = 5 total attempts.
_RETRY_DELAYS = (2, 4, 8, 16)
# Minimum gap between consecutive outbound requests to avoid self-bursting.
_INTER_REQUEST_DELAY = 1.0
_last_request_time: float = 0.0


# ---------------------------------------------------------------------------
# HTTP helper
# ---------------------------------------------------------------------------

def _get(url: str, params: dict) -> bytes:
    global _last_request_time

    # Space out consecutive calls so we don't self-burst the shared pool.
    elapsed = time.time() - _last_request_time
    if elapsed < _INTER_REQUEST_DELAY:
        time.sleep(_INTER_REQUEST_DELAY - elapsed)

    full_url = url + "?" + urllib.parse.urlencode(params)
    req = urllib.request.Request(full_url)
    req.add_header("User-Agent", "Prof-AI/1.0 (academic research tool)")
    if SEMANTIC_SCHOLAR_API_KEY:
        req.add_header("x-api-key", SEMANTIC_SCHOLAR_API_KEY)

    for attempt in range(len(_RETRY_DELAYS) + 1):
        try:
            _last_request_time = time.time()
            with urllib.request.urlopen(req, timeout=15) as resp:
                return resp.read()
        except urllib.error.HTTPError as e:
            # Non-429 errors are always bugs — let them surface.
            if e.code != 429 or attempt == len(_RETRY_DELAYS):
                raise
            # Honor the server's own Retry-After if present; fall back to backoff schedule.
            try:
                wait = float(e.headers.get("Retry-After") or _RETRY_DELAYS[attempt])
            except ValueError:
                wait = _RETRY_DELAYS[attempt]
            print(
                f"[SemanticScholar] 429, retry {attempt + 1}/{len(_RETRY_DELAYS)} in {wait:.0f}s...",
                flush=True,
            )
            time.sleep(wait)

    # Unreachable, but satisfies the type checker.
    raise RuntimeError("_get: retry loop exited without returning")


# ---------------------------------------------------------------------------
# Parsing
# ---------------------------------------------------------------------------

def _parse_paper(paper: dict) -> dict:
    external_ids = paper.get("externalIds") or {}
    doi = external_ids.get("DOI") or None
    authors = [a["name"] for a in (paper.get("authors") or []) if a.get("name")]
    return {
        "id": paper.get("paperId", ""),
        "title": (paper.get("title") or "").strip(),
        "abstract": (paper.get("abstract") or "").strip(),
        "authors": authors,
        "year": paper.get("year"),
        "venue": (paper.get("venue") or "").strip(),
        "citation_count": paper.get("citationCount"),
        "doi": doi,
        "source": "semantic_scholar",
    }


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def search_semantic_scholar(
    query: str,
    max_results: int = 5,
    recency_years: int = RECENCY_YEARS,
) -> list[dict]:
    """Search Semantic Scholar and return a list of paper dicts.

    Returns [] on error (logged). On 429, retries up to 4 times with exponential
    backoff (2s, 4s, 8s, 16s), honoring Retry-After if the server sends one.
    """
    today = date.today()
    start_year = today.year - recency_years
    year_range = f"{start_year}-{today.year}"

    cache_key = f"{query}|{max_results}|{year_range}"
    if cache_key in _cache:
        return _cache[cache_key]

    print(f"[SemanticScholar] searching: {query!r} year={year_range}", flush=True)
    try:
        params = {
            "query": query,
            "year": year_range,
            "fields": _FIELDS,
            "limit": min(max_results, RESULTS_LIMIT),
        }
        data = json.loads(_get(f"{API_BASE}/paper/search", params))
        papers = data.get("data", [])
        results = [_parse_paper(p) for p in papers if p.get("paperId")]
        results = [r for r in results if r["title"] and r["abstract"]]
        print(f"[SemanticScholar] {len(results)} result(s)", flush=True)
        for r in results:
            print(f"  - [{r['id'][:8]}...] {r['title'][:80]} ({r['year']})", flush=True)
    except urllib.error.HTTPError as e:
        if e.code == 429:
            print(
                f"[SemanticScholar] rate limited, skipping after {len(_RETRY_DELAYS)} retries",
                flush=True,
            )
            return []
        logger.exception("[SemanticScholar] HTTP %d for query %r", e.code, query)
        return []
    except Exception:
        logger.exception("[SemanticScholar] search failed for query %r", query)
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
    print(f"Searching Semantic Scholar for: {q!r}\n")
    results = search_semantic_scholar(q, max_results=3)
    if not results:
        print("No results.")
    for r in results:
        author_str = ", ".join(r["authors"][:3]) + (" et al." if len(r["authors"]) > 3 else "")
        print(f"[{r['id']}] {r['title']}")
        print(f"  {r['venue']} | {r['year']} | citations: {r['citation_count']}")
        print(f"  {author_str}")
        print(f"  DOI: {r['doi']}")
        print(f"  {r['abstract'][:200]}{'...' if len(r['abstract']) > 200 else ''}\n")
