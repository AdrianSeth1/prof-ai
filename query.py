"""
query.py — ask a question against ingested course materials.
Usage:
  python query.py "question"
  python query.py --latest "question"
  python query.py --session 2026-05-21_143000 "question"
"""

import argparse
import logging
import re
import sys
import time
from concurrent.futures import ThreadPoolExecutor, TimeoutError as FuturesTimeoutError, as_completed
from pathlib import Path
from typing import Callable, Iterator

import chromadb
import ollama

from manifest import load_manifest
from pubmed import search_pubmed
from semantic_scholar import search_semantic_scholar
from openalex import search_openalex

CHROMA_DIR = Path("chroma_db")
SESSIONS_MANIFEST = Path("sessions.json")
COLLECTION_NAME = "course_material"
EMBED_MODEL = "nomic-embed-text"
LLM_MODEL = "qwen3:30b-a3b"
REFORMULATE_MODEL = "qwen3:30b-a3b"
TOP_K = 6
_SOURCE_TIMEOUT = 8  # seconds: max wall-clock time per literature source before it's abandoned

_THINK_RE = re.compile(r"<think>.*?</think>", re.DOTALL)
_THINK_OPEN = "<think>"
_THINK_CLOSE = "</think>"
logger = logging.getLogger(__name__)

SYSTEM_PROMPT = (
    "You are a research assistant helping a professor. Answer based on the provided "
    "course materials. If the materials don't contain the answer, say so plainly rather "
    "than guessing. When you make a claim, mention which source file it came from."
)

SYSTEM_PROMPT_WITH_LITERATURE = (
    "You are a research assistant helping a professor. You have been given both the "
    "professor's course materials and recent academic literature from one or more databases. "
    "Cite sources inline: [Doc: <filename>] for course materials, [PubMed: <PMID>] for PubMed, "
    "[Semantic Scholar: <ID>] for Semantic Scholar, [OpenAlex: <ID>] for OpenAlex. "
    "Distinguish clearly between content from the professor's materials and recent literature. "
    "When literature contradicts or extends the course materials, note it explicitly. "
    "When recent literature is provided, integrate it into your answer — the user has "
    "explicitly requested current research. Do not refuse to answer just because the "
    "course materials are older. Use the literature to extend or update what the slides "
    "cover. Always cite which source you drew from. "
    "If neither source answers the question, say so plainly rather than guessing."
)


# ---------------------------------------------------------------------------
# Shared query reformulation (works for all three literature sources)
# ---------------------------------------------------------------------------

def reformulate_for_search(
    user_question: str,
    conversation_history: str,
    selected_doc_titles: list[str],
) -> str | None:
    """Rewrite a natural-language question into a broad academic keyword search query.

    Returns None if the model produces an unusable result.
    """
    titles_str = ", ".join(selected_doc_titles) if selected_doc_titles else "none"
    prompt = (
        "You convert a researcher's question into a broad academic keyword search query. "
        "Return ONLY the search query string. No explanation. No quotes. "
        "Do NOT include date filters — those are added separately.\n\n"
        "GUIDELINES:\n"
        "- Use 2-3 concept groups joined by AND.\n"
        "- For exploratory questions, prefer broader queries with OR synonyms.\n"
        "- Do not invent constraints not present in the question.\n"
        "- Boolean syntax (AND, OR, parentheses) is fine.\n\n"
        f"Selected source materials: {titles_str}\n"
        f"Recent conversation: {conversation_history}\n"
        f"Question: {user_question}\n\n"
        "Search query:"
    )
    try:
        t0 = time.time()
        response = ollama.chat(
            model=REFORMULATE_MODEL,
            messages=[{"role": "user", "content": prompt}],
            think=False,
            keep_alive="30m",
        )
        print(f"[Literature] llm {time.time() - t0:.1f}s", flush=True)
        raw = _THINK_RE.sub("", response["message"]["content"]).strip().strip("\"'")
        if len(raw) < 5:
            logger.warning("Search reformulator returned too-short result: %r", raw)
            return None
        return raw
    except Exception:
        logger.exception("Search reformulator failed for question %r", user_question)
        return None


# ---------------------------------------------------------------------------
# Multi-source literature search
# ---------------------------------------------------------------------------

def _dedupe_literature(results: list[dict]) -> list[dict]:
    """Remove duplicates by DOI first, then by normalized title."""
    seen_dois: set[str] = set()
    seen_titles: set[str] = set()
    deduped = []
    for r in results:
        doi = r.get("doi")
        if doi and doi in seen_dois:
            continue
        title_norm = re.sub(r"\W+", " ", (r.get("title") or "").lower()).strip()
        if title_norm in seen_titles:
            continue
        if doi:
            seen_dois.add(doi)
        if title_norm:
            seen_titles.add(title_norm)
        deduped.append(r)
    return deduped


def build_literature_context(articles: list[dict]) -> str:
    """Build a context block for the LLM from a mixed list of literature results."""
    blocks = []
    for a in articles:
        source = a.get("source", "")
        authors = a.get("authors", [])
        author_str = ", ".join(authors[:5]) + (" et al." if len(authors) > 5 else "")
        abstract = a.get("abstract") or "No abstract available."

        if source == "pubmed":
            label = f"[PubMed: {a.get('pmid', '')}]"
            venue_str = f"{a.get('journal', '')} | {a.get('pub_date', '')}"
        elif source == "semantic_scholar":
            label = f"[Semantic Scholar: {a.get('id', '')}]"
            venue_str = f"{a.get('venue', '') or 'unknown venue'} | {a.get('year', '')}"
        elif source == "openalex":
            label = f"[OpenAlex: {a.get('id', '')}]"
            venue_str = f"{a.get('venue', '') or 'unknown venue'} | {a.get('year', '')}"
        else:
            label = f"[Literature: {a.get('id', '?')}]"
            venue_str = str(a.get("year", ""))

        blocks.append(
            f"{label} {a.get('title', '')} ({venue_str})\n"
            f"Authors: {author_str}\n"
            f"Abstract: {abstract}"
        )
    return "\n\n".join(blocks)


def search_literature(
    question: str,
    sources: list[str],
    conversation_history: str = "",
    selected_doc_titles: list[str] | None = None,
    max_results: int = 5,
    on_status: Callable[[str], None] | None = None,
) -> list[dict]:
    """Search selected literature sources in parallel and return deduplicated results.

    sources: any subset of ["pubmed", "semantic_scholar", "openalex"].
    Returns [] when sources is empty or all searches fail.

    on_status, if given, is called with "reformulating" before the query-reformulation
    LLM call and "searching_literature" once reformulation succeeds and the parallel
    per-source fetch is about to start. Callers that don't need pipeline progress
    (e.g. app.py) can omit it — default None means no calls, no behavior change.
    """
    if not sources:
        return []

    if on_status:
        on_status("reformulating")
    query = reformulate_for_search(question, conversation_history, selected_doc_titles or [])
    if not query:
        print("[Literature] reformulation failed — skipping all sources", flush=True)
        return []
    print(f"[Literature] reformulated query: {query!r}", flush=True)
    if on_status:
        on_status("searching_literature")

    def _search_one(source: str) -> list[dict]:
        try:
            if source == "pubmed":
                results = search_pubmed(query, max_results=max_results)
                return [{**r, "source": "pubmed"} for r in results]
            elif source == "semantic_scholar":
                return search_semantic_scholar(query, max_results=max_results)
            elif source == "openalex":
                return search_openalex(query, max_results=max_results)
            else:
                print(f"[Literature] unknown source {source!r}", flush=True)
                return []
        except Exception as e:
            print(f"[Literature] {source} failed: {e}", flush=True)
            return []

    all_results: list[dict] = []
    with ThreadPoolExecutor(max_workers=len(sources)) as pool:
        futures = {pool.submit(_search_one, s): s for s in sources}
        try:
            for fut in as_completed(futures, timeout=_SOURCE_TIMEOUT):
                source = futures[fut]
                batch = fut.result()
                print(f"[Literature] {source}: {len(batch)} result(s)", flush=True)
                all_results.extend(batch)
        except FuturesTimeoutError:
            slow = [s for f, s in futures.items() if not f.done()]
            print(f"[Literature] timeout after {_SOURCE_TIMEOUT}s — abandoned: {slow}", flush=True)

    deduped = _dedupe_literature(all_results)
    print(f"[Literature] {len(deduped)} after dedup (from {len(all_results)} raw)", flush=True)
    return deduped


def embed(text: str) -> list[float]:
    response = ollama.embeddings(model=EMBED_MODEL, prompt=text, keep_alive="30m")
    return response["embedding"]


def build_context(results: dict) -> tuple[str, list[str], list[dict]]:
    """Return (context_str, unique_sources, chunk_details).

    chunk_details entries: {source_file, location, preview}
    """
    blocks, sources, details = [], [], []
    docs = results.get("documents", [[]])[0]
    metas = results.get("metadatas", [[]])[0]

    for doc, meta in zip(docs, metas):
        src = meta.get("source_file", "unknown")
        label_parts = [f"Source: {src}"]
        location = ""

        if meta.get("file_type") == "lecture_audio" and "timestamp_seconds" in meta:
            ts = int(meta["timestamp_seconds"])
            mm, ss = divmod(ts, 60)
            location = f"@{mm:02d}:{ss:02d}"
            label_parts.append(location)
        elif "page_or_slide" in meta:
            key = "slide" if meta.get("file_type") == "pptx" else "page"
            location = f"{key} {meta['page_or_slide']}"
            label_parts.append(location)

        blocks.append(f"[{' | '.join(label_parts)}]\n{doc}")
        if src not in sources:
            sources.append(src)
        details.append({
            "source_file": src,
            "location": location,
            "preview": doc[:150] + ("…" if len(doc) > 150 else ""),
        })

    return "\n\n---\n\n".join(blocks), sources, details


def build_pubmed_context(articles: list[dict]) -> str:
    blocks = []
    for a in articles:
        author_str = ", ".join(a["authors"][:5])
        if len(a["authors"]) > 5:
            author_str += " et al."
        blocks.append(
            f"[PubMed: {a['pmid']}] {a['title']} ({a['journal']}, {a['pub_date']})\n"
            f"Authors: {author_str}\n"
            f"Abstract: {a['abstract']}"
        )
    return "\n\n".join(blocks)


def _stream_reasoning_and_content(messages: list[dict]) -> Iterator[dict]:
    """Yield {"reasoning": text} / {"content": text} chunks from a think=True chat stream.

    Qwen3's reasoning trace arrives one of two ways depending on Ollama/qwen version:
    as a separate chunk["message"]["thinking"] field, or inline in content wrapped in
    <think>...</think>. We check the thinking field first; if a stream never populates
    it, we fall back to splitting <think> tags out of content (same approach as
    gaps.py's _think_filter).
    """
    saw_thinking_field = False
    in_think = False
    buf = ""
    for chunk in ollama.chat(model=LLM_MODEL, messages=messages, stream=True, think=True, keep_alive="30m"):
        msg = chunk.get("message", {})
        thinking = msg.get("thinking")
        if thinking:
            saw_thinking_field = True
            yield {"reasoning": thinking}

        content = msg.get("content") or ""
        if not content:
            continue
        if saw_thinking_field:
            # This stream uses the dedicated field — content is answer-only.
            yield {"content": content}
            continue

        # Fallback: no thinking field seen yet — split <think> tags out of content.
        buf += content
        while buf:
            if not in_think:
                idx = buf.find(_THINK_OPEN)
                if idx == -1:
                    safe = max(0, len(buf) - len(_THINK_OPEN) + 1)
                    if safe:
                        yield {"content": buf[:safe]}
                    buf = buf[safe:]
                    break
                if idx:
                    yield {"content": buf[:idx]}
                buf = buf[idx + len(_THINK_OPEN):]
                in_think = True
            else:
                idx = buf.find(_THINK_CLOSE)
                if idx == -1:
                    safe = max(0, len(buf) - len(_THINK_CLOSE) + 1)
                    if safe:
                        yield {"reasoning": buf[:safe]}
                    buf = buf[safe:]
                    break
                if idx:
                    yield {"reasoning": buf[:idx]}
                buf = buf[idx + len(_THINK_CLOSE):]
                in_think = False
    if buf and not in_think:
        yield {"content": buf}


def _build_where(
    session_id: str | None,
    doc_ids: list[str] | None,
) -> dict:
    # Chat always restricts to ingested source documents, not live lecture transcripts.
    clauses = [{"content_type": "source_document"}]
    if session_id:
        clauses.append({"session_id": session_id})
    if doc_ids:
        clauses.append({"source_file": {"$in": doc_ids}})
    return clauses[0] if len(clauses) == 1 else {"$and": clauses}


def query_stream(
    question: str,
    session_id: str | None = None,
    doc_ids: list[str] | None = None,
    literature_results: list[dict] | None = None,
    conversation_history: str = "",
    stream_thinking: bool = False,
    on_status: Callable[[str], None] | None = None,
) -> Iterator[str | dict]:
    """Yield LLM response tokens, then finally yield {"sources": [...], "pubmed": [...]}.

    When stream_thinking is False (default, matches all existing callers), behavior is
    unchanged: plain str tokens are yielded, then one terminal dict with "sources" etc.

    When stream_thinking is True, each non-terminal yield is instead a dict shaped
    {"reasoning": "..."} or {"content": "..."} so a caller can render the model's
    reasoning trace separately from its answer. The terminal dict is unchanged either way.

    Raises ValueError if the collection is unavailable.
    """
    if not CHROMA_DIR.exists():
        raise ValueError("No chroma_db found. Run ingest.py first.")
    client = chromadb.PersistentClient(path=str(CHROMA_DIR))
    try:
        collection = client.get_collection(COLLECTION_NAME)
    except Exception:
        raise ValueError(f'Collection "{COLLECTION_NAME}" not found.')
    if collection.count() == 0:
        raise ValueError("Collection is empty. Run ingest.py first.")

    if on_status:
        on_status("retrieving")
    query_kwargs: dict = {
        "query_embeddings": [embed(question)],
        "n_results": min(TOP_K, collection.count()),
        "include": ["documents", "metadatas"],
        "where": _build_where(session_id, doc_ids),
    }

    results = collection.query(**query_kwargs)
    local_context, sources, source_details = build_context(results)

    if literature_results:
        lit_context = build_literature_context(literature_results)
        combined_context = (
            "=== LOCAL DOCUMENT CONTEXT ===\n\n"
            + local_context
            + "\n\n=== RECENT LITERATURE ===\n\n"
            + lit_context
        )
        system = SYSTEM_PROMPT_WITH_LITERATURE
    else:
        combined_context = local_context
        system = SYSTEM_PROMPT

    history_section = (
        f"\n\n=== CONVERSATION HISTORY ===\n\n{conversation_history}"
        if conversation_history else ""
    )
    messages = [
        {"role": "system", "content": system},
        {"role": "user", "content": f"Context:\n\n{combined_context}{history_section}\n\nQuestion: {question}"},
    ]
    if stream_thinking:
        for piece in _stream_reasoning_and_content(messages):
            yield piece
    else:
        for chunk in ollama.chat(model=LLM_MODEL, messages=messages, stream=True, keep_alive="30m"):
            yield chunk["message"]["content"]
    yield {"sources": sources, "source_details": source_details, "literature": literature_results or []}


SYSTEM_PROMPT_COLLABORATE = (
    "You are a teaching collaborator and co-instructor helping a professor develop and improve "
    "their lecture. The course materials below give you context for the topic and what the "
    "professor plans to teach. Be generative: offer fresh ideas, new examples, analogies, and "
    "demonstrations; suggest clearer or alternative ways to explain a concept so it lands for "
    "students; and when asked, teach the concept yourself, clearly and substantively. "
    "Draw freely on your own subject knowledge — you are NOT limited to the source material "
    "and do NOT need to cite sources or refuse when something isn't in the materials. "
    "Be warm, collegial, and concrete. Stay accurate: if you're genuinely unsure of a fact, "
    "say so rather than inventing it."
)


def collaborate_stream(
    question: str,
    doc_ids: list[str] | None = None,
    conversation_history: str = "",
    stream_thinking: bool = False,
    on_status: Callable[[str], None] | None = None,
) -> Iterator[str | dict]:
    """Yield LLM tokens for Collaborate mode, then yield {"sources": [...], "literature": []}.

    Reuses the same ChromaDB retrieval as query_stream but drives the teaching/ideation
    system prompt instead. No literature search, no citation requirement.
    See query_stream for the stream_thinking contract (default off, backward-compatible).
    Raises ValueError if the collection is unavailable.
    """
    if not CHROMA_DIR.exists():
        raise ValueError("No chroma_db found. Run ingest.py first.")
    client = chromadb.PersistentClient(path=str(CHROMA_DIR))
    try:
        collection = client.get_collection(COLLECTION_NAME)
    except Exception:
        raise ValueError(f'Collection "{COLLECTION_NAME}" not found.')
    if collection.count() == 0:
        raise ValueError("Collection is empty. Run ingest.py first.")

    if on_status:
        on_status("retrieving")
    results = collection.query(
        query_embeddings=[embed(question)],
        n_results=min(TOP_K, collection.count()),
        include=["documents", "metadatas"],
        where=_build_where(None, doc_ids),
    )
    local_context, sources, source_details = build_context(results)

    history_section = (
        f"\n\n=== CONVERSATION HISTORY ===\n\n{conversation_history}"
        if conversation_history else ""
    )
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT_COLLABORATE},
        {
            "role": "user",
            "content": (
                f"Course material context:\n\n{local_context}"
                f"{history_section}"
                f"\n\nQuestion: {question}"
            ),
        },
    ]
    if stream_thinking:
        for piece in _stream_reasoning_and_content(messages):
            yield piece
    else:
        for chunk in ollama.chat(model=LLM_MODEL, messages=messages, stream=True, keep_alive="30m"):
            yield chunk["message"]["content"]
    yield {"sources": sources, "source_details": source_details, "literature": []}


def resolve_session(args: argparse.Namespace) -> str | None:
    if args.session:
        return args.session
    if args.latest:
        sessions = load_manifest(SESSIONS_MANIFEST)
        if not sessions:
            print("No recorded sessions found in sessions.json.")
            sys.exit(1)
        # session IDs are YYYY-MM-DD_HHMMSS, so lexicographic max == most recent
        return max(sessions.keys())
    return None


def main() -> None:
    parser = argparse.ArgumentParser(description="Query course materials")
    parser.add_argument("question", nargs="+", help="Question to ask")
    parser.add_argument("--session", default=None, metavar="SESSION_ID",
                        help="Scope retrieval to this session ID")
    parser.add_argument("--latest", action="store_true",
                        help="Scope retrieval to the most recent recorded session")
    args = parser.parse_args()

    question = " ".join(args.question)
    session_id = resolve_session(args)
    if session_id:
        print(f"Scoping to session: {session_id}")

    sources: list[str] = []
    print()
    try:
        for token in query_stream(question, session_id):
            if isinstance(token, dict):
                sources = token["sources"]
            else:
                print(token, end="", flush=True)
    except ValueError as e:
        print(e)
        sys.exit(1)
    print("\n")
    print("Sources consulted:")
    for src in sources:
        print(f"  - {src}")


if __name__ == "__main__":
    main()
