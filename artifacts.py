"""
artifacts.py — post-session learning artifact generation.

Four artifacts per session:
  1. Lecture summary   — prose recap of what was taught
  2. Coverage report   — covered / partial / uncovered vs planned slides (reuses gaps_stream)
  3. Study guide       — key concepts + explanations from transcript + linked docs
  4. Quiz              — 5 MC + 3 short-answer + 2 essay questions

Output layout:  exports/{session_id}/
  summary.md
  coverage_report.md
  study_guide.md
  quiz.md
  {session_id}_artifacts.docx   (combined, if python-docx is available)
"""

import re
import sys
from pathlib import Path
from typing import Callable, Iterator

import chromadb
import ollama

from manifest import load_manifest

# ── Constants ─────────────────────────────────────────────────────────────────

EXPORTS_DIR      = Path("exports")
CHROMA_DIR       = Path("chroma_db")
COLLECTION_NAME  = "course_material"
SESSIONS_FILE    = Path("sessions.json")
LLM_MODEL        = "qwen3:30b-a3b"

# Caps to keep context within Qwen3-30b's 32k-token window
_TRANSCRIPT_MAX  = 12_000   # ~15k tokens
_CHUNKS_MAX      = 8_000    # ~10k tokens
_CHUNKS_N        = 25       # max ChromaDB chunks to fetch

_THINK_RE = re.compile(r"<think>.*?</think>", re.DOTALL)


# ── LLM helper ────────────────────────────────────────────────────────────────

def _llm(prompt: str) -> str:
    """Blocking ollama.chat call. Strips any leaked <think> blocks."""
    response = ollama.chat(
        model=LLM_MODEL,
        messages=[{"role": "user", "content": prompt}],
        think=False,
        keep_alive="30m",
    )
    return _THINK_RE.sub("", response["message"]["content"]).strip()


# ── Data loading ──────────────────────────────────────────────────────────────

def _load_transcript_safe(session_id: str) -> str:
    """Return transcript text, or raise ValueError (not sys.exit) if missing."""
    from gaps import TRANSCRIPTS_DIR, TIMESTAMP_RE
    path = TRANSCRIPTS_DIR / f"{session_id}.txt"
    if not path.exists():
        raise ValueError(f"Transcript file not found: {path}")
    body: list[str] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.startswith("--- session ended"):
            break
        if TIMESTAMP_RE.match(line) or body:
            body.append(line)
    return "\n".join(body).strip()


def _load_linked_chunks(linked_docs: list[str]) -> str:
    """Fetch all linked-doc chunks from ChromaDB, truncated to _CHUNKS_MAX chars."""
    if not linked_docs or not CHROMA_DIR.exists():
        return ""
    try:
        client = chromadb.PersistentClient(path=str(CHROMA_DIR))
        collection = client.get_collection(COLLECTION_NAME)
    except Exception:
        return ""

    where: dict = {
        "$and": [
            {"file_type": {"$ne": "lecture_audio"}},
            {"source_file": {"$in": linked_docs}},
        ]
    }
    try:
        results = collection.get(
            where=where,
            include=["documents", "metadatas"],
            limit=_CHUNKS_N,
        )
    except Exception:
        return ""

    blocks: list[str] = []
    total = 0
    for doc, meta in zip(
        results.get("documents") or [],
        results.get("metadatas") or [],
    ):
        src  = (meta or {}).get("source_file", "?")
        page = (meta or {}).get("page_or_slide", "")
        label = f"[{src}{f' | slide {page}' if page else ''}]"
        block = f"{label}\n{doc}"
        total += len(block)
        if total > _CHUNKS_MAX:
            break
        blocks.append(block)

    return "\n\n---\n\n".join(blocks)


# ── Individual generators ─────────────────────────────────────────────────────

def generate_summary(transcript: str) -> str:
    t = transcript[:_TRANSCRIPT_MAX]
    prompt = (
        "You are a teaching assistant. Write a concise, well-structured prose summary "
        "of the following lecture transcript. Cover the key topics in the order they "
        "were discussed. Use markdown headings (##) for major sections. "
        "Aim for 300-500 words — thorough but not padded.\n\n"
        f"LECTURE TRANSCRIPT:\n{t}"
    )
    return _llm(prompt)


def generate_study_guide(transcript: str, chunks: str) -> str:
    t = transcript[:_TRANSCRIPT_MAX]
    context = f"LECTURE TRANSCRIPT:\n{t}"
    if chunks:
        context += f"\n\nCOURSE MATERIALS:\n{chunks[:_CHUNKS_MAX]}"
    prompt = (
        "You are a teaching assistant creating a study guide for students.\n\n"
        f"{context}\n\n"
        "Create a structured study guide. For each key concept:\n"
        "- Use a ## heading for the concept name\n"
        "- Write a clear definition or explanation\n"
        "- Note any relationships to other concepts\n"
        "- Flag anything the professor emphasized\n\n"
        "Cover everything from the transcript and course materials. "
        "Use clear markdown formatting."
    )
    return _llm(prompt)


def generate_quiz(transcript: str, chunks: str) -> str:
    t = transcript[:_TRANSCRIPT_MAX]
    context = f"LECTURE TRANSCRIPT:\n{t}"
    if chunks:
        context += f"\n\nCOURSE MATERIALS:\n{chunks[:_CHUNKS_MAX]}"
    prompt = (
        "You are a teaching assistant creating a quiz for students.\n\n"
        f"{context}\n\n"
        "Generate a quiz with exactly:\n"
        "## Multiple Choice (5 questions)\n"
        "Each question has options A, B, C, D. Mark the correct answer with **(correct)**.\n\n"
        "## Short Answer (3 questions)\n"
        "Each requires 2-4 sentences. Include a model answer after each question.\n\n"
        "## Essay Questions (2 questions)\n"
        "Each should require a paragraph-length response. Include key points the answer should cover.\n\n"
        "Base all questions directly on the transcript and course materials."
    )
    return _llm(prompt)


def generate_coverage_report(session_id: str) -> str:
    """Buffer the full output of gaps_stream as a string."""
    try:
        from gaps import gaps_stream
        result = ""
        for token in gaps_stream(session_id, show_thinking=False):
            result += token
        return result
    except SystemExit:
        raise ValueError("Transcript file not found — cannot generate coverage report.")


# ── DOCX export ───────────────────────────────────────────────────────────────

def _parse_inline(para, text: str) -> None:
    """Add runs to a python-docx paragraph, honouring **bold** and *italic*."""
    # Split on bold markers first
    bold_parts = re.split(r"\*\*([^*]+)\*\*", text)
    for bi, bpart in enumerate(bold_parts):
        if bi % 2 == 1:
            # Inside **bold**
            para.add_run(bpart).bold = True
        else:
            # May contain *italic*
            italic_parts = re.split(r"\*([^*]+)\*", bpart)
            for ii, ipart in enumerate(italic_parts):
                if ipart:
                    run = para.add_run(ipart)
                    run.italic = (ii % 2 == 1)


def _append_md_to_doc(doc, md_text: str) -> None:
    """Parse simple markdown and append content to a python-docx Document."""
    for line in md_text.splitlines():
        stripped = line.strip()

        if not stripped:
            continue

        # ATX headings
        if stripped.startswith("#### "):
            doc.add_heading(stripped[5:], level=4)
        elif stripped.startswith("### "):
            doc.add_heading(stripped[4:], level=3)
        elif stripped.startswith("## "):
            doc.add_heading(stripped[3:], level=2)
        elif stripped.startswith("# "):
            doc.add_heading(stripped[2:], level=1)
        # Horizontal rule
        elif stripped in ("---", "***", "___"):
            doc.add_paragraph("─" * 50)
        # Bullet list
        elif stripped.startswith("- ") or stripped.startswith("* "):
            p = doc.add_paragraph(style="List Bullet")
            _parse_inline(p, stripped[2:])
        # Numbered list
        elif re.match(r"^\d+\.\s+", stripped):
            text = re.sub(r"^\d+\.\s+", "", stripped)
            p = doc.add_paragraph(style="List Number")
            _parse_inline(p, text)
        else:
            p = doc.add_paragraph()
            _parse_inline(p, stripped)


def _write_docx(
    session_id: str,
    sections: dict[str, str],
    output_dir: Path,
) -> Path | None:
    """Write a combined DOCX. Returns path, or None if python-docx is absent."""
    try:
        from docx import Document
        from docx.shared import Pt, RGBColor
    except ImportError:
        print("[ARTIFACTS] python-docx not installed — skipping DOCX export", flush=True)
        return None

    doc = Document()

    # Document title
    title_para = doc.add_heading(f"Session Artifacts — {session_id}", level=0)

    section_titles = {
        "summary":         "Lecture Summary",
        "coverage_report": "Coverage Report",
        "study_guide":     "Study Guide",
        "quiz":            "Quiz",
    }

    for key, heading in section_titles.items():
        if key not in sections:
            continue
        doc.add_page_break()
        doc.add_heading(heading, level=1)
        _append_md_to_doc(doc, sections[key])

    path = output_dir / f"{session_id}_artifacts.docx"
    doc.save(str(path))
    return path


# ── Orchestrator ──────────────────────────────────────────────────────────────

ProgressFn = Callable[[str, str], None]   # (stage, message)


def generate_artifacts(
    session_id: str,
    on_progress: ProgressFn | None = None,
) -> list[dict]:
    """
    Generate all four artifacts for session_id.

    Calls on_progress(stage, message) as each stage starts and completes.
    Returns a list of file dicts: [{label, name, url}, ...].
    Raises ValueError if the session or transcript cannot be found.
    """
    def progress(stage: str, msg: str) -> None:
        print(f"[ARTIFACTS] {msg}", flush=True)
        if on_progress:
            on_progress(stage, msg)

    output_dir = EXPORTS_DIR / session_id
    output_dir.mkdir(parents=True, exist_ok=True)

    # ── Load source data ──────────────────────────────────────────────
    sessions = load_manifest(SESSIONS_FILE)
    session_record = sessions.get(session_id)
    if session_record is None:
        raise ValueError(f"Session '{session_id}' not found in sessions.json")

    linked_docs: list[str] = session_record.get("linked_documents") or []

    transcript = _load_transcript_safe(session_id)
    if not transcript:
        raise ValueError("Transcript is empty — record a lecture first.")

    chunks = _load_linked_chunks(linked_docs)
    if not chunks:
        print(f"[ARTIFACTS] no linked-doc chunks found for session {session_id}", flush=True)

    # ── Generate each artifact ────────────────────────────────────────
    sections: dict[str, str] = {}

    progress("summary", "Generating lecture summary…")
    sections["summary"] = generate_summary(transcript)
    (output_dir / "summary.md").write_text(sections["summary"], encoding="utf-8")
    progress("summary_done", "Lecture summary done")

    progress("coverage_report", "Running coverage analysis…")
    try:
        sections["coverage_report"] = generate_coverage_report(session_id)
    except ValueError as exc:
        sections["coverage_report"] = f"Coverage report unavailable: {exc}"
    (output_dir / "coverage_report.md").write_text(sections["coverage_report"], encoding="utf-8")
    progress("coverage_report_done", "Coverage report done")

    progress("study_guide", "Generating study guide…")
    sections["study_guide"] = generate_study_guide(transcript, chunks)
    (output_dir / "study_guide.md").write_text(sections["study_guide"], encoding="utf-8")
    progress("study_guide_done", "Study guide done")

    progress("quiz", "Generating quiz…")
    sections["quiz"] = generate_quiz(transcript, chunks)
    (output_dir / "quiz.md").write_text(sections["quiz"], encoding="utf-8")
    progress("quiz_done", "Quiz done")

    progress("docx", "Writing DOCX…")
    docx_path = _write_docx(session_id, sections, output_dir)
    progress("docx_done", "DOCX written" if docx_path else "DOCX skipped (python-docx not installed)")

    # ── Build file list ───────────────────────────────────────────────
    base = f"/api/sessions/{session_id}/artifacts"
    files: list[dict] = [
        {"label": "Lecture Summary",   "name": "summary.md",         "url": f"{base}/summary.md"},
        {"label": "Coverage Report",   "name": "coverage_report.md", "url": f"{base}/coverage_report.md"},
        {"label": "Study Guide",       "name": "study_guide.md",     "url": f"{base}/study_guide.md"},
        {"label": "Quiz",              "name": "quiz.md",            "url": f"{base}/quiz.md"},
    ]
    if docx_path:
        fname = docx_path.name
        files.append({"label": "All Artifacts (DOCX)", "name": fname, "url": f"{base}/{fname}"})

    progress("done", "All artifacts ready")
    return files


def list_artifacts(session_id: str) -> list[dict]:
    """Return file list for a session if exports already exist, else []."""
    output_dir = EXPORTS_DIR / session_id
    if not output_dir.exists():
        return []
    base = f"/api/sessions/{session_id}/artifacts"
    label_map = {
        "summary.md":         "Lecture Summary",
        "coverage_report.md": "Coverage Report",
        "study_guide.md":     "Study Guide",
        "quiz.md":            "Quiz",
    }
    order = list(label_map.keys())
    files: list[dict] = []
    for name in order:
        if (output_dir / name).exists():
            files.append({"label": label_map[name], "name": name, "url": f"{base}/{name}"})
    # DOCX last
    docx = next(output_dir.glob("*_artifacts.docx"), None)
    if docx:
        files.append({"label": "All Artifacts (DOCX)", "name": docx.name, "url": f"{base}/{docx.name}"})
    return files
