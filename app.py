"""
app.py — Gradio web UI for Prof AI.
Run: python app.py  →  http://127.0.0.1:7860
"""

import shutil
import threading
import time
import traceback
import uuid
from datetime import datetime
from pathlib import Path

import gradio as gr
import sounddevice as sd

from batch_transcribe import transcribe_file
from gaps import gaps_stream, resolve_session_id
from ingest import DOCS_DIR, ingest_paths
from live_gap import LiveGapWorker
from live_transcribe import (
    get_current_session, start_live_session, stop_live_session,
    pause_recording, resume_recording,
)
from manifest import load_manifest, save_manifest
from modules import (
    load_modules, save_modules, create_module, rename_module,
    delete_module, expand_selection,
)
from pubmed import search_pubmed, reformulate_for_pubmed
from qa import answer_question
from query import query_stream
from session import Mode
from tts import PiperTTS

SESSIONS_MANIFEST = Path("sessions.json")
DOC_MANIFEST = Path("manifest.json")
LECTURES_DIR = Path("lectures")
QA_AUDIO_DIR = Path("transcripts") / "qa_audio"
AUDIO_EXTS = {".mp3", ".wav", ".m4a", ".flac"}
DOC_EXTS = {".pdf", ".docx", ".pptx"}

# ---------------------------------------------------------------------------
# Live gap analysis shared state
# ---------------------------------------------------------------------------

_live_gap_result: str = ""
_live_gap_updated: datetime | None = None
_live_gap_lock = threading.Lock()
_live_gap_worker: LiveGapWorker | None = None

# ---------------------------------------------------------------------------
# Mode status — updated by background thread, polled by timer
# ---------------------------------------------------------------------------

_mode_status: str = "Stopped"
_mode_status_lock = threading.Lock()


def _set_status(text: str) -> None:
    global _mode_status
    with _mode_status_lock:
        _mode_status = text


def _get_status() -> str:
    with _mode_status_lock:
        return _mode_status


# ---------------------------------------------------------------------------
# TTS singleton — lazy-loaded; silently disabled if model not found
# ---------------------------------------------------------------------------

_tts: PiperTTS | None = None
_tts_available: bool = True


def _get_tts() -> PiperTTS | None:
    global _tts, _tts_available
    if not _tts_available:
        return None
    if _tts is None:
        try:
            _tts = PiperTTS()
        except Exception:
            print("[TTS] Failed to load — voice disabled:", flush=True)
            traceback.print_exc()
            _tts_available = False
            return None
    return _tts


# ---------------------------------------------------------------------------
# Q&A audio — written by background thread, polled by UI timer
# ---------------------------------------------------------------------------

_qa_audio_path: str | None = None
_qa_audio_lock = threading.Lock()


def _init_qa_audio_dir() -> None:
    QA_AUDIO_DIR.mkdir(parents=True, exist_ok=True)
    cutoff = time.time() - 3600
    for f in QA_AUDIO_DIR.glob("*.wav"):
        try:
            if f.stat().st_mtime < cutoff:
                f.unlink()
        except OSError:
            pass


def _save_qa_audio(wav_bytes: bytes) -> str:
    QA_AUDIO_DIR.mkdir(parents=True, exist_ok=True)
    path = QA_AUDIO_DIR / f"{uuid.uuid4().hex}.wav"
    path.write_bytes(wav_bytes)
    return str(path)


def _set_qa_audio(path: str | None) -> None:
    global _qa_audio_path
    with _qa_audio_lock:
        _qa_audio_path = path


def _get_and_clear_qa_audio() -> str | None:
    global _qa_audio_path
    with _qa_audio_lock:
        path = _qa_audio_path
        _qa_audio_path = None
    return path


def _resume_after_playback(session) -> None:
    """Called by threading.Timer after estimated audio duration elapses."""
    if session and session.current_mode() == Mode.PROCESSING:
        resume_recording()
        session.set_mode(Mode.LECTURE)
        _set_status("● Recording lecture")
        print("[Q&A] mic resumed after estimated playback duration", flush=True)
    else:
        print("[Q&A] playback timer fired — mode already changed, skipping", flush=True)


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------

def scope_choices() -> list[str]:
    sessions = sorted(load_manifest(SESSIONS_MANIFEST).keys(), reverse=True)
    return ["All material", "Latest session"] + sessions


def scope_to_session_id(scope: str) -> str | None:
    if scope == "All material":
        return None
    if scope == "Latest session":
        sessions = load_manifest(SESSIONS_MANIFEST)
        return max(sessions.keys()) if sessions else None
    return scope


def get_input_devices() -> list[str]:
    return [
        f"{i}: {d['name']}"
        for i, d in enumerate(sd.query_devices())
        if d["max_input_channels"] > 0
    ]


def parse_device(s: str | None) -> int | None:
    return int(s.split(":")[0]) if s else None


def doc_dropdown_choices() -> list[str]:
    return sorted({Path(k).name for k in load_manifest(DOC_MANIFEST) if Path(k).suffix.lower() in DOC_EXTS})


def unified_source_choices() -> list[tuple[str, str]]:
    """Modules first (prefixed '[Module]'), then individual filenames. Used in Chat and Live Lecture."""
    choices: list[tuple[str, str]] = [
        (f"[Module] {m['name']}", m["id"]) for m in load_modules()
    ]
    choices += [(f, f) for f in doc_dropdown_choices()]
    return choices


def module_dd_choices() -> list[tuple[str, str]]:
    """Choices for the Modules tab module selector (name + doc count)."""
    return [(f"{m['name']} ({len(m['documents'])} docs)", m["id"]) for m in load_modules()]


# ---------------------------------------------------------------------------
# Modules tab event handlers
# ---------------------------------------------------------------------------

def create_module_fn(name: str):
    if not name.strip():
        return gr.update(), gr.update(), gr.update(), "", "Enter a module name."
    mod = create_module(name.strip())
    mod_choices = module_dd_choices()
    src_choices = unified_source_choices()
    return (
        gr.update(choices=mod_choices, value=mod["id"]),
        gr.update(choices=src_choices),
        gr.update(choices=src_choices),
        "",
        f"Created '{mod['name']}'.",
    )


def rename_module_fn(mod_id: str | None, new_name: str):
    if not mod_id:
        return gr.update(), gr.update(), gr.update(), "No module selected."
    if not new_name.strip():
        return gr.update(), gr.update(), gr.update(), "Enter a new name."
    rename_module(mod_id, new_name.strip())
    mod_choices = module_dd_choices()
    src_choices = unified_source_choices()
    return (
        gr.update(choices=mod_choices, value=mod_id),
        gr.update(choices=src_choices),
        gr.update(choices=src_choices),
        f"Renamed to '{new_name.strip()}'.",
    )


def delete_module_stage1(mod_id: str | None):
    if not mod_id:
        return gr.update(interactive=False), "No module selected."
    mods_by_id = {m["id"]: m for m in load_modules()}
    name = mods_by_id.get(mod_id, {}).get("name", mod_id)
    return gr.update(interactive=True), f"⚠ Click 'Confirm Delete' to remove '{name}'."


def delete_module_stage2(mod_id: str | None):
    if not mod_id:
        return gr.update(), gr.update(), gr.update(), gr.update(interactive=False), "No module selected."
    mods_by_id = {m["id"]: m for m in load_modules()}
    name = mods_by_id.get(mod_id, {}).get("name", mod_id)
    delete_module(mod_id)
    mod_choices = module_dd_choices()
    src_choices = unified_source_choices()
    new_val = mod_choices[0][1] if mod_choices else None
    return (
        gr.update(choices=mod_choices, value=new_val),
        gr.update(choices=src_choices),
        gr.update(choices=src_choices),
        gr.update(interactive=False),
        f"Deleted '{name}'.",
    )


def load_module_docs_fn(mod_id: str | None):
    if not mod_id:
        return gr.update(value=[])
    mods_by_id = {m["id"]: m for m in load_modules()}
    mod = mods_by_id.get(mod_id, {})
    return gr.update(value=mod.get("documents", []))


def save_module_docs_fn(mod_id: str | None, selected: list[str]):
    if not mod_id:
        return gr.update(), gr.update(), gr.update(), "No module selected."
    mods = load_modules()
    for mod in mods:
        if mod["id"] == mod_id:
            mod["documents"] = selected or []
            save_modules(mods)
            mod_choices = module_dd_choices()
            src_choices = unified_source_choices()
            return (
                gr.update(choices=mod_choices, value=mod_id),
                gr.update(choices=src_choices),
                gr.update(choices=src_choices),
                f"Saved {len(selected or [])} doc(s) to '{mod['name']}'.",
            )
    return gr.update(), gr.update(), gr.update(), "Module not found."


def refresh_modules_tab():
    mod_choices = module_dd_choices()
    return gr.update(choices=mod_choices, value=mod_choices[0][1] if mod_choices else None)


def session_label(sid: str, entry: dict) -> str:
    dt = datetime.fromisoformat(entry["start_time"])
    ts = dt.strftime("%Y-%m-%d %H:%M")
    count = entry.get("segment_count", 0)
    name = entry.get("name", "")
    if count == 0:
        return f"(empty) {ts}"
    return f"{name} — {ts} ({count} segments)" if name else f"{ts} ({count} segments)"


def gap_dropdown_choices() -> list[tuple[str, str]]:
    sessions = load_manifest(SESSIONS_MANIFEST)
    ordered = sorted(sessions.items(), key=lambda kv: kv[1].get("start_time", kv[0]), reverse=True)
    return [(session_label(sid, e), sid) for sid, e in ordered]


def refresh_gap_dropdown():
    choices = gap_dropdown_choices()
    return gr.update(choices=choices, value=choices[0][1] if choices else None)


def linked_docs_info(sid: str | None) -> tuple[str, list[str]]:
    if not sid:
        return "No session selected.", []
    entry = load_manifest(SESSIONS_MANIFEST).get(sid, {})
    linked = entry.get("linked_documents", [])
    lbl = ("Comparing against: " + ", ".join(linked)) if linked \
        else "No specific documents linked — comparing against all course material"
    return lbl, linked


def save_linked_docs_fn(sid: str | None, linked: list[str]) -> str:
    if not sid:
        return "No session selected."
    sessions = load_manifest(SESSIONS_MANIFEST)
    if sid not in sessions:
        return "Session not found."
    sessions[sid]["linked_documents"] = linked or []
    save_manifest(SESSIONS_MANIFEST, sessions)
    lbl, _ = linked_docs_info(sid)
    return lbl


def refresh_gaps_tab():
    choices = gap_dropdown_choices()
    v = choices[0][1] if choices else None
    lbl, edit = linked_docs_info(v)
    return gr.update(choices=choices, value=v), lbl, edit


# ---------------------------------------------------------------------------
# Tab 1 — Chat helpers
# ---------------------------------------------------------------------------

def _format_history(history, max_turns=6):
    """Format Gradio chat history into a plain text block for the LLM prompt.

    Handles both string content and Gradio's list-of-parts content format.
    """
    if not history:
        return ""

    recent = history[-(max_turns * 2):] if max_turns else history
    formatted = []

    for entry in recent:
        raw = entry.get("content")

        # Content can be a string OR a list of message parts. Normalize.
        if isinstance(raw, list):
            text_parts = []
            for part in raw:
                if isinstance(part, dict):
                    text_parts.append(part.get("text", ""))
                elif isinstance(part, str):
                    text_parts.append(part)
            content = " ".join(text_parts)
        elif isinstance(raw, str):
            content = raw
        else:
            content = str(raw) if raw is not None else ""

        content = content.strip()
        if not content:
            continue

        role = entry.get("role", "user").upper()
        formatted.append(f"{role}: {content}")

    return "\n".join(formatted)


def _format_docs_md(details: list[dict]) -> str:
    if not details:
        return "*No local documents retrieved.*"
    by_file: dict[str, list[dict]] = {}
    for d in details:
        by_file.setdefault(d["source_file"], []).append(d)
    lines = []
    for src, chunks in by_file.items():
        lines.append(f"**{src}**")
        for c in chunks:
            loc = f"`{c['location']}` " if c["location"] else ""
            lines.append(f"- {loc}{c['preview']}")
        lines.append("")
    return "\n".join(lines).strip()


def _format_pubmed_html(articles: list[dict]) -> str:
    if not articles:
        return (
            "<p style='color:#6b7280; font-style:italic;'>"
            "Enable the PubMed toggle and ask a question to see recent literature."
            "</p>"
        )
    cards = []
    for a in articles:
        author_str = ", ".join(a["authors"][:3])
        if len(a["authors"]) > 3:
            author_str += " et al."
        cards.append(
            f"<div style='border:1px solid #d1d5db; border-radius:8px; padding:14px;"
            f" margin:8px 0; background:#f9fafb;'>"
            f"<a href='{a['url']}' target='_blank' rel='noopener noreferrer'"
            f" style='font-weight:600; font-size:0.95em; color:#1d4ed8;"
            f" text-decoration:none;'>{a['title']}</a>"
            f"<div style='margin-top:6px; font-size:0.85em; color:#4b5563;'>"
            f"<strong>{a['journal']}</strong> &bull; {a['pub_date']} &bull; "
            f"<a href='{a['url']}' target='_blank' rel='noopener noreferrer'"
            f" style='color:#6b7280;'>PMID {a['pmid']}</a></div>"
            f"<div style='margin-top:4px; font-size:0.82em; color:#6b7280;'>"
            f"{author_str}</div>"
            f"</div>"
        )
    return "".join(cards)


# ---------------------------------------------------------------------------
# Tab 1 — Chat
# ---------------------------------------------------------------------------

def chat_fn(
    message: str,
    history: list,
    scope: str,
    doc_filter: list[str],
    pubmed_on: bool,
    pubmed_max: int,
):
    session_id = scope_to_session_id(scope)
    partial, source_details, pubmed_articles = "", [], []

    # Expand any module IDs in the filter to their constituent filenames.
    expanded_filter = expand_selection(doc_filter) if doc_filter else []

    history_for_llm = _format_history(history, max_turns=6)
    history_for_pubmed = _format_history(history, max_turns=3)

    pubmed_results = None
    if pubmed_on:
        query = reformulate_for_pubmed(message, history_for_pubmed, expanded_filter)
        if query:
            pubmed_results = search_pubmed(query, max_results=int(pubmed_max))
        else:
            print("[PubMed] reformulation returned empty — skipping", flush=True)

    try:
        for token in query_stream(
            message, session_id,
            doc_ids=expanded_filter or None,
            pubmed_results=pubmed_results,
            conversation_history=history_for_llm,
        ):
            if isinstance(token, dict):
                source_details = token.get("source_details", [])
                pubmed_articles = token.get("pubmed", [])
            else:
                partial += token
                yield partial, "", ""
    except Exception as e:
        yield f"⚠ {e}", "", ""
        return

    yield partial, _format_docs_md(source_details), _format_pubmed_html(pubmed_articles)


# ---------------------------------------------------------------------------
# Tab 2 — Live Lecture
# ---------------------------------------------------------------------------

def _on_live_gap(text: str, ts: datetime | None) -> None:
    global _live_gap_result, _live_gap_updated
    with _live_gap_lock:
        _live_gap_result = text
        _live_gap_updated = ts


def poll_live_gap() -> tuple[str, str]:
    with _live_gap_lock:
        text = _live_gap_result
        ts = _live_gap_updated
    ts_str = f"*Last updated: {ts.strftime('%H:%M:%S')}*" if ts else ""
    return text, ts_str


def _qa_handler(session) -> None:
    """Full Q&A cycle: RAG → synthesize → browser audio. Runs in a daemon thread."""
    question = session.pending_question
    print(f"[Q&A] handler started, question: {question!r}", flush=True)
    try:
        _set_status("Thinking...")
        pause_recording()
        print("[Q&A] mic paused, calling answer_question()", flush=True)
        answer, sources = answer_question(question, session.linked_documents, session)
        print(f"[Q&A] got answer ({len(answer)} chars)", flush=True)
        session.add_qa_entry(question, answer, sources)
        _set_status("Speaking...")
        tts = _get_tts()
        if tts:
            print("[Q&A] synthesizing audio", flush=True)
            wav_bytes = tts.synthesize(answer)
            audio_path = _save_qa_audio(wav_bytes)
            _set_qa_audio(audio_path)
            duration_s = len(wav_bytes) / (tts.sample_rate * 2)
            print(f"[Q&A] audio ready ({duration_s:.1f}s), mic resumes in {duration_s + 0.5:.1f}s", flush=True)
            threading.Timer(duration_s + 0.5, _resume_after_playback, args=(session,)).start()
            return
        else:
            print("[Q&A] TTS unavailable", flush=True)
    except Exception:
        traceback.print_exc()
        try:
            tts = _get_tts()
            if tts:
                wav_bytes = tts.synthesize("Sorry, I could not answer that.")
                audio_path = _save_qa_audio(wav_bytes)
                _set_qa_audio(audio_path)
                duration_s = len(wav_bytes) / (tts.sample_rate * 2)
                threading.Timer(duration_s + 0.5, _resume_after_playback, args=(session,)).start()
                return
        except Exception:
            traceback.print_exc()
    # Fallback: TTS unavailable or synthesis failed — resume immediately
    resume_recording()
    session.set_mode(Mode.LECTURE)
    _set_status("● Recording lecture")
    print("[Q&A] handler done (no audio), mode → LECTURE", flush=True)


def start_recording(device_str: str | None, lecture_name: str, linked_docs: list[str]):
    global _live_gap_worker, _live_gap_result, _live_gap_updated
    try:
        # Expand any module IDs to their constituent document filenames.
        expanded_docs = expand_selection(linked_docs or [])
        session = start_live_session(
            device=parse_device(device_str),
            name=lecture_name.strip(),
            linked_documents=expanded_docs,
        )
        session.register_qa_handler(_qa_handler)
        if expanded_docs:
            _set_status("● Recording lecture")
        else:
            _set_status("● Recording lecture (no source material linked — Ask AI will be disabled until you link a document)")
        with _live_gap_lock:
            _live_gap_result = (
                "Link a document to enable live gap analysis"
                if not expanded_docs else "Waiting for more content..."
            )
            _live_gap_updated = None
        _live_gap_worker = LiveGapWorker(get_current_session, _on_live_gap)
        _live_gap_worker.start()
        return (
            gr.update(interactive=False),
            gr.update(interactive=True),
            gr.update(interactive=True),
        )
    except Exception as e:
        _set_status(f"⚠ {e}")
        return gr.update(), gr.update(), gr.update()


def stop_recording():
    global _live_gap_worker
    if _live_gap_worker:
        _live_gap_worker.stop()
        _live_gap_worker = None
    stop_live_session()
    _set_status("■ Stopped")
    choices = gap_dropdown_choices()
    return (
        gr.update(interactive=True),
        gr.update(interactive=False),
        gr.update(choices=choices, value=choices[0][1] if choices else None),
        gr.update(interactive=False),
        gr.update(interactive=False),
    )


def poll_transcript() -> str:
    session = get_current_session()
    if session is None:
        return gr.update()
    lines = []
    for seg in session.segments:
        mm, ss = divmod(int(seg["start_seconds"]), 60)
        lines.append(f"[{mm:02d}:{ss:02d}] {seg['text']}")
    return "\n".join(lines)


def poll_status() -> str:
    return _get_status()


def poll_qa_log() -> str:
    session = get_current_session()
    if session is None:
        return gr.update()
    return session.format_qa_log()


def ask_ai_fn() -> str:
    print("[ASK AI] button clicked", flush=True)
    session = get_current_session()
    if session is None:
        print("[ASK AI] no active session", flush=True)
        return "Not recording."
    print(f"[ASK AI] current mode: {session.current_mode()}", flush=True)
    if session.current_mode() != Mode.LECTURE:
        print("[ASK AI] not in LECTURE mode — ignoring", flush=True)
        return _get_status()
    if not session.linked_documents:
        print("[ASK AI] no linked documents — refusing to enter Q&A mode")
        msg = "● No document linked — select source material first"
        _set_status(msg)
        return msg
    session.set_mode(Mode.AWAITING_QUESTION)
    _set_status("Listening for question...")
    print("[ASK AI] mode → AWAITING_QUESTION", flush=True)
    return "Listening for question..."


def cancel_qa() -> str:
    """Force mode back to LECTURE, clear pending audio, and resume the mic."""
    _set_qa_audio(None)
    resume_recording()
    session = get_current_session()
    if session:
        session.set_mode(Mode.LECTURE)
    _set_status("● Recording lecture")
    return "● Recording lecture"


def poll_qa_audio():
    path = _get_and_clear_qa_audio()
    if path:
        return gr.update(value=path)
    return gr.update()


def poll_cancel_btn() -> dict:
    session = get_current_session()
    active = session is not None and session.current_mode() != Mode.LECTURE
    return gr.update(interactive=active)


# ---------------------------------------------------------------------------
# Tab 3 — Add Materials
# ---------------------------------------------------------------------------

def process_uploads(files):
    if not files:
        yield "No files selected."
        return
    DOCS_DIR.mkdir(exist_ok=True)
    LECTURES_DIR.mkdir(exist_ok=True)
    log = ""
    for f in files:
        src = Path(f.name)
        suffix = src.suffix.lower()
        if suffix in DOC_EXTS:
            dest = DOCS_DIR / src.name
            shutil.copy2(src, dest)
            for msg in ingest_paths([dest]):
                log += msg + "\n"
                yield log
        elif suffix in AUDIO_EXTS:
            dest = LECTURES_DIR / src.name
            shutil.copy2(src, dest)
            for msg in transcribe_file(dest):
                log += msg + "\n"
                yield log
        else:
            log += f"Skipped {src.name} (unsupported type)\n"
            yield log
    log += "\nAll done."
    yield log


# ---------------------------------------------------------------------------
# Tab 4 — Gaps Analysis
# ---------------------------------------------------------------------------

def run_gaps(session_choice: str, use_latest: bool, show_thinking: bool):
    try:
        session_id = resolve_session_id(
            session_id=None if use_latest else (session_choice or None),
            latest=use_latest,
        )
    except ValueError as e:
        yield str(e)
        return
    result = ""
    for token in gaps_stream(session_id, show_thinking):
        result += token
        yield result


# ---------------------------------------------------------------------------
# Build UI
# ---------------------------------------------------------------------------

def build_ui() -> gr.Blocks:
    _init_qa_audio_dir()
    gap_choices = gap_dropdown_choices()
    doc_choices = doc_dropdown_choices()

    with gr.Blocks(title="Prof AI") as demo:

        # ── Tab 1: Chat ──────────────────────────────────────────────────
        with gr.Tab("Chat") as chat_tab:
            # Declare output components with render=False so they can be
            # passed to ChatInterface but rendered below it in the layout.
            docs_md = gr.Markdown(
                "*Ask a question to see document sources.*", render=False
            )
            pubmed_html = gr.HTML(
                "<p style='color:#6b7280; font-style:italic;'>"
                "Enable the PubMed toggle to include recent literature.</p>",
                render=False,
            )

            with gr.Row():
                scope_dd = gr.Dropdown(choices=scope_choices(), value="All material",
                                       label="Scope queries to", scale=1)
                doc_filter_dd = gr.Dropdown(
                    choices=unified_source_choices(), multiselect=True,
                    label="Search in (leave empty for all)", scale=3,
                )
            with gr.Row():
                pubmed_toggle = gr.Checkbox(
                    label="Include recent literature (PubMed)", value=False, scale=2,
                )
                pubmed_max_num = gr.Number(
                    value=5, minimum=1, maximum=10, precision=0,
                    label="Max papers to include", scale=0, min_width=180,
                )

            gr.ChatInterface(
                fn=chat_fn,
                additional_inputs=[scope_dd, doc_filter_dd, pubmed_toggle, pubmed_max_num],
                additional_outputs=[docs_md, pubmed_html],
                autoscroll=True,
            )

            # Sources accordion — rendered here so it appears below the chat
            with gr.Accordion("Sources", open=False):
                with gr.Tabs():
                    with gr.Tab("Documents"):
                        docs_md.render()
                    with gr.Tab("Recent Literature"):
                        pubmed_html.render()

            chat_tab.select(
                fn=lambda: gr.update(choices=unified_source_choices()),
                outputs=[doc_filter_dd],
            )

        # ── Tab 2: Live Lecture ──────────────────────────────────────────
        with gr.Tab("Live Lecture"):
            # Top controls
            lecture_name_box = gr.Textbox(label="Lecture name (optional)",
                                          placeholder="e.g. Neuro Week 5")
            with gr.Row():
                linked_docs_dd = gr.Dropdown(
                    choices=unified_source_choices(), multiselect=True,
                    label="Source material for this lecture", scale=3,
                )
                refresh_docs_btn = gr.Button("↻", scale=0, min_width=40)
            with gr.Row():
                device_dd = gr.Dropdown(choices=get_input_devices(),
                                        label="Input device", scale=3)
                status_box = gr.Textbox(value="■ Stopped", label="Status",
                                        interactive=False, scale=1)
            with gr.Row():
                start_btn = gr.Button("▶ Start Lecture", variant="primary")
                stop_btn = gr.Button("■ Stop Lecture", variant="stop", interactive=False)
                ask_ai_btn = gr.Button("Ask AI", interactive=False)
                cancel_btn = gr.Button("✕ Cancel", interactive=False)

            # Side-by-side: transcript | live gap analysis
            with gr.Row():
                with gr.Column():
                    transcript_box = gr.Textbox(label="Live Transcript", lines=20,
                                                interactive=False, autoscroll=True)
                with gr.Column():
                    gap_box = gr.Textbox(label="Live Gap Analysis", lines=19,
                                         interactive=False)
                    gap_ts_md = gr.Markdown("")

            # Q&A history (collapsed by default)
            with gr.Accordion("Q&A History", open=False):
                qa_log_box = gr.Textbox(label="", lines=8,
                                        interactive=False, autoscroll=True)

            qa_audio = gr.Audio(
                label="AI Response",
                autoplay=True,
                visible=True,
                streaming=False,
                interactive=False,
            )

            # Timers
            timer = gr.Timer(value=2)
            timer.tick(fn=poll_transcript, outputs=[transcript_box])
            timer.tick(fn=poll_status, outputs=[status_box])
            timer.tick(fn=poll_qa_log, outputs=[qa_log_box])
            timer.tick(fn=poll_cancel_btn, outputs=[cancel_btn])
            timer.tick(fn=poll_qa_audio, outputs=[qa_audio])
            gap_timer = gr.Timer(value=5)
            gap_timer.tick(fn=poll_live_gap, outputs=[gap_box, gap_ts_md])

            # Intra-tab events
            refresh_docs_btn.click(
                fn=lambda: gr.update(choices=unified_source_choices()),
                outputs=[linked_docs_dd],
            )
            ask_ai_btn.click(fn=ask_ai_fn, outputs=[status_box])
            cancel_btn.click(fn=cancel_qa, outputs=[status_box])

        # ── Tab 3: Add Materials ─────────────────────────────────────────
        with gr.Tab("Add Materials"):
            upload = gr.File(
                label="Upload files (.pdf, .docx, .pptx, .mp3, .wav, .m4a, .flac)",
                file_count="multiple",
                file_types=[".pdf", ".docx", ".pptx", ".mp3", ".wav", ".m4a", ".flac"],
            )
            process_btn = gr.Button("Process", variant="primary")
            process_log = gr.Textbox(label="Progress log", lines=10, interactive=False)
            process_btn.click(fn=process_uploads, inputs=[upload], outputs=[process_log])

        # ── Tab 4: Modules ───────────────────────────────────────────────
        with gr.Tab("Modules") as modules_tab:
            with gr.Row():
                module_dd = gr.Dropdown(
                    choices=module_dd_choices(), label="Module", scale=3,
                )
                mod_refresh_btn = gr.Button("↻", scale=0, min_width=40)
            with gr.Row():
                new_mod_tb = gr.Textbox(
                    label="New module name",
                    placeholder="e.g. Week 3 – Synaptic Plasticity",
                    scale=3,
                )
                create_mod_btn = gr.Button("Create", scale=0, min_width=80)
            with gr.Row():
                rename_mod_tb = gr.Textbox(
                    label="Rename selected to", placeholder="New name", scale=3,
                )
                rename_mod_btn = gr.Button("Rename", scale=0, min_width=80)
            with gr.Row():
                delete_mod_btn = gr.Button("Delete Selected", variant="stop",
                                           scale=0, min_width=120)
                confirm_delete_btn = gr.Button("Confirm Delete", variant="stop",
                                               interactive=False, scale=0, min_width=120)
            mod_docs_dd = gr.Dropdown(
                choices=doc_dropdown_choices(), multiselect=True,
                label="Documents in this module",
            )
            save_mod_docs_btn = gr.Button("Save Documents", variant="primary")
            modules_status_tb = gr.Textbox(label="", interactive=False, lines=1)

            mod_refresh_btn.click(fn=refresh_modules_tab, outputs=[module_dd])
            modules_tab.select(fn=refresh_modules_tab, outputs=[module_dd])
            module_dd.change(
                fn=load_module_docs_fn, inputs=[module_dd], outputs=[mod_docs_dd],
            )
            create_mod_btn.click(
                fn=create_module_fn,
                inputs=[new_mod_tb],
                outputs=[module_dd, linked_docs_dd, doc_filter_dd, new_mod_tb, modules_status_tb],
            )
            rename_mod_btn.click(
                fn=rename_module_fn,
                inputs=[module_dd, rename_mod_tb],
                outputs=[module_dd, linked_docs_dd, doc_filter_dd, modules_status_tb],
            )
            delete_mod_btn.click(
                fn=delete_module_stage1,
                inputs=[module_dd],
                outputs=[confirm_delete_btn, modules_status_tb],
            )
            confirm_delete_btn.click(
                fn=delete_module_stage2,
                inputs=[module_dd],
                outputs=[module_dd, linked_docs_dd, doc_filter_dd,
                         confirm_delete_btn, modules_status_tb],
            )
            save_mod_docs_btn.click(
                fn=save_module_docs_fn,
                inputs=[module_dd, mod_docs_dd],
                outputs=[module_dd, linked_docs_dd, doc_filter_dd, modules_status_tb],
            )

        # ── Tab 5: Gaps Analysis ─────────────────────────────────────────
        with gr.Tab("Gaps Analysis") as gaps_tab:
            with gr.Row():
                gap_dd = gr.Dropdown(
                    choices=gap_choices,
                    value=gap_choices[0][1] if gap_choices else None,
                    label="Session", scale=3,
                )
                gap_refresh_btn = gr.Button("↻ Refresh", scale=0, min_width=90)
                use_latest_cb = gr.Checkbox(label="Use latest session", scale=1)
            linked_docs_label = gr.Textbox(
                value=linked_docs_info(gap_choices[0][1] if gap_choices else None)[0],
                label="Document scope", interactive=False, lines=1,
            )
            with gr.Row():
                linked_docs_edit_dd = gr.Dropdown(
                    choices=doc_choices,
                    value=linked_docs_info(gap_choices[0][1] if gap_choices else None)[1],
                    multiselect=True, label="Linked documents (edit & save)", scale=3,
                )
                save_links_btn = gr.Button("Save", scale=0, min_width=60)
            show_thinking_cb = gr.Checkbox(label="Show reasoning trace")
            gaps_btn = gr.Button("Find Gaps", variant="primary")
            gaps_out = gr.Textbox(label="Gap analysis", lines=20, interactive=False)

            # Intra-tab events
            gaps_btn.click(fn=run_gaps,
                           inputs=[gap_dd, use_latest_cb, show_thinking_cb],
                           outputs=[gaps_out])
            gap_dd.change(fn=linked_docs_info, inputs=[gap_dd],
                          outputs=[linked_docs_label, linked_docs_edit_dd])
            save_links_btn.click(fn=save_linked_docs_fn,
                                 inputs=[gap_dd, linked_docs_edit_dd],
                                 outputs=[linked_docs_label])
            gap_refresh_btn.click(fn=refresh_gaps_tab,
                                  outputs=[gap_dd, linked_docs_label, linked_docs_edit_dd])
            gaps_tab.select(fn=refresh_gaps_tab,
                            outputs=[gap_dd, linked_docs_label, linked_docs_edit_dd])

        # ── Cross-tab events ──────────────────────────────────────────────
        start_btn.click(
            fn=start_recording,
            inputs=[device_dd, lecture_name_box, linked_docs_dd],
            outputs=[start_btn, stop_btn, ask_ai_btn],
        )
        stop_btn.click(
            fn=stop_recording,
            outputs=[start_btn, stop_btn, gap_dd, ask_ai_btn, cancel_btn],
        )

    return demo


if __name__ == "__main__":
    build_ui().launch(server_name="0.0.0.0", server_port=7860, theme=gr.themes.Soft(), share=True, auth=("prof", "password"))
