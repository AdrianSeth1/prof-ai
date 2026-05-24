"""
app.py — Gradio web UI for Prof AI.
Run: python app.py  →  http://127.0.0.1:7860
"""

import shutil
import threading
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
from qa import answer_question
from query import query_stream
from session import Mode
from tts import PiperTTS

SESSIONS_MANIFEST = Path("sessions.json")
DOC_MANIFEST = Path("manifest.json")
LECTURES_DIR = Path("lectures")
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
        except Exception as e:
            print(f"[TTS] Not available: {e}", flush=True)
            _tts_available = False
            return None
    return _tts


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
# Tab 1 — Chat
# ---------------------------------------------------------------------------

def chat_fn(message: str, history: list, scope: str):
    session_id = scope_to_session_id(scope)
    partial, sources = "", []
    try:
        for token in query_stream(message, session_id):
            if isinstance(token, dict):
                sources = token.get("sources", [])
            else:
                partial += token
                yield partial, ""
    except Exception as e:
        yield f"⚠ {e}", ""
        return
    yield partial, "Sources:\n" + "\n".join(f"• {s}" for s in sources)


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
    """Full Q&A cycle: RAG → answer → TTS. Runs in a daemon thread."""
    question = session.pending_question
    try:
        _set_status("Thinking...")
        pause_recording()
        answer = answer_question(question, session.linked_documents)
        session.add_qa_entry(question, answer)
        _set_status("Speaking...")
        tts = _get_tts()
        if tts:
            tts.speak(answer)
    except Exception as e:
        print(f"[Q&A error] {e}", flush=True)
        tts = _get_tts()
        if tts:
            try:
                tts.speak("Sorry, I could not answer that.")
            except Exception:
                pass
    finally:
        resume_recording()
        session.set_mode(Mode.LECTURE)
        _set_status("● Recording lecture")


def start_recording(device_str: str | None, lecture_name: str, linked_docs: list[str]):
    global _live_gap_worker, _live_gap_result, _live_gap_updated
    try:
        session = start_live_session(
            device=parse_device(device_str),
            name=lecture_name.strip(),
            linked_documents=linked_docs or [],
        )
        session.register_qa_handler(_qa_handler)
        _set_status("● Recording lecture")
        with _live_gap_lock:
            _live_gap_result = (
                "Link a document to enable live gap analysis"
                if not linked_docs else "Waiting for more content..."
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
    session = get_current_session()
    if session is None:
        return "Not recording."
    if session.current_mode() != Mode.LECTURE:
        return _get_status()
    session.set_mode(Mode.AWAITING_QUESTION)
    _set_status("Listening for question...")
    return "Listening for question..."


def cancel_qa() -> str:
    """Force mode back to LECTURE, stop TTS, and resume the mic."""
    tts = _get_tts()
    if tts:
        tts.stop()
    resume_recording()
    session = get_current_session()
    if session:
        session.set_mode(Mode.LECTURE)
    _set_status("● Recording lecture")
    return "● Recording lecture"


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
    gap_choices = gap_dropdown_choices()
    doc_choices = doc_dropdown_choices()

    with gr.Blocks(title="Prof AI") as demo:

        # ── Tab 1: Chat ──────────────────────────────────────────────────
        with gr.Tab("Chat"):
            sources_box = gr.Textbox(label="Sources consulted", interactive=False, lines=2)
            scope_dd = gr.Dropdown(choices=scope_choices(), value="All material",
                                   label="Scope queries to", scale=1)
            gr.ChatInterface(fn=chat_fn, additional_inputs=[scope_dd],
                             additional_outputs=[sources_box], autoscroll=True)

        # ── Tab 2: Live Lecture ──────────────────────────────────────────
        with gr.Tab("Live Lecture"):
            # Top controls
            lecture_name_box = gr.Textbox(label="Lecture name (optional)",
                                          placeholder="e.g. Neuro Week 5")
            with gr.Row():
                linked_docs_dd = gr.Dropdown(
                    choices=doc_choices, multiselect=True,
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

            # Timers
            timer = gr.Timer(value=2)
            timer.tick(fn=poll_transcript, outputs=[transcript_box])
            timer.tick(fn=poll_status, outputs=[status_box])
            timer.tick(fn=poll_qa_log, outputs=[qa_log_box])
            timer.tick(fn=poll_cancel_btn, outputs=[cancel_btn])
            gap_timer = gr.Timer(value=5)
            gap_timer.tick(fn=poll_live_gap, outputs=[gap_box, gap_ts_md])

            # Intra-tab events
            refresh_docs_btn.click(
                fn=lambda: gr.update(choices=doc_dropdown_choices()),
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

        # ── Tab 4: Gaps Analysis ─────────────────────────────────────────
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
