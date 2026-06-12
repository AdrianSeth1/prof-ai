"""
live_transcribe.py — real-time lecture transcription.
Usage: python live_transcribe.py [--device <index>]
"""

import argparse
import re
import signal
import sys
import threading
import time

import numpy as np
import ollama

# On Windows, ctranslate2 uses LoadLibrary which ignores add_dll_directory.
# Prepend nvidia wheel DLL paths to PATH before the model loads.
import os as _os, sys as _sys
_nvidia_dirs = [
    _os.path.join(_p, "nvidia", _sub, "bin")
    for _p in _sys.path
    for _sub in ("cublas", "cudnn")
    if _os.path.isdir(_os.path.join(_p, "nvidia", _sub, "bin"))
]
if _nvidia_dirs:
    _os.environ["PATH"] = _os.pathsep.join(_nvidia_dirs) + _os.pathsep + _os.environ.get("PATH", "")
del _nvidia_dirs

from faster_whisper import WhisperModel

from recorder import SAMPLE_RATE, VAD_SILENCE_MS
from session import LectureSession, Mode

WHISPER_MODEL = "large-v3-turbo"
VOCAB_MODEL = "qwen3:14b"
MIN_WORDS = 2
MIN_AVG_LOGPROB = -1.0
NO_SPEECH_THRESHOLD = 0.7
LOG_PROB_THRESHOLD = -0.8

QA_BUFFER_SECONDS = 4.0    # initial window to collect speech after Ask AI is clicked
QA_TAIL_SECONDS = 1.5      # silence after last speech before closing the buffer
QA_TRUNCATION_WAIT = 2.0   # extra wait when transcript looks like a mid-sentence fragment

# Words that signal a truncated question when they appear at the end with no terminal punctuation
_TRUNCATION_TERMINALS = {
    "the", "a", "an",
    "of", "for", "with", "about", "between", "from",
    "and", "or", "but",
}

# Whisper's initial_prompt has a ~224 token cap; 900 chars is a safe ceiling
_WHISPER_PROMPT_MAX_CHARS = 900
_THINK_RE = re.compile(r"<think>.*?</think>", re.DOTALL)

DEFAULT_INITIAL_PROMPT = (
    "This is a university lecture. The speaker may use academic terminology, "
    "cite authors and theorists by name, reference published works, and use "
    "field-specific jargon. Names, acronyms, and theory titles should be "
    "transcribed accurately."
)

# ---------------------------------------------------------------------------
# Module-level singletons — managed by start/stop_live_session()
# ---------------------------------------------------------------------------

_whisper_model: WhisperModel | None = None
_active_session: LectureSession | None = None
_session_start: float = 0.0


# ---------------------------------------------------------------------------
# Whisper initial prompt — built from linked document vocabulary at session start
# ---------------------------------------------------------------------------

def _natural_list(items: list[str]) -> str:
    """Join items with Oxford-comma grammar: 'A, B, C, and D'."""
    if not items:
        return ""
    if len(items) == 1:
        return items[0]
    if len(items) == 2:
        return f"{items[0]} and {items[1]}"
    return ", ".join(items[:-1]) + ", and " + items[-1]


def _format_whisper_prompt(concepts: list[str], proper: list[str], max_chars: int) -> str:
    """Build a sentence-shaped prompt from classified term lists.

    Drops whole items from the end to stay under max_chars — never slices mid-word.
    Drops from concepts first (lower priority than named references).
    """
    lead = "This is a university lecture."
    # Work on copies so callers aren't mutated
    concepts, proper = list(concepts), list(proper)
    while True:
        parts = [lead]
        if concepts:
            parts.append(f"Topics may include: {_natural_list(concepts)}.")
        if proper:
            parts.append(f"Speakers may reference: {_natural_list(proper)}.")
        result = " ".join(parts)
        if len(result) <= max_chars:
            return result
        if concepts:
            concepts.pop()
        elif proper:
            proper.pop()
        else:
            return lead


def build_whisper_prompt(session: "LectureSession") -> str:
    """Pull vocabulary from linked doc chunks via qwen3:14b; fall back to DEFAULT_INITIAL_PROMPT."""
    if not session.linked_documents:
        print("[Whisper] no linked documents — using default initial prompt", flush=True)
        return DEFAULT_INITIAL_PROMPT

    sampled_texts: list[str] = []
    for doc_name in session.linked_documents:
        try:
            results = session._collection.get(
                where={"$and": [
                    {"source_file": {"$eq": doc_name}},
                    {"content_type": {"$eq": "source_document"}},
                ]},
                include=["documents"],
            )
            sampled_texts.extend((results["documents"] or [])[:2])
        except Exception as e:
            print(f"[Whisper] could not fetch chunks for {doc_name!r}: {e}", flush=True)

    if not sampled_texts:
        print("[Whisper] no chunks found for linked documents — using default initial prompt", flush=True)
        return DEFAULT_INITIAL_PROMPT

    combined = "\n\n".join(sampled_texts)[:4000]
    extraction_prompt = (
        "/no_think List 20-30 technical terms, proper nouns, or specialized vocabulary "
        "from the following text. Output as a comma-separated list with no explanation, "
        "no numbering, no preamble. Focus on words that a general speech recognizer might "
        "mis-transcribe — names, jargon, acronyms, theory names, author names. "
        f"Text: {combined}"
    )

    try:
        print("[Whisper] generating vocabulary prompt from linked documents...", flush=True)
        response = ollama.chat(
            model=VOCAB_MODEL,
            messages=[{"role": "user", "content": extraction_prompt}],
        )
        raw_vocab = _THINK_RE.sub("", response["message"]["content"]).strip()
        if not raw_vocab:
            raise ValueError("empty response from vocabulary model")

        # Split into terms and classify: multi-word + starts-capital → proper noun, else concept
        terms = [t.strip() for t in raw_vocab.split(",") if t.strip()]
        proper, concepts = [], []
        for term in terms:
            words = term.split()
            if len(words) > 1 and words[0][0].isupper():
                proper.append(term)
            else:
                concepts.append(term)

        formatted = _format_whisper_prompt(concepts, proper, _WHISPER_PROMPT_MAX_CHARS)
        print(f"[Whisper] initial prompt: {formatted!r}", flush=True)
        return formatted
    except Exception as e:
        print(f"[Whisper] vocabulary extraction failed ({e}) — using default initial prompt", flush=True)
        return DEFAULT_INITIAL_PROMPT


# ---------------------------------------------------------------------------
# Transcription helpers
# ---------------------------------------------------------------------------

def _get_model() -> WhisperModel:
    global _whisper_model
    if _whisper_model is None:
        print(f"[Whisper] loading model={WHISPER_MODEL!r} device=cuda compute_type=float16", flush=True)
        _whisper_model = WhisperModel(WHISPER_MODEL, device="cuda", compute_type="float16")
        print(f"[Whisper] {WHISPER_MODEL!r} ready", flush=True)
    return _whisper_model


def transcribe_audio(model: WhisperModel, audio: np.ndarray, initial_prompt: str = "") -> tuple[str, float]:
    segments_gen, _ = model.transcribe(
        audio,
        language="en",
        beam_size=5,
        no_speech_threshold=NO_SPEECH_THRESHOLD,
        log_prob_threshold=LOG_PROB_THRESHOLD,
        condition_on_previous_text=False,
        initial_prompt=initial_prompt if initial_prompt else None,
    )
    texts, logprobs = [], []
    for seg in segments_gen:
        t = seg.text.strip()
        if t:
            texts.append(t)
            logprobs.append(seg.avg_logprob)
    if not texts:
        return "", 0.0
    return " ".join(texts), sum(logprobs) / len(logprobs)


def process_utterance(
    audio: np.ndarray,
    model: WhisperModel,
    session: LectureSession,
    session_start: float,
) -> None:
    text, confidence = transcribe_audio(model, audio, session.whisper_initial_prompt)
    if not text or len(text.split()) < MIN_WORDS or confidence < MIN_AVG_LOGPROB:
        return
    end_s = time.time() - session_start
    start_s = max(0.0, end_s - len(audio) / SAMPLE_RATE)
    session.append_segment(text, start_s, end_s)
    mm, ss = divmod(int(start_s), 60)
    print(f"[{mm:02d}:{ss:02d}] {text}", flush=True)


# ---------------------------------------------------------------------------
# Question-capture helper
# ---------------------------------------------------------------------------

def _is_truncated(text: str) -> bool:
    """True when the transcript looks like a mid-sentence fragment."""
    if not text:
        return False
    words = text.split()
    if len(words) < 4:
        return True
    last_word = words[-1].lower().rstrip(".,!?;:")
    return last_word in _TRUNCATION_TERMINALS and text[-1] not in ".!?"


def _handle_question_utterance(
    audio: np.ndarray,
    model: WhisperModel,
    session: LectureSession,
    recorder: "AudioRecorder",
) -> None:
    print("[Q] transcribing combined buffer audio...", flush=True)
    text, confidence = transcribe_audio(model, audio, session.whisper_initial_prompt)
    print(f"[Q] transcribed: {text!r} (confidence {confidence:.2f})", flush=True)

    # Truncation guard: wait for continuation if the transcript looks incomplete
    if _is_truncated(text):
        print(
            f"[Q] transcript looks truncated, waiting up to {QA_TRUNCATION_WAIT}s for continuation...",
            flush=True,
        )
        extra = recorder.get_utterance(timeout=QA_TRUNCATION_WAIT)
        if extra is not None:
            audio = np.concatenate([audio, extra])
            text, confidence = transcribe_audio(model, audio, session.whisper_initial_prompt)
            print(f"[Q] retranscribed: {text!r} (confidence {confidence:.2f})", flush=True)
        else:
            print(f"[Q] no continuation in {QA_TRUNCATION_WAIT}s, using as-is", flush=True)

    reasons = []
    if not text:
        reasons.append("empty")
    if text and len(text.split()) < MIN_WORDS:
        reasons.append(f"too short ({len(text.split())} words < {MIN_WORDS})")
    if confidence < MIN_AVG_LOGPROB:
        reasons.append(f"low confidence ({confidence:.2f} < {MIN_AVG_LOGPROB})")
    if reasons:
        print(f"[Q] rejected — {', '.join(reasons)} — returning to LECTURE", flush=True)
        session.set_mode(Mode.LECTURE)
        return

    session.pending_question = text
    session.set_mode(Mode.PROCESSING)

    if session._qa_handler:
        print("[Q] firing _qa_handler thread", flush=True)
        threading.Thread(target=session._qa_handler, args=(session,), daemon=True).start()
    else:
        print("[Q] no _qa_handler registered!", flush=True)
        session.set_mode(Mode.LECTURE)


# ---------------------------------------------------------------------------
# Public API for app.py
# ---------------------------------------------------------------------------

def start_live_session(
    device: int | None = None,
    name: str = "",
    linked_documents: list[str] | None = None,
) -> LectureSession:
    global _active_session, _session_start
    if _active_session is not None and _active_session.is_active:
        raise RuntimeError("A recording session is already active.")
    session = LectureSession(name=name, linked_documents=linked_documents)
    session.start()  # must come first — initialises session._collection
    session.whisper_initial_prompt = build_whisper_prompt(session)
    print(
        f"[Whisper] no_speech_threshold={NO_SPEECH_THRESHOLD} "
        f"log_prob_threshold={LOG_PROB_THRESHOLD} "
        f"vad_silence_ms={VAD_SILENCE_MS}",
        flush=True,
    )
    _active_session = session
    _session_start = time.time()
    return session


def stop_live_session() -> LectureSession | None:
    global _active_session
    session = _active_session
    if session and session.is_active:
        session.stop()
    _active_session = None
    return session


def get_current_session() -> LectureSession | None:
    return _active_session


def pause_recording() -> None:
    """No-op: browser mic cannot be paused server-side.

    Audio arriving during Mode.PROCESSING is discarded in handle_audio_chunk.
    """
    pass


def resume_recording() -> None:
    """No-op: browser mic resumes automatically.

    Mode is reset to LECTURE by _resume_after_playback in app.py.
    """
    pass


# ---------------------------------------------------------------------------
# Browser audio entry points — called from app.py stream handler
# ---------------------------------------------------------------------------

def push_audio(audio_16k: np.ndarray) -> None:
    """Transcribe a 16kHz float32 buffer and commit segments to the active session."""
    if _active_session is None:
        return
    model = _get_model()
    process_utterance(audio_16k, model, _active_session, _session_start)


def push_question_audio(audio_16k: np.ndarray) -> None:
    """Transcribe a 16kHz float32 buffer as a Q&A question and dispatch to the handler."""
    session = _active_session
    if session is None:
        return
    model = _get_model()
    print("[Q] transcribing question buffer...", flush=True)
    text, confidence = transcribe_audio(model, audio_16k, session.whisper_initial_prompt)
    print(f"[Q] transcribed: {text!r} (confidence {confidence:.2f})", flush=True)

    reasons = []
    if not text:
        reasons.append("empty")
    elif len(text.split()) < MIN_WORDS:
        reasons.append(f"too short ({len(text.split())} words < {MIN_WORDS})")
    if confidence < MIN_AVG_LOGPROB:
        reasons.append(f"low confidence ({confidence:.2f} < {MIN_AVG_LOGPROB})")
    if reasons:
        print(f"[Q] rejected — {', '.join(reasons)} — returning to LECTURE", flush=True)
        session.set_mode(Mode.LECTURE)
        return

    session.pending_question = text
    session.set_mode(Mode.PROCESSING)

    if session._qa_handler:
        print("[Q] firing _qa_handler thread", flush=True)
        threading.Thread(target=session._qa_handler, args=(session,), daemon=True).start()
    else:
        print("[Q] no _qa_handler registered!", flush=True)
        session.set_mode(Mode.LECTURE)


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description="Real-time lecture transcription")
    parser.add_argument("--device", type=int, default=None, help="Input device index")
    args = parser.parse_args()

    print(f"Loading Whisper {WHISPER_MODEL!r} (CUDA float16)…")
    _get_model()  # pre-load so first utterance isn't slow

    session = start_live_session(device=args.device)
    print(f"Session: {session.session_id}")
    print("Listening — press Ctrl+C to stop.\n")

    def shutdown(sig, frame) -> None:
        print("\nStopping…")
        stop_live_session()
        sys.exit(0)

    signal.signal(signal.SIGINT, shutdown)
    while True:
        time.sleep(1)


if __name__ == "__main__":
    main()
