"""
live_transcribe.py — real-time lecture transcription.
Usage: python live_transcribe.py [--device <index>]
"""

import argparse
import signal
import sys
import threading
import time

import numpy as np
from faster_whisper import WhisperModel

from recorder import SAMPLE_RATE, AudioRecorder
from session import LectureSession, Mode

WHISPER_MODEL = "small"
MIN_WORDS = 3
MIN_AVG_LOGPROB = -1.0

# ---------------------------------------------------------------------------
# Module-level singletons — managed by start/stop_live_session()
# ---------------------------------------------------------------------------

_whisper_model: WhisperModel | None = None
_active_session: LectureSession | None = None
_active_recorder: AudioRecorder | None = None
_worker_thread: threading.Thread | None = None
_stop_event = threading.Event()
_session_start: float = 0.0


# ---------------------------------------------------------------------------
# Transcription helpers
# ---------------------------------------------------------------------------

def _get_model() -> WhisperModel:
    global _whisper_model
    if _whisper_model is None:
        _whisper_model = WhisperModel(WHISPER_MODEL, device="cuda", compute_type="float16")
    return _whisper_model


def transcribe_audio(model: WhisperModel, audio: np.ndarray) -> tuple[str, float]:
    segments_gen, _ = model.transcribe(audio, language="en", beam_size=5)
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
    text, confidence = transcribe_audio(model, audio)
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

def _handle_question_utterance(
    audio: np.ndarray,
    model: WhisperModel,
    session: LectureSession,
) -> None:
    print("[Q] utterance received, transcribing...", flush=True)
    text, confidence = transcribe_audio(model, audio)
    print(f"[Q] transcribed: {text!r} (confidence {confidence:.2f})", flush=True)
    if not text or len(text.split()) < MIN_WORDS or confidence < MIN_AVG_LOGPROB:
        print("[Q] below threshold — returning to LECTURE", flush=True)
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
# Background thread
# ---------------------------------------------------------------------------

def _transcription_loop(device: int | None, session: LectureSession, t0: float) -> None:
    global _active_recorder
    recorder = AudioRecorder(device=device)
    _active_recorder = recorder
    recorder.start()
    model = _get_model()
    while not _stop_event.is_set():
        audio = recorder.get_utterance(timeout=1.0)
        if audio is not None:
            mode = session.current_mode()
            if mode == Mode.LECTURE:
                process_utterance(audio, model, session, t0)
            elif mode == Mode.AWAITING_QUESTION:
                _handle_question_utterance(audio, model, session)
            # Mode.PROCESSING: discard — defense in depth while handler runs
    # Drain VAD buffer flushed on stop
    recorder.stop()
    while True:
        audio = recorder.get_utterance(timeout=0.5)
        if audio is None:
            break
        process_utterance(audio, model, session, t0)
    _active_recorder = None


# ---------------------------------------------------------------------------
# Public API for app.py
# ---------------------------------------------------------------------------

def start_live_session(
    device: int | None = None,
    name: str = "",
    linked_documents: list[str] | None = None,
) -> LectureSession:
    global _active_session, _worker_thread, _stop_event, _session_start
    if _worker_thread and _worker_thread.is_alive():
        raise RuntimeError("A recording session is already active.")
    _stop_event = threading.Event()
    session = LectureSession(name=name, linked_documents=linked_documents)
    session.start()
    _active_session = session
    _session_start = time.time()
    _worker_thread = threading.Thread(
        target=_transcription_loop, args=(device, session, _session_start), daemon=True
    )
    _worker_thread.start()
    return session


def stop_live_session() -> LectureSession | None:
    global _active_session
    if not (_worker_thread and _worker_thread.is_alive()):
        return _active_session
    _stop_event.set()
    _worker_thread.join(timeout=15.0)
    session = _active_session
    if session and session.is_active:
        session.stop()
    _active_session = None
    return session


def get_current_session() -> LectureSession | None:
    return _active_session


def pause_recording() -> None:
    """Stop the mic stream during TTS playback so we don't transcribe our own voice."""
    if _active_recorder and _active_recorder._stream:
        try:
            _active_recorder._stream.stop()
        except Exception:
            pass  # already stopped


def resume_recording() -> None:
    """Restart the mic stream after TTS finishes."""
    if _active_recorder and _active_recorder._stream:
        try:
            _active_recorder._stream.start()
        except Exception:
            pass  # already running


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description="Real-time lecture transcription")
    parser.add_argument("--device", type=int, default=None, help="Input device index")
    args = parser.parse_args()

    print("Loading Whisper small (CUDA float16)…")
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
