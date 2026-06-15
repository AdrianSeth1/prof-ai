"""
spike/audio_spike.py — standalone realtime audio transport spike.

Proves: mic (browser AudioWorklet) → WebSocket → Whisper (CUDA) → transcript
        → TTS (Piper) → WAV bytes back → Web Audio playback in browser.

Run from the project root:
    python spike/audio_spike.py
    # or:
    uvicorn spike.audio_spike:app --host 127.0.0.1 --port 8100

Then open http://localhost:8100 in the browser.
"""

import asyncio
import concurrent.futures
import json
import sys
import time
from contextlib import asynccontextmanager
from pathlib import Path

import numpy as np
import uvicorn
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import HTMLResponse

# ── Project root on sys.path ──────────────────────────────────────
# Must be before the live_transcribe import; importing it also runs
# the Windows NVIDIA DLL PATH fix (_nvidia_dirs block at the top).
_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from live_transcribe import _get_model, transcribe_audio  # noqa: E402
from tts import PiperTTS                                  # noqa: E402

# ── Globals (populated at startup) ───────────────────────────────

_whisper = None
_tts:     PiperTTS | None = None
_POOL     = concurrent.futures.ThreadPoolExecutor(max_workers=2)

_INDEX_HTML = Path(__file__).parent / "index.html"


# ── Lifespan — preload both models before accepting connections ───

@asynccontextmanager
async def lifespan(app: FastAPI):
    global _whisper, _tts
    loop = asyncio.get_running_loop()

    print("[SPIKE] preloading Whisper (large-v3-turbo · CUDA float16)…", flush=True)
    _whisper = await loop.run_in_executor(_POOL, _get_model)
    print("[SPIKE] Whisper ready", flush=True)

    print("[SPIKE] preloading Piper TTS…", flush=True)
    _tts = await loop.run_in_executor(_POOL, PiperTTS)
    print(f"[SPIKE] TTS ready — sample_rate={_tts.sample_rate} Hz", flush=True)

    print("[SPIKE] listening on http://127.0.0.1:8100", flush=True)
    yield
    # nothing to teardown


app = FastAPI(title="Audio Spike", docs_url=None, lifespan=lifespan)


# ── Static page ───────────────────────────────────────────────────

@app.get("/")
def index():
    return HTMLResponse(_INDEX_HTML.read_text(encoding="utf-8"))


# ── WebSocket handler ─────────────────────────────────────────────

@app.websocket("/ws/audio")
async def ws_audio(ws: WebSocket):
    await ws.accept()
    loop   = asyncio.get_running_loop()
    buf: list[bytes] = []   # accumulated Int16 PCM chunks for the current utterance
    client = ws.client
    print(f"[SPIKE] client connected {client}", flush=True)

    async def send_tts(text: str) -> None:
        """Synthesize text and send tts_start / WAV bytes / tts_end."""
        t1 = time.perf_counter()
        tts_inst = _tts   # local ref so lambda captures correctly
        wav = await loop.run_in_executor(_POOL, lambda: tts_inst.synthesize(text))
        ms = int((time.perf_counter() - t1) * 1000)
        print(f"[SPIKE] TTS {len(wav):,} bytes in {ms} ms", flush=True)
        await ws.send_text(json.dumps({"type": "tts_start", "ms": ms}))
        await ws.send_bytes(wav)
        await ws.send_text(json.dumps({"type": "tts_end"}))

    try:
        while True:
            msg = await ws.receive()

            # ── disconnect ─────────────────────────────────────
            if msg.get("type") == "websocket.disconnect":
                break

            # ── binary: raw Int16 PCM frame ────────────────────
            raw_bytes = msg.get("bytes")
            if raw_bytes:
                buf.append(raw_bytes)
                continue

            # ── text: JSON control message ─────────────────────
            text_data = msg.get("text")
            if not text_data:
                continue

            try:
                data = json.loads(text_data)
            except json.JSONDecodeError:
                continue

            kind = data.get("type")

            # ── flush: end-of-utterance → transcribe ───────────
            if kind == "flush":
                if not buf:
                    continue

                raw = b"".join(buf)
                buf.clear()
                n_samples = len(raw) // 2   # Int16 = 2 bytes
                duration_ms = n_samples * 1000 // 16000
                print(
                    f"[SPIKE] flush: {n_samples} samples "
                    f"({duration_ms} ms) → transcribing…",
                    flush=True,
                )

                t0 = time.perf_counter()

                def do_transcribe(raw_bytes: bytes = raw) -> tuple[str, float]:
                    pcm_i16 = np.frombuffer(raw_bytes, dtype=np.int16)
                    pcm_f32 = pcm_i16.astype(np.float32) / 32768.0
                    return transcribe_audio(_whisper, pcm_f32, "")

                text, confidence = await loop.run_in_executor(_POOL, do_transcribe)
                whisper_ms = int((time.perf_counter() - t0) * 1000)

                print(
                    f"[SPIKE] transcript: {text!r}  "
                    f"conf={confidence:.3f}  {whisper_ms} ms",
                    flush=True,
                )

                await ws.send_text(json.dumps({
                    "type":       "transcript",
                    "text":       text,
                    "confidence": round(confidence, 3),
                    "ms":         whisper_ms,
                }))

                # Auto-echo non-empty transcript as TTS
                if text:
                    await send_tts(text)

            # ── speak: explicit TTS request (Speak last button) ─
            elif kind == "speak":
                speak_text = (data.get("text") or "").strip()
                if speak_text:
                    print(f"[SPIKE] explicit speak: {speak_text!r}", flush=True)
                    await send_tts(speak_text)

    except WebSocketDisconnect:
        pass
    except Exception:
        import traceback
        print(f"[SPIKE] ws error:\n{traceback.format_exc()}", flush=True)
    finally:
        print(f"[SPIKE] client disconnected {client}", flush=True)


# ── Entry point ───────────────────────────────────────────────────

if __name__ == "__main__":
    uvicorn.run(
        "spike.audio_spike:app",
        host="127.0.0.1",
        port=8100,
        reload=False,
        log_level="info",
    )
