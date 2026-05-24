"""
recorder.py — AudioRecorder: microphone capture + silero-VAD → utterance queue.
"""

import queue
import threading

import numpy as np
import sounddevice as sd
from silero_vad import VADIterator, load_silero_vad

SAMPLE_RATE = 16_000
CHUNK_SIZE = 512          # silero-vad requires exactly 512 samples at 16 kHz
MAX_SPEECH_SECONDS = 30   # hard-cap utterance length so Whisper stays fast


class AudioRecorder:
    def __init__(self, device: int | None = None):
        self._device = device
        self._utterance_q: queue.Queue[np.ndarray] = queue.Queue()
        self._raw_q: queue.Queue[np.ndarray] = queue.Queue()
        self._stop_event = threading.Event()

        self._model = load_silero_vad()
        self._vad = VADIterator(self._model, sampling_rate=SAMPLE_RATE)

        self._stream: sd.InputStream | None = None
        self._worker: threading.Thread | None = None

    # ------------------------------------------------------------------
    # Audio callback — runs in sounddevice's real-time thread.
    # Must be fast: just push data, no VAD here.
    # ------------------------------------------------------------------

    def _audio_callback(
        self, indata: np.ndarray, frames: int, time, status
    ) -> None:
        self._raw_q.put(indata[:, 0].copy())

    # ------------------------------------------------------------------
    # VAD worker — separate thread so torch never touches the RT thread.
    # ------------------------------------------------------------------

    def _vad_worker(self) -> None:
        chunk_buf: list[float] = []
        speech_buf: list[float] = []
        in_speech = False

        while not (self._stop_event.is_set() and self._raw_q.empty()):
            try:
                audio = self._raw_q.get(timeout=0.1)
            except queue.Empty:
                continue

            chunk_buf.extend(audio.tolist())

            while len(chunk_buf) >= CHUNK_SIZE:
                window = np.array(chunk_buf[:CHUNK_SIZE], dtype=np.float32)
                chunk_buf = chunk_buf[CHUNK_SIZE:]

                event = self._vad(window, return_seconds=False)

                if event and "start" in event:
                    in_speech = True
                    speech_buf = list(window)
                elif event and "end" in event:
                    if in_speech:
                        speech_buf.extend(window.tolist())
                        self._utterance_q.put(np.array(speech_buf, dtype=np.float32))
                        speech_buf = []
                        in_speech = False
                elif in_speech:
                    speech_buf.extend(window.tolist())
                    # Force-emit at 30 s so Whisper receives a bounded clip
                    if len(speech_buf) >= MAX_SPEECH_SECONDS * SAMPLE_RATE:
                        self._utterance_q.put(np.array(speech_buf, dtype=np.float32))
                        speech_buf = []
                        in_speech = False
                        self._vad.reset_states()

        # Flush any speech that was mid-utterance when stop() was called
        if speech_buf:
            self._utterance_q.put(np.array(speech_buf, dtype=np.float32))

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def start(self) -> None:
        self._stream = sd.InputStream(
            samplerate=SAMPLE_RATE,
            channels=1,
            dtype="float32",
            blocksize=CHUNK_SIZE,
            device=self._device,
            callback=self._audio_callback,
        )
        self._worker = threading.Thread(target=self._vad_worker, daemon=True)
        self._stream.start()
        self._worker.start()

    def stop(self) -> None:
        if self._stream:
            self._stream.stop()
            self._stream.close()
            self._stream = None
        self._stop_event.set()
        if self._worker:
            self._worker.join(timeout=5.0)

    def get_utterance(self, timeout: float = 1.0) -> np.ndarray | None:
        try:
            return self._utterance_q.get(timeout=timeout)
        except queue.Empty:
            return None
