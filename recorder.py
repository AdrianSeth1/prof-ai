"""
recorder.py — AudioRecorder: microphone capture + silero-VAD -> utterance queue.
"""

import queue
import threading

import numpy as np
import sounddevice as sd
from silero_vad import VADIterator, load_silero_vad

SAMPLE_RATE = 16_000
CHUNK_SIZE = 512          # silero-vad requires exactly 512 samples at 16 kHz
MAX_SPEECH_SECONDS = 30   # hard-cap utterance length so Whisper stays fast
VAD_SILENCE_MS = 600      # ms of silence before VAD closes an utterance boundary

# Set to a substring of the mic name to pin by name regardless of device index.
# Example: "Blue Yeti" or "USB PnP Audio"
# When empty, preferred_index (from the UI dropdown) is tried first, then
# the first available MME input device is used as a final fallback.
INPUT_DEVICE_NAME = ""


def resolve_input_device(
    preferred_name: str | None = None,
    preferred_index: int | None = None,
) -> int:
    """Return a valid MME input device index, resolved at runtime.

    Resolution order:
    1. First device whose name contains preferred_name (case-insensitive),
       has input channels, and uses MME
    2. preferred_index if in range, has input channels, and uses MME
    3. First MME device with input channels
    4. Raises ValueError listing all available devices
    """
    devices = sd.query_devices()
    hostapis = sd.query_hostapis()

    mme_index = None
    for i, api in enumerate(hostapis):
        if "mme" in api["name"].lower():
            mme_index = i
            break

    def _device_list_str() -> str:
        lines = []
        for i, d in enumerate(devices):
            api_name = hostapis[d["hostapi"]]["name"] if d["hostapi"] < len(hostapis) else "?"
            lines.append(
                f"  [{i}] {d['name']!r}  api={api_name!r}  in={d['max_input_channels']}"
            )
        return "\n".join(lines)

    if mme_index is None:
        raise ValueError(
            f"MME host API not found. Available devices:\n{_device_list_str()}"
        )

    def _is_usable(d) -> bool:
        return d["max_input_channels"] > 0 and d["hostapi"] == mme_index

    # Priority 1: name match
    if preferred_name:
        for i, d in enumerate(devices):
            if preferred_name.lower() in d["name"].lower() and _is_usable(d):
                return i

    # Priority 2: stored/UI index (the old path, preserved as fallback)
    if preferred_index is not None and 0 <= preferred_index < len(devices):
        if _is_usable(devices[preferred_index]):
            return preferred_index

    # Priority 3: first available MME input device
    for i, d in enumerate(devices):
        if _is_usable(d):
            return i

    raise ValueError(
        f"No usable MME input device found. Available devices:\n{_device_list_str()}"
    )


class AudioRecorder:
    def __init__(self, device: int | None = None):
        self._device = device
        self._utterance_q: queue.Queue[np.ndarray] = queue.Queue()
        self._raw_q: queue.Queue[np.ndarray] = queue.Queue()
        self._stop_event = threading.Event()

        self._model = load_silero_vad()
        self._vad = VADIterator(self._model, sampling_rate=SAMPLE_RATE, min_silence_duration_ms=VAD_SILENCE_MS)

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
        resolved = resolve_input_device(
            preferred_name=INPUT_DEVICE_NAME if INPUT_DEVICE_NAME else None,
            preferred_index=self._device,
        )
        dev_name = sd.query_devices(resolved)["name"]
        print(f"[REC] input device {resolved}: {dev_name!r} (MME)", flush=True)
        self._stream = sd.InputStream(
            samplerate=SAMPLE_RATE,
            channels=1,
            dtype="float32",
            blocksize=CHUNK_SIZE,
            device=resolved,
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
