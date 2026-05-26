"""
tts.py — Text-to-speech via the piper-tts Python package.
"""

import io
import sys
import threading
import wave
from pathlib import Path

import numpy as np
import sounddevice as sd
from piper.voice import PiperVoice

MODELS_DIR = Path(__file__).parent / "models" / "piper"
DEFAULT_MODEL = "en_US-lessac-medium.onnx"


class PiperTTS:
    def __init__(self, model_path: str | Path | None = None):
        if model_path is None:
            model_path = MODELS_DIR / DEFAULT_MODEL
        model_path = Path(model_path)
        config_path = Path(str(model_path) + ".json")
        self._voice = PiperVoice.load(str(model_path), config_path=str(config_path))
        self._sample_rate: int = self._voice.config.sample_rate
        self._lock = threading.Lock()

    def speak(self, text: str) -> None:
        """Synthesize text and play through the default output device. Blocking."""
        wav_io = io.BytesIO()
        with wave.open(wav_io, "wb") as wav_out:
            wav_out.setnchannels(1)
            wav_out.setsampwidth(2)  # 16-bit PCM
            wav_out.setframerate(self._sample_rate)
            self._voice.synthesize(text, wav_out)
        wav_io.seek(0)
        with wave.open(wav_io, "rb") as wav_in:
            raw = wav_in.readframes(wav_in.getnframes())
        audio = np.frombuffer(raw, dtype=np.int16).astype(np.float32) / 32768.0
        with self._lock:
            sd.play(audio, samplerate=self._sample_rate, blocking=True)

    def stop(self) -> None:
        """Interrupt current playback."""
        sd.stop()


if __name__ == "__main__":
    text = sys.argv[1] if len(sys.argv) > 1 else "Hello, this is a test."
    print(f"Loading model from {MODELS_DIR / DEFAULT_MODEL} ...")
    tts = PiperTTS()
    print(f"Speaking: {text!r}")
    tts.speak(text)
