"""Speech-to-text for the rating app's note box (faster-whisper, local, CPU).

Typing the note was the biggest cost per rating (pilot: ~3.4 s per word, a
median note of 26 words). Speaking is several times faster.

Choices, measured on a 14 s spoken note full of production jargon:
  small.en + vocabulary prompt   3.1 s on CPU, word-perfect ("reese bass")
  small.en, no prompt            2.8 s, "re-space" for "reese bass"
  distil-large-v3 (+/- prompt)   7.0 s, still "re-space" - bigger isn't better here
CPU int8 keeps the GPU free for pipeline stages running alongside.

Audio never touches disk: the browser decodes its recording to 16 kHz mono
float32 samples and posts those (so the server needs no codec - PyAV 19 also
breaks faster-whisper 1.2.1's file decoding), and only the text is kept.

The model is a plain folder under data/models/whisper/, not the Hugging Face
cache, whose symlinks need admin rights or Developer Mode on Windows.
"""

import logging
import threading

import numpy as np
from stemsim import config

logger = logging.getLogger(__name__)

MODEL = "small.en"
MODEL_DIR = config.DATA_DIR / "models" / "whisper"
SAMPLE_RATE = 16_000
MAX_SECONDS = 120
# Whisper spells jargon better when it has seen it (see the table above)
PROMPT = (
    "Music production notes comparing songs: reese bass, 808, sub bass, kick, snare, hi-hats, claps, "
    "percussion, pads, synths, arpeggios, vocal chops, reverb, delay, sidechain, groove, lo-fi."
)


def _load_whisper(name):
    from faster_whisper import WhisperModel, download_model  # heavy; optional dependency

    path = download_model(name, output_dir=str(MODEL_DIR / name))
    return WhisperModel(path, device="cpu", compute_type="int8")


class Dictation:
    """Loads the model in the background at startup, so the first note
    doesn't wait for it; `status` is "loading", "ready" or "unavailable"."""

    def __init__(self, name=MODEL, loader=_load_whisper):
        self.status = "loading"
        self.error = None
        self._model = None
        self.loaded = threading.Event()  # set once status is final
        self._lock = threading.Lock()  # one transcription at a time - ctranslate2 models aren't shared
        threading.Thread(target=self._load, args=(name, loader), daemon=True).start()

    def _load(self, name, loader):
        try:
            self._model = loader(name)
            self.status = "ready"
        except Exception as exc:  # missing package, failed download: the app works without dictation
            self.error = f"{type(exc).__name__}: {exc}"
            self.status = "unavailable"
            logger.warning("Dictation unavailable: %s", self.error)
        finally:
            self.loaded.set()

    def transcribe(self, pcm_bytes):
        """Raw little-endian float32 samples at SAMPLE_RATE -> text."""
        if self.status != "ready":
            raise RuntimeError(f"dictation is {self.status}" + (f" ({self.error})" if self.error else ""))
        samples = samples_from_bytes(pcm_bytes)
        with self._lock:
            segments, _ = self._model.transcribe(
                samples, language="en", initial_prompt=PROMPT, beam_size=5, vad_filter=True
            )
            return " ".join(segment.text.strip() for segment in segments).strip()


def samples_from_bytes(pcm_bytes):
    if len(pcm_bytes) % 4:
        raise ValueError("audio must be float32 samples")
    samples = np.frombuffer(pcm_bytes, dtype="<f4")
    if not len(samples):
        raise ValueError("no audio")
    if len(samples) > MAX_SECONDS * SAMPLE_RATE:
        raise ValueError(f"recording longer than {MAX_SECONDS} s")
    return samples.astype(np.float32)
