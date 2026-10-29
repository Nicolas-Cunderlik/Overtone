"""Audio -> stems -> embeddings: the one analysis behind every stem vector.

The research pipeline runs it over the catalog (stemsim's embed stage) and
the app runs it on a live song's preview, so the two are comparable:

    wav, sr = load_audio(path)
    result = analyse(wav, sr)          # Demucs stems + MuQ-MuLan, full audio
    result.emb[0], result.rms_db[0]    # (channels, 512), (channels,) in stems.CHANNELS order

Per window and channel: a 512-d embedding (not normalized) and the RMS level
in dB, for silence gating - a near-silent stem's embedding is meaningless and
matches every other silent stem. Window 0 is always the full audio.

Separation is deterministic: Demucs' default "shift trick" (apply_model
shifts=1) offsets the input by a random amount on every call, so the same
track gets different stems each run. shifts=0 makes separation exactly
repeatable; the trick's ~0.2 dB SDR gain doesn't matter for embeddings.

Needs torch, Demucs and MuQ; the models load on first use (GPU if present),
once per process. Loads the official htdemucs checkpoint from Ultimate Vocal
Remover's model folder when present, else Demucs downloads it.
"""

import logging
import os
import threading
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import soundfile as sf
import torch
import torchaudio.functional as AF

from .stems import CHANNELS, DEMUCS_MODEL, EMBED_DIMS, EMBED_MODEL, MIN_WINDOW_S, STEM_SR, STEMS

logger = logging.getLogger(__name__)

# Ultimate Vocal Remover's copy of the official checkpoint; None without LOCALAPPDATA
# (e.g. Linux), rather than a path relative to the working directory
UVR_DEMUCS_REPO = (
    Path(os.environ["LOCALAPPDATA"]) / "Programs/Ultimate Vocal Remover/models/Demucs_Models/v3_v4_repo"
    if os.environ.get("LOCALAPPDATA")
    else None
)


class AudioTooShort(ValueError):
    """The audio is shorter than stems.MIN_WINDOW_S."""

    def __init__(self, duration_s):
        super().__init__(f"audio is {duration_s:.3g} s, shorter than {MIN_WINDOW_S:g} s")
        self.duration_s = duration_s


@dataclass
class Analysis:
    windows: np.ndarray  # (W, 2) float32 start/end seconds; window 0 is the full audio
    emb: np.ndarray  # (W, channels, EMBED_DIMS) float32
    rms_db: np.ndarray  # (W, channels) float32


def load_audio(path):
    """(2, samples) float32 at the file's own rate."""
    wav, sr = sf.read(path, dtype="float32", always_2d=True)
    wav = wav.T
    if wav.shape[0] == 1:
        wav = np.repeat(wav, 2, axis=0)
    return wav[:2], sr


def analyse(wav, sr, extra_windows=(), separator=None, embedder=None):
    """Separates `wav` ((2, samples) at `sr`) into stems in memory and embeds
    the mix and every stem: the full audio, plus each of `extra_windows`
    ((start, end) seconds) that lies inside it. Raises AudioTooShort."""
    return embed_channels(separate(wav, sr, separator), extra_windows, embedder)


def separate(wav, sr, separator=None):
    """(channels, samples) mono at STEM_SR, in CHANNELS order: the mix, then
    the Demucs stems."""
    separator = separator or get_separator()
    started = time.monotonic()
    stems = separator(wav, sr)
    logger.debug("Separated %.1f s of audio in %.1f s", wav.shape[1] / sr, time.monotonic() - started)
    return stack([mix_channel(wav, sr), *(stems[name].cpu() for name in STEMS)])


def mix_channel(wav, sr):
    """The mix channel: `wav` downmixed to mono at STEM_SR."""
    return AF.resample(torch.from_numpy(wav).mean(0), sr, STEM_SR)


def stack(channels):
    """Channel waveforms trimmed to the shortest and stacked: (channels, samples)."""
    length = min(len(c) for c in channels)
    return torch.stack([c[:length] for c in channels])


def embed_channels(channels, extra_windows=(), embedder=None):
    """Embeds (channels, samples) at STEM_SR: the full audio plus each of
    `extra_windows` that starts inside it (clipped to its end). Windows
    shorter than MIN_WINDOW_S are dropped. Raises AudioTooShort."""
    duration = channels.shape[1] / STEM_SR
    windows = [(0.0, duration)] + [(start, min(end, duration)) for start, end in extra_windows if start < duration]
    windows = [w for w in dict.fromkeys(windows) if w[1] - w[0] >= MIN_WINDOW_S]
    if not windows:  # extra windows lie inside the full audio, so it's too short too
        raise AudioTooShort(duration)
    embedder = embedder or get_embedder()
    started = time.monotonic()
    emb = np.zeros((len(windows), len(CHANNELS), EMBED_DIMS), np.float32)
    levels = np.zeros((len(windows), len(CHANNELS)), np.float32)
    for w, (start, end) in enumerate(windows):
        excerpt = channels[:, int(start * STEM_SR) : int(end * STEM_SR)]
        emb[w] = embedder(excerpt)
        levels[w] = [rms_db(channel) for channel in excerpt]
    logger.debug("Embedded %d window(s) in %.1f s", len(windows), time.monotonic() - started)
    return Analysis(np.array(windows, np.float32), emb, levels)


def rms_db(x):
    return float(20 * torch.log10(x.pow(2).mean().sqrt() + 1e-9))


class Separator:
    def __init__(self, device=None):
        from demucs.pretrained import get_model  # heavy import; only when separation actually runs

        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        if UVR_DEMUCS_REPO is not None and (UVR_DEMUCS_REPO / f"{DEMUCS_MODEL}.yaml").exists():
            model = get_model(DEMUCS_MODEL, repo=UVR_DEMUCS_REPO)
        else:
            model = get_model(DEMUCS_MODEL)
        self.model = model.to(self.device).eval()

    def __call__(self, wav, sr):
        """{stem: mono tensor at STEM_SR} for a (2, samples) array."""
        from demucs.apply import apply_model

        x = torch.from_numpy(wav).to(self.device)
        if sr != self.model.samplerate:
            x = AF.resample(x, sr, self.model.samplerate)
        ref = x.mean(0)
        mean, std = ref.mean(), ref.std() + 1e-8  # Demucs expects normalized input
        with torch.no_grad():
            out = apply_model(
                self.model, ((x - mean) / std)[None], device=self.device, shifts=0, split=True, overlap=0.25
            )[0]
        out = out * std + mean
        return {
            name: AF.resample(out[i].mean(0), self.model.samplerate, STEM_SR)
            for i, name in enumerate(self.model.sources)
        }


class MuQEmbedder:
    def __init__(self, device=None):
        from muq import MuQMuLan  # heavy import; only when embedding actually runs

        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.model = MuQMuLan.from_pretrained(EMBED_MODEL).to(self.device).eval()

    def __call__(self, waves):
        """(channels, samples) at STEM_SR -> (channels, EMBED_DIMS) numpy."""
        with torch.no_grad():
            return self.model(wavs=waves.to(self.device)).float().cpu().numpy()


_models = {}
_models_lock = threading.Lock()


def get_separator():
    """One Separator per process, shared by every caller."""
    return _shared("separator", Separator)


def get_embedder():
    """One MuQ model per process, shared by every caller: a second load
    doesn't get the first copy's memory back and can exhaust a 16 GB machine,
    killing the process silently."""
    return _shared("embedder", MuQEmbedder)


def _shared(name, factory):
    with _models_lock:  # the app may ask from two worker threads at once
        if name not in _models:
            logger.info("Loading %s (%s)", name, factory.__name__)
            started = time.monotonic()
            _models[name] = factory()
            logger.info("Loaded %s on %s in %.1f s", name, _models[name].device, time.monotonic() - started)
        return _models[name]
