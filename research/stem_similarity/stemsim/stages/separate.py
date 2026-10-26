"""Stage: split each track's audio into drums / bass / vocals / other (Demucs
htdemucs, 4 stems) and keep the stems on disk - for perceptual-dataset tracks.

Why on disk: separation is the expensive step (~0.9 s/track on an RTX 3070
vs ~0.2 s to embed), and the perceptual tracks are the ones every experiment
re-reads (other embedding models, the DSP comparator). Catalog tracks - the
candidate pool, ~3x as many - are separated in memory by the embed stage and
keep only their vectors; stems for everything would need ~87 GB.

Format: 16-bit FLAC, mono, 24 kHz (MuQ's input rate), ~2.9 MB per 30 s track.
Lossy formats were measured and rejected: MuQ embeddings of Vorbis-encoded
stems drifted to cosine 0.44-0.97 against the in-memory stems (default
quality) and 0.89 at best quality; 16-bit FLAC stays >= 0.998.

Separation runs on the full audio file; choosing excerpts is the embedding
stage's job. Output: data/stems/<h[:2]>/<h>/<stem>.flac (h = hash of the
track id), written to a temp dir and renamed so a crash never leaves partial
stems.

Determinism: Demucs' default "shift trick" (apply_model shifts=1) offsets the
input by a random amount on every call, so the same track gave different
stems each run - drum embeddings agreed as little as cosine 0.69 between
runs (the pilot was affected). shifts=0 makes separation exactly repeatable;
the trick's ~0.2 dB SDR gain doesn't matter for embeddings.

Needs the research venv (torch + demucs). Loads the official htdemucs
checkpoint from Ultimate Vocal Remover's model folder when present.
"""

import hashlib
import json
import logging
import os
import shutil
import time
from pathlib import Path

import numpy as np
import soundfile as sf
import torch
import torchaudio.functional as AF
from demucs.apply import apply_model
from demucs.pretrained import get_model

from .. import config
from ..registry import DONE, FAILED

logger = logging.getLogger(__name__)

STAGE = config.SEPARATE_STAGE
REQUIRES = config.AUDIO_STAGE
STEMS = ("drums", "bass", "vocals", "other")
STEM_SR = 24_000
MIN_SECONDS = 1.0
STEM_FORMAT = {"format": "FLAC", "subtype": "PCM_16"}
STEM_EXT = "flac"
UVR_DEMUCS_REPO = (
    Path(os.environ.get("LOCALAPPDATA", "")) / "Programs/Ultimate Vocal Remover/models/Demucs_Models/v3_v4_repo"
)


def track_hash(track_id):
    return hashlib.sha1(track_id.encode(), usedforsecurity=False).hexdigest()[:20]


def stem_dir(registry, track_id):
    h = track_hash(track_id)
    return registry.data_dir / "stems" / h[:2] / h


def load_audio(path):
    """(2, samples) float32 at the file's own rate."""
    wav, sr = sf.read(path, dtype="float32", always_2d=True)
    wav = wav.T
    if wav.shape[0] == 1:
        wav = np.repeat(wav, 2, axis=0)
    return wav[:2], sr


class Separator:
    def __init__(self, device=None):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        if (UVR_DEMUCS_REPO / "htdemucs.yaml").exists():
            model = get_model("htdemucs", repo=UVR_DEMUCS_REPO)
        else:
            model = get_model("htdemucs")
        self.model = model.to(self.device).eval()

    def __call__(self, wav, sr):
        """{stem: mono tensor at STEM_SR} for a (2, samples) array."""
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


_SEPARATOR = None


def get_separator():
    """One Separator per process, reused across datasets (see embed.get_embedder)."""
    global _SEPARATOR
    if _SEPARATOR is None:
        _SEPARATOR = Separator()
    return _SEPARATOR


def run(registry, dataset=None, limit=None):
    """Separates pending tracks of a perceptual dataset; returns {state: count}.
    Other datasets are skipped - the embed stage separates those in memory."""
    if dataset is None or not registry.keeps_stems(dataset):
        return {}
    pending = registry.pending(STAGE, dataset=dataset, limit=limit, requires=REQUIRES)
    if not pending:
        return {}
    separator = get_separator()
    counts, started = {}, time.time()
    for i, track in enumerate(pending, start=1):
        state, detail, artifact = _separate_one(registry, separator, track)
        registry.set_status(track["track_id"], STAGE, state, detail, artifact)
        counts[state] = counts.get(state, 0) + 1
        if i % 25 == 0:
            rate = (time.time() - started) / i
            logger.info("separate: %d/%d, %.2f s/track, %s", i, len(pending), rate, counts)
    return counts


def _separate_one(registry, separator, track):
    source = registry.output(track["track_id"], REQUIRES)
    if source is None:
        return FAILED, "audio output missing", None
    try:
        wav, sr = load_audio(source[0])
        duration = wav.shape[1] / sr
        if duration < MIN_SECONDS:
            return FAILED, f"audio is only {duration:.2f} s", None
        stems = separator(wav, sr)
    except Exception as exc:  # corrupt file, decoder error, CUDA OOM - recorded, retried next run
        return FAILED, f"{type(exc).__name__}: {exc}", None

    final = stem_dir(registry, track["track_id"])
    tmp = final.with_name(final.name + ".tmp")
    shutil.rmtree(tmp, ignore_errors=True)
    tmp.mkdir(parents=True)
    for name in STEMS:
        sf.write(tmp / f"{name}.{STEM_EXT}", stems[name].cpu().numpy(), STEM_SR, **STEM_FORMAT)
    shutil.rmtree(final, ignore_errors=True)
    tmp.rename(final)
    detail = json.dumps({"duration_s": round(duration, 3), "stem_sr": STEM_SR})
    return DONE, detail, os.path.relpath(final, registry.data_dir)
