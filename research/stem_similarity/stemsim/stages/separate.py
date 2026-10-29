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

Separation itself is overtone_core.ml.stem_analysis (deterministic, the
same code the app runs on live songs).
"""

import hashlib
import json
import logging
import os
import shutil
import time

import soundfile as sf
from overtone_core.ml.stem_analysis import get_separator, load_audio
from overtone_core.ml.stems import STEM_SR, STEMS

from .. import config
from ..registry import DONE, FAILED

logger = logging.getLogger(__name__)

STAGE = config.SEPARATE_STAGE
REQUIRES = config.AUDIO_STAGE
MIN_SECONDS = 1.0
STEM_FORMAT = {"format": "FLAC", "subtype": "PCM_16"}
STEM_EXT = "flac"


def track_hash(track_id):
    return hashlib.sha1(track_id.encode(), usedforsecurity=False).hexdigest()[:20]


def stem_dir(registry, track_id):
    h = track_hash(track_id)
    return registry.data_dir / "stems" / h[:2] / h


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
