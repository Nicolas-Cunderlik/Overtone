"""Stage: embed the mix and each stem with a frozen audio model (MuQ-MuLan).

One stage per model ("embed:<model key>"), so adding CLAP or MERT later is a
second run over the same stems, never a migration.

The analysis is overtone_core.ml.stem_analysis - the same code the app runs
on a live song - so catalog vectors and live vectors are comparable.

Windows: every track gets its full audio, plus every excerpt a dataset's
listeners judged on it (memberships' clip_start/clip_end - e.g. the 10-20 s
excerpts of the author's ratings, CCMSim's 4-24 s). Evaluation scores a
judgment on the window its raters heard when there is one.

A track whose audio is shorter than stems.MIN_WINDOW_S (e.g. a ~1 s stub
Deezer serves instead of a preview) is recorded as unavailable, so `done`
always means window 0 holds the full audio.

Stems come from the separate stage's FLACs for perceptual datasets; for the
rest (the catalog - the candidate pool) the analysis separates in memory and
only the vectors are kept (see separate.py for why). A track that has stems
on disk always uses them, whichever dataset it's processed under.

Output: data/embeddings/<model>/<h[:2]>/<h>.npz with
  windows (W, 2) seconds, channels (5,), emb (W, 5, 512) float32, rms_db (W, 5)
written via a temp file + rename.
"""

import json
import logging
import os
import time

import numpy as np
import soundfile as sf
import torch
from overtone_core.ml import stem_analysis
from overtone_core.ml.stem_analysis import AudioTooShort, get_embedder, get_separator, load_audio
from overtone_core.ml.stems import CHANNELS, STEMS

from .. import config
from ..registry import DONE, FAILED, UNAVAILABLE
from .separate import STAGE as SEPARATE_STAGE
from .separate import STEM_EXT, track_hash

logger = logging.getLogger(__name__)

MODEL_KEY = config.EMBED_MODEL
STAGE = config.EMBED_STAGE


def embedding_path(registry, track_id, model_key=MODEL_KEY):
    h = track_hash(track_id)
    return registry.data_dir / "embeddings" / model_key / h[:2] / f"{h}.npz"


def run(registry, dataset=None, limit=None):
    stems_on_disk = dataset is not None and registry.keeps_stems(dataset)
    requires = SEPARATE_STAGE if stems_on_disk else config.AUDIO_STAGE
    pending = registry.pending(STAGE, dataset=dataset, limit=limit, requires=requires)
    if not pending:
        return {}
    embedder = get_embedder()
    separator = None if stems_on_disk else get_separator()
    counts, started = {}, time.time()
    for i, track in enumerate(pending, start=1):
        state, detail, artifact = _embed_one(registry, embedder, separator, track["track_id"])
        registry.set_status(track["track_id"], STAGE, state, detail, artifact)
        counts[state] = counts.get(state, 0) + 1
        if i % 50 == 0:
            rate = (time.time() - started) / i
            logger.info("embed: %d/%d, %.2f s/track, %s", i, len(pending), rate, counts)
    return counts


def load_channels(registry, track_id, separator=None):
    """(5, samples) mono at STEM_SR in CHANNELS order: the mix (from the audio
    stage) + the 4 stems - read from disk when the separate stage kept them,
    else separated here with `separator`."""
    wav, sr = load_audio(registry.output(track_id, config.AUDIO_STAGE)[0])
    on_disk = registry.output(track_id, SEPARATE_STAGE)
    if on_disk is None:
        if separator is None:
            raise FileNotFoundError("no stems on disk and no separator to make them")
        return stem_analysis.separate(wav, sr, separator), False
    channels = [stem_analysis.mix_channel(wav, sr)]
    for name in STEMS:
        wave, _ = sf.read(on_disk[0] / f"{name}.{STEM_EXT}", dtype="float32")
        channels.append(torch.from_numpy(wave))
    return stem_analysis.stack(channels), True


def _embed_one(registry, embedder, separator, track_id):
    try:
        channels, from_disk = load_channels(registry, track_id, separator)
        result = stem_analysis.embed_channels(channels, registry.clip_windows(track_id), embedder)
    except AudioTooShort as exc:
        return UNAVAILABLE, str(exc), None
    except Exception as exc:  # missing/corrupt audio or stems, CUDA OOM - recorded, retried next run
        return FAILED, f"{type(exc).__name__}: {exc}", None

    path = embedding_path(registry, track_id)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.stem + ".tmp.npz")
    np.savez(tmp, windows=result.windows, channels=np.array(CHANNELS), emb=result.emb, rms_db=result.rms_db)
    tmp.replace(path)
    detail = json.dumps({"windows": len(result.windows), "stems": "disk" if from_disk else "memory"})
    return DONE, detail, os.path.relpath(path, registry.data_dir)
