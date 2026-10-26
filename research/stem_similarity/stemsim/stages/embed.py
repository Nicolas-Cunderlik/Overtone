"""Stage: embed the mix and each stem with a frozen audio model (MuQ-MuLan).

One stage per model ("embed:<model key>"), so adding CLAP or MERT later is a
second run over the same stems, never a migration.

Windows: every track gets its full audio, plus every excerpt a dataset's
listeners judged on it (memberships' clip_start/clip_end - e.g. the 10-20 s
excerpts of the author's ratings, CCMSim's 4-24 s). Evaluation scores a
judgment on the window its raters heard when there is one.

Per window and channel (mix, drums, bass, vocals, other): a 512-d embedding
and the RMS level in dB (for silence gating - a near-silent stem's embedding
is meaningless and matches every other silent stem).

Stems come from the separate stage's FLACs for perceptual datasets; for the
rest (the catalog - the candidate pool) this stage separates in memory and
keeps only the vectors (see separate.py for why). A track that has stems on
disk always uses them, whichever dataset it's processed under.

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
import torchaudio.functional as AF

from .. import config
from ..registry import DONE, FAILED
from .separate import STAGE as SEPARATE_STAGE
from .separate import STEM_EXT, STEM_SR, STEMS, get_separator, load_audio, track_hash

logger = logging.getLogger(__name__)

MODEL_KEY = config.EMBED_MODEL
HF_MODEL = "OpenMuQ/MuQ-MuLan-large"
STAGE = config.EMBED_STAGE
CHANNELS = ("mix", *STEMS)
MIN_WINDOW_S = 1.0


def embedding_path(registry, track_id, model_key=MODEL_KEY):
    h = track_hash(track_id)
    return registry.data_dir / "embeddings" / model_key / h[:2] / f"{h}.npz"


def rms_db(x):
    return float(20 * torch.log10(x.pow(2).mean().sqrt() + 1e-9))


class MuQEmbedder:
    def __init__(self, device=None):
        from muq import MuQMuLan  # heavy import; only when the stage actually runs

        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.model = MuQMuLan.from_pretrained(HF_MODEL).to(self.device).eval()

    def __call__(self, waves):
        """(channels, samples) at STEM_SR -> (channels, 512) numpy."""
        with torch.no_grad():
            return self.model(wavs=waves.to(self.device)).float().cpu().numpy()


_EMBEDDER = None


def get_embedder():
    """One MuQ model per process, reused across datasets. Loading it a second
    time in the same process ran a 16 GB machine out of RAM (the first copy's
    memory isn't fully returned) and crashed the stage silently."""
    global _EMBEDDER
    if _EMBEDDER is None:
        _EMBEDDER = MuQEmbedder()
    return _EMBEDDER


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
    """(5, samples) mono at STEM_SR: the mix (from the audio stage) + the 4
    stems - read from disk when the separate stage kept them, else separated
    here with `separator`."""
    mix, sr = load_audio(registry.output(track_id, config.AUDIO_STAGE)[0])
    channels = [AF.resample(torch.from_numpy(mix).mean(0), sr, STEM_SR)]
    on_disk = registry.output(track_id, SEPARATE_STAGE)
    if on_disk is not None:
        for name in STEMS:
            wave, _ = sf.read(on_disk[0] / f"{name}.{STEM_EXT}", dtype="float32")
            channels.append(torch.from_numpy(wave))
    elif separator is not None:
        stems = separator(mix, sr)
        channels += [stems[name].cpu() for name in STEMS]
    else:
        raise FileNotFoundError("no stems on disk and no separator to make them")
    length = min(len(c) for c in channels)
    return torch.stack([c[:length] for c in channels]), on_disk is not None


def _embed_one(registry, embedder, separator, track_id):
    try:
        channels, from_disk = load_channels(registry, track_id, separator)
        duration = channels.shape[1] / STEM_SR
        windows = [(0.0, duration)] + [
            (start, min(end, duration)) for start, end in registry.clip_windows(track_id) if start < duration
        ]
        windows = [w for w in dict.fromkeys(windows) if w[1] - w[0] >= MIN_WINDOW_S]
        emb = np.zeros((len(windows), len(CHANNELS), 512), np.float32)
        levels = np.zeros((len(windows), len(CHANNELS)), np.float32)
        for w, (start, end) in enumerate(windows):
            excerpt = channels[:, int(start * STEM_SR) : int(end * STEM_SR)]
            emb[w] = embedder(excerpt)
            levels[w] = [rms_db(channel) for channel in excerpt]
    except Exception as exc:  # missing/corrupt audio or stems, CUDA OOM - recorded, retried next run
        return FAILED, f"{type(exc).__name__}: {exc}", None

    path = embedding_path(registry, track_id)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.stem + ".tmp.npz")
    np.savez(tmp, windows=np.array(windows, np.float32), channels=np.array(CHANNELS), emb=emb, rms_db=levels)
    tmp.replace(path)
    detail = json.dumps({"windows": len(windows), "stems": "disk" if from_disk else "memory"})
    return DONE, detail, os.path.relpath(path, registry.data_dir)
