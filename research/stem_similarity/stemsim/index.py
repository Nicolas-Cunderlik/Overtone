"""Builds the next stem-muq model version: every embedded catalog song's
channel vectors, the fitted channel weights and the Spotify track IDs, for the
app to score candidates against.

    python -m stemsim index --weights-run <stem-weights MLflow run id>

Writes models/stem-muq/vN/ (overtone.ml.model_store) with the files named in
overtone.ml.paths:
  tracks.csv      one row per song, in the order of every array below:
                  track_id (registry), spotify_track_id, title, artist
  <channel>.npy   (songs, 512) float16 unit vectors of the full preview,
                  one file per channel (mix, drums, bass, vocals, other)
  rms_db.npy      (songs, channels) float32 level in dB, for silence gating
Manifest params: channels (column order of rms_db.npy), weights, z_mean,
z_std and silent_db - everything scoring needs besides the files.

A later facet (melody, DSP) adds a file and a channel name, not a new format.

Read-only on the registry. Scoring is brute force: 20.7k songs x 5 channels x
512 dims is ~53M multiply-adds per query, tens of ms in numpy. An ANN
shortlist only pays off at a few hundred thousand songs or more.
"""

import logging

import numpy as np
import pandas as pd

from overtone.ml import model_store
from overtone.ml.paths import MODELS_DIR, STEM_CHANNEL_FILE, STEM_LEVELS_FILE, STEM_MODEL, STEM_TRACKS_FILE

from . import config
from .registry import DONE
from .schema import CHANNELS

logger = logging.getLogger(__name__)

DATASET = "spotify_catalog"
EMBED_DIMS = 512
FULL_AUDIO = 0  # window index of the full preview in every embedding file (embed.py)
CHECK_QUERIES = 200
CHECK_TOP_K = 10


def catalog_songs(registry):
    """Catalog songs with a finished embedding, in a fixed order (by track_id)."""
    return registry.conn.execute(
        """SELECT t.track_id, json_extract(t.meta, '$.spotify_track_id') AS spotify_track_id,
                  t.title, t.artist, s.artifact
           FROM memberships m
           JOIN tracks t ON t.track_id = m.track_id
           JOIN stage_status s ON s.track_id = m.track_id AND s.stage = ? AND s.state = ?
           WHERE m.dataset = ?
           ORDER BY t.track_id""",
        (config.EMBED_STAGE, DONE, DATASET),
    ).fetchall()


def build(registry, weights, silent_db, source, root=MODELS_DIR):
    """Writes the next stem-muq version and returns its model_store build
    (.version, .path).

    weights: a stem-weights run's weights.json ("final", "z_mean", "z_std",
    each keyed by channel). source: what the manifest records about the inputs.
    """
    songs = catalog_songs(registry)
    if not songs:
        raise ValueError(f"no {DATASET} songs have embeddings ({config.EMBED_STAGE})")
    emb = np.zeros((len(songs), len(CHANNELS), EMBED_DIMS), np.float32)
    levels = np.zeros((len(songs), len(CHANNELS)), np.float32)
    for i, song in enumerate(songs):
        data = np.load(registry.data_dir / song["artifact"].replace("\\", "/"))  # stored with "\" on Windows
        if list(data["channels"]) != list(CHANNELS) or not len(data["windows"]):
            # embed.py marks a track done only with every channel and the full-audio window
            raise ValueError(
                f"{song['artifact']}: channels {list(data['channels'])}, {len(data['windows'])} windows - "
                f"expected {list(CHANNELS)} and at least the full audio"
            )
        emb[i] = data["emb"][FULL_AUDIO]
        levels[i] = data["rms_db"][FULL_AUDIO]
    emb /= np.linalg.norm(emb, axis=2, keepdims=True).clip(1e-9)
    stored = emb.astype(np.float16)

    params = {
        "channels": list(CHANNELS),
        "weights": {c: weights["final"][c] for c in CHANNELS},
        "z_mean": {c: weights["z_mean"][c] for c in CHANNELS},
        "z_std": {c: weights["z_std"][c] for c in CHANNELS},
        "silent_db": silent_db,
        "window": "full preview",
        "dtype": "float16",
        "float16_check": float16_check(emb, stored),
    }
    tracks = pd.DataFrame(
        [(s["track_id"], s["spotify_track_id"], s["title"], s["artist"]) for s in songs],
        columns=["track_id", "spotify_track_id", "title", "artist"],
    )
    source = {**source, "dataset": DATASET, "songs": len(songs), "embed_stage": config.EMBED_STAGE}
    with model_store.new_version(STEM_MODEL, source=source, params=params, root=root) as built:
        tracks.to_csv(built.path / STEM_TRACKS_FILE, index=False)
        for c, channel in enumerate(CHANNELS):
            np.save(built.path / STEM_CHANNEL_FILE.format(channel=channel), stored[:, c])
        np.save(built.path / STEM_LEVELS_FILE, levels)
    logger.info("wrote %s %s: %d songs to %s", STEM_MODEL, built.version, len(songs), built.path)
    return built


def float16_check(exact, stored, queries=CHECK_QUERIES, top_k=CHECK_TOP_K, seed=0):
    """How much storing vectors as float16 changes cosine similarity, per channel.

    For a random sample of songs, cosine to every song with the float32 vectors
    vs the float16 ones: the largest absolute difference, and the mean share of
    the float32 top-k neighbours that float16 keeps.
    """
    rng = np.random.default_rng(seed)
    sample = rng.choice(len(exact), size=min(queries, len(exact)), replace=False)
    k = min(top_k, len(exact))
    out = {}
    for c, channel in enumerate(CHANNELS):
        a = exact[sample, c] @ exact[:, c].T
        b = stored[sample, c].astype(np.float32) @ stored[:, c].astype(np.float32).T
        top_a = np.argpartition(-a, k - 1, axis=1)[:, :k]
        top_b = np.argpartition(-b, k - 1, axis=1)[:, :k]
        kept = [len(set(x) & set(y)) / k for x, y in zip(top_a, top_b, strict=True)]
        out[channel] = {"max_abs_error": float(np.abs(a - b).max()), f"top{k}_overlap": float(np.mean(kept))}
    return out
