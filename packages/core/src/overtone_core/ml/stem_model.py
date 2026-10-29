"""The stem-similarity model: ranks catalog songs by how alike they sound,
mix and stems scored separately and combined with the fitted channel weights.

Reads a stem-muq version (see stemsim/index.py for the files and manifest
params). Scoring is the stem-weights experiment's: per channel, cosine
similarity z-scored over random audible pairs so channels share a scale,
with silence gating - a channel silent in exactly one song scores -1 (one
has bass, the other doesn't), silent in both scores 0 (no evidence either
way) - then the weighted sum over channels.

Numpy only: a catalog song is scored from its stored vectors without torch.
A song outside the catalog needs stem_analysis first.

    model = StemModel()
    row = model.find(spotify_track_id="...", title="...", artist="...")
    model.similar_to(*model.vectors(row), k=3, exclude_rows=[row])
"""

import numpy as np
import pandas as pd

from . import dedup, model_store
from .paths import MODELS_DIR, STEM_CHANNEL_FILE, STEM_LEVELS_FILE, STEM_MODEL, STEM_MODEL_VERSION, STEM_TRACKS_FILE
from .stems import CHANNELS


class StemModel:
    def __init__(self, version=STEM_MODEL_VERSION, root=MODELS_DIR):
        manifest = model_store.read_manifest(STEM_MODEL, version, root)
        params = manifest["params"]
        if tuple(params["channels"]) != CHANNELS:
            # vectors from stem_analysis come in CHANNELS order; a different index layout needs a new loader
            raise ValueError(f"{STEM_MODEL} {version} has channels {params['channels']}, expected {list(CHANNELS)}")
        self.version = version
        self.weights = np.array([params["weights"][c] for c in CHANNELS])
        self.z_mean = np.array([params["z_mean"][c] for c in CHANNELS])
        self.z_std = np.array([params["z_std"][c] for c in CHANNELS])
        self.silent_db = params["silent_db"]

        folder = model_store.version_dir(STEM_MODEL, version, root)
        self.tracks = pd.read_csv(folder / STEM_TRACKS_FILE, dtype=str, keep_default_na=False)
        # float32 for fast scoring: ~0.2 GB for 20k songs (stored as float16)
        self._emb = np.stack(
            [np.load(folder / STEM_CHANNEL_FILE.format(channel=c)).astype(np.float32) for c in CHANNELS], axis=1
        )
        self._levels = np.load(folder / STEM_LEVELS_FILE)
        self._row_by_spotify_id = {sid: i for i, sid in enumerate(self.tracks["spotify_track_id"]) if sid}
        self._row_by_song = {
            dedup.song_key(title, artist): i
            for i, (title, artist) in enumerate(zip(self.tracks["title"], self.tracks["artist"], strict=True))
        }

    def __len__(self):
        return len(self.tracks)

    def find(self, spotify_track_id=None, title=None, artist=None):
        """The catalog row of a song, by Spotify track ID, else by title and
        artist (the catalog's dedup rules, so "feat."/remaster variants of a
        catalog song count as it); None if it isn't in the catalog."""
        row = self._row_by_spotify_id.get(spotify_track_id)
        if row is None and title and artist:
            row = self._row_by_song.get(dedup.song_key(title, artist))
        return row

    def vectors(self, row):
        """A catalog song's (emb (channels, dims), rms_db (channels,)), as similar_to takes them."""
        return self._emb[row], self._levels[row]

    def similar_to(self, emb, rms_db, k=3, exclude_rows=()):
        """The k best-scoring catalog songs for a song's vectors (stem_analysis
        output, or vectors()), best first. Each: track_id (Spotify), track_name,
        track_artist, score (weighted sum) and channels ({channel: z-scored
        similarity}, the terms of the sum before weighting)."""
        z = self._channel_scores(np.asarray(emb, np.float32), np.asarray(rms_db) > self.silent_db)
        total = z @ self.weights
        total[list(exclude_rows)] = -np.inf
        k = min(k, len(total) - len(exclude_rows))
        if k <= 0:
            return []
        top = np.argpartition(-total, k - 1)[:k]
        top = top[np.argsort(-total[top])]
        return [
            {
                "track_id": self.tracks["spotify_track_id"].iat[i],
                "track_name": self.tracks["title"].iat[i],
                "track_artist": self.tracks["artist"].iat[i],
                "score": float(total[i]),
                "channels": {c: float(z[i, j]) for j, c in enumerate(CHANNELS)},
            }
            for i in top
        ]

    def _channel_scores(self, emb, audible):
        """(songs, channels) z-scored cosine of every catalog song to `emb`, silence-gated."""
        query = emb / np.linalg.norm(emb, axis=1, keepdims=True).clip(1e-9)
        cos = np.einsum("scd,cd->sc", self._emb, query)
        z = (cos - self.z_mean) / self.z_std
        catalog_audible = self._levels > self.silent_db
        return np.where(catalog_audible & audible, z, np.where(catalog_audible ^ audible, -1.0, 0.0))
