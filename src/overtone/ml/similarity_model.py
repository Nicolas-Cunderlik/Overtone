"""Loads the artifacts produced by train.py and serves nearest-neighbor lookups.

Two ways in:
  - find_similar(track_id): a song in the static training catalog, looked up
    by Spotify track ID (or any deduplicated alias of it).
  - find_similar_to_features(audio_features): any other song, given its audio
    features in Spotify's legacy audio-features format (e.g. from FreqBlog -
    Spotify's own endpoint is restricted for this app). The features are
    embedded with the same scaler and genre classifier the catalog was.
Results always come from the catalog. An empty result means "not enough data",
not "no similar tracks exist".

The catalog is deduplicated at training time (dedup.py). Lookups re-check
results with the same rules as a redundant guard, so a copy that slips through
is still never shown - but it's logged as a leak, since it means the training
dedup missed something and should be fixed there.

Neighbors are searched in the learned genre embedding (embedding.py), while the
duplicate check uses the plain standardized features its distance threshold
was measured in.
"""

import logging
import os

import joblib
import pandas as pd

from . import dedup, embedding, features
from .paths import (
    ARTIFACT_DIR,
    CLASSIFIER_FILE,
    EMBEDDING_FILE,
    FEATURES_FILE,
    INDEX_FILE,
    SCALER_FILE,
    TRACK_META_FILE,
    VARIANT_FILE,
)

logger = logging.getLogger(__name__)

# Extra neighbors fetched beyond k so the duplicate filter can drop some and
# still return k results. Leaks should be rare, so a small margin suffices.
DUPLICATE_MARGIN = 5


class SimilarityModel:
    def __init__(self, artifact_dir=ARTIFACT_DIR):
        def load(name):
            return joblib.load(os.path.join(artifact_dir, name))

        self.index = load(INDEX_FILE)
        self.X = load(FEATURES_FILE)
        self.Z = load(EMBEDDING_FILE)
        self.scaler = load(SCALER_FILE)
        self.variant = load(VARIANT_FILE)
        self.classifier = load(CLASSIFIER_FILE)  # None if trained without the embedding
        meta = pd.read_csv(os.path.join(artifact_dir, TRACK_META_FILE))
        self.track_ids = meta["track_id"].tolist()
        self.track_names = meta["track_name"].tolist()
        self.track_artists = meta["track_artist"].tolist()
        self._row_by_id = {track_id: i for i, track_id in enumerate(self.track_ids)}
        # Every copy of a deduplicated song resolves to its canonical row
        for i, aliases in enumerate(meta["alias_track_ids"]):
            for alias in str(aliases).split(";"):
                self._row_by_id.setdefault(alias, i)

    def has_track(self, track_id):
        return track_id in self._row_by_id

    def find_similar(self, track_id, k=3):
        row = self._row_by_id.get(track_id)
        if row is None:
            return []
        return self._search(self.Z[row : row + 1], k, exclude_row=row)

    def find_similar_to_features(self, audio_features, k=3, track_name=None, track_artist=None):
        """`audio_features`: dict with every column in features.RAW_AUDIO_COLUMNS.
        With `track_name` and `track_artist`, catalog copies of the song itself
        (same song under another track ID) are left out of the results."""
        frame = pd.DataFrame([audio_features], columns=features.RAW_AUDIO_COLUMNS)
        x, _ = features.build_feature_matrix(frame, variant=self.variant, scaler=self.scaler)
        z = embedding.transform(x, self.classifier) if self.classifier is not None else x
        query = (track_name, track_artist, x[0]) if track_name and track_artist else None
        return self._search(z, k, query=query)

    def _search(self, z, k, exclude_row=None, query=None):
        """The k catalog songs nearest to embedding row `z`, skipping
        `exclude_row` (a catalog query, always its own nearest neighbor),
        copies of `query` (title, artist, features) and copies of each other."""
        n_neighbors = min(k + 1 + DUPLICATE_MARGIN, len(self.track_ids))
        distances, indices = self.index.kneighbors(z, n_neighbors=n_neighbors)

        picked_rows = []
        results = []
        for dist, idx in zip(distances[0], indices[0], strict=True):
            if idx == exclude_row:
                continue
            if query is not None and dedup.is_duplicate(
                query[0], query[1], self.track_names[idx], self.track_artists[idx], query[2], self.X[idx]
            ):
                logger.debug(
                    "Skipping '%s' by %s (%s): a catalog copy of the query song",
                    self.track_names[idx],
                    self.track_artists[idx],
                    self.track_ids[idx],
                )
                continue
            others = picked_rows if exclude_row is None else [exclude_row] + picked_rows
            duplicate_of = next((other for other in others if self._is_duplicate(idx, other)), None)
            if duplicate_of is not None:
                logger.warning(
                    "Dedup leak: '%s' by %s (%s) duplicates '%s' by %s (%s) - filtered at "
                    "lookup, but the training catalog should have merged them",
                    self.track_names[idx],
                    self.track_artists[idx],
                    self.track_ids[idx],
                    self.track_names[duplicate_of],
                    self.track_artists[duplicate_of],
                    self.track_ids[duplicate_of],
                )
                continue
            picked_rows.append(idx)
            results.append(
                {
                    "track_id": self.track_ids[idx],
                    "track_name": self.track_names[idx],
                    "track_artist": self.track_artists[idx],
                    "similarity": float(1 - dist),
                }
            )
            if len(results) == k:
                break
        return results

    def _is_duplicate(self, row_a, row_b):
        return dedup.is_duplicate(
            self.track_names[row_a],
            self.track_artists[row_a],
            self.track_names[row_b],
            self.track_artists[row_b],
            self.X[row_a],
            self.X[row_b],
        )
