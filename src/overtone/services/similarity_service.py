"""Similarity service wrapper around the offline-trained song-similarity model.

The model is loaded once, lazily, on first use. If artifacts are missing
(train.py hasn't been run) or loading fails for any reason, lookups just
return an empty list instead of raising.
"""

import logging
import threading

logger = logging.getLogger(__name__)


class SimilarityService:
    def __init__(self):
        self._model = None
        self._load_failed = False
        # A superseded FetchDataWorker can still be running when the next one
        # starts, so two threads may hit the first lookup at once - load once
        self._load_lock = threading.Lock()

    def _ensure_model(self):
        with self._load_lock:
            if self._model is None and not self._load_failed:
                try:
                    from ..ml.similarity_model import SimilarityModel

                    self._model = SimilarityModel()
                except Exception:
                    logger.exception("Failed to load similarity model")
                    self._load_failed = True
        return self._model

    def find_similar(self, track_id, k=3):
        model = self._ensure_model()
        if model is None:
            return []
        try:
            return model.find_similar(track_id, k=k)
        except Exception:
            logger.exception("Similarity lookup failed for %s", track_id)
            return []

    def has_track(self, track_id):
        model = self._ensure_model()
        return model is not None and model.has_track(track_id)

    def find_similar_to_features(self, audio_features, k=3, track_name=None, track_artist=None):
        """For songs outside the catalog - see SimilarityModel.find_similar_to_features."""
        model = self._ensure_model()
        if model is None:
            return []
        try:
            return model.find_similar_to_features(audio_features, k=k, track_name=track_name, track_artist=track_artist)
        except Exception:
            logger.exception("Similarity lookup by features failed for '%s'", track_name)
            return []
