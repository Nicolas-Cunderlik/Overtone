"""Similar songs for the playing track, from the stem-similarity model.

A catalog song is scored from its stored vectors (milliseconds). Any other
song is analysed from its Deezer preview - never the audio playing on the
user's device - with the same code that built the catalog
(overtone_core.ml.stem_analysis), then cached by Spotify track id:

    catalog? -> cache? -> Deezer preview (ISRC, else title search) -> stems + embeddings -> cache -> score

Every failure is caught and logged with its detail; callers get None and
show a generic message. The stem model, Demucs and MuQ each load once, on
first use; analyses run one at a time because they share those models.
"""

import logging
import os
import tempfile
import threading
import time

from .stem_cache import StemCache

logger = logging.getLogger(__name__)


class SimilarityService:
    def __init__(self, model_factory=None, deezer=None, analyse=None, cache=None):
        """Each None means the real one: StemModel, DeezerClient,
        stem_analysis on the MP3 bytes, StemCache()."""
        self._model_factory = model_factory
        self._deezer = deezer
        self._analyse = analyse
        self._cache = cache or StemCache()
        self._model = None
        self._load_failed = False
        self._analysed_one = False  # the first analysis also loads Demucs and MuQ
        # A superseded worker can still be running when the next one starts,
        # so two threads may reach the first lookup or an analysis at once
        self._load_lock = threading.Lock()
        self._analysis_lock = threading.Lock()

    def find_similar(self, track, k=3, on_analyzing=None, cancelled=None):
        """The k most similar catalog songs to a Spotify track object (the
        playback item), best first, each with per-channel scores (see
        StemModel.similar_to); None on any failure or if cancelled.

        on_analyzing(first) is called before a song outside the catalog is
        analysed (seconds on a GPU, longer on CPU); `first` is True until an
        analysis has succeeded, as that one also loads the models (~1 min).
        cancelled() is polled between the slow steps so a superseded request
        stops early."""
        cancelled = cancelled or (lambda: False)
        track_id, title = track.get("id"), track.get("name")
        artist = next((a.get("name") for a in track.get("artists") or []), None)
        try:
            model = self._ensure_model()
            if model is None:
                return None
            row = model.find(track_id, title, artist)
            if row is not None:
                results = model.similar_to(*model.vectors(row), k=k, exclude_rows=[row])
                source = "catalog"
            else:
                vectors = self._cache.get(track_id) if track_id else None
                source = "cache"
                if vectors is None:
                    if on_analyzing:
                        on_analyzing(not self._analysed_one)
                    vectors = self._analyse_preview(track, title, artist, cancelled)
                    source = "analysis"
                if vectors is None:
                    return None
                results = model.similar_to(*vectors, k=k)
        except Exception:
            logger.exception("Similar tracks failed for %s (%s - %s)", track_id, title, artist)
            return None
        logger.info(
            "Similar to %s - %s (%s): %s",
            title,
            artist,
            source,
            "; ".join(
                f"{r['track_name']} - {r['track_artist']} {r['score']:.2f} "
                + " ".join(f"{c}={v:.2f}" for c, v in r["channels"].items())
                for r in results
            ),
        )
        return results

    def _analyse_preview(self, track, title, artist, cancelled):
        """(emb, rms_db) of the track's Deezer preview, cached; None if there
        is no preview or the request was cancelled."""
        deezer = self._deezer_client()
        isrc = (track.get("external_ids") or {}).get("isrc")
        duration_s = track["duration_ms"] / 1000 if track.get("duration_ms") else None
        match = deezer.find_preview(title, artist, isrc, duration_s)
        if match is None:
            logger.info("No Deezer preview for %s - %s (isrc %s)", title, artist, isrc)
            return None
        hit, matched_by = match
        logger.debug("Downloading Deezer preview %s (matched by %s)", hit.get("id"), matched_by)
        audio = deezer.preview_bytes(hit)
        if cancelled():
            return None
        with self._analysis_lock:
            if cancelled():
                return None
            started = time.monotonic()
            emb, rms_db = self._analyse_audio(audio)
            self._analysed_one = True
        logger.info(
            "Analysed %s - %s from Deezer %s (matched by %s) in %.1f s",
            title,
            artist,
            hit.get("id"),
            matched_by,
            time.monotonic() - started,
        )
        if track.get("id"):
            self._cache.put(track["id"], emb, rms_db)
        return emb, rms_db

    def _analyse_audio(self, audio):
        """(emb, rms_db) of the full audio, from MP3 bytes."""
        if self._analyse is not None:
            return self._analyse(audio)
        from overtone_core.ml import stem_analysis  # heavy import; only once a song needs analysing

        # libsndfile can't detect MP3 in a memory buffer, so decode from a
        # file, as the pipeline does; deleted right after (previews are never kept)
        with tempfile.TemporaryDirectory(prefix="overtone-preview-") as folder:
            path = os.path.join(folder, "preview.mp3")
            with open(path, "wb") as f:
                f.write(audio)
            wav, sr = stem_analysis.load_audio(path)
        result = stem_analysis.analyse(wav, sr)
        return result.emb[0], result.rms_db[0]

    def _deezer_client(self):
        if self._deezer is None:
            from overtone_core.clients.deezer import DeezerClient

            self._deezer = DeezerClient()
        return self._deezer

    def _ensure_model(self):
        with self._load_lock:
            if self._model is None and not self._load_failed:
                try:
                    if self._model_factory is None:
                        from overtone_core.ml.stem_model import StemModel

                        self._model_factory = StemModel
                    self._model = self._model_factory()
                except Exception:
                    logger.exception("Failed to load the stem model")
                    self._load_failed = True
        return self._model
