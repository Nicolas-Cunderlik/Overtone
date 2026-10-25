"""Worker thread to fetch album image bytes, similar tracks, and AI chord suggestions.
Emits raw bytes for the UI thread to convert to QPixmap (QPixmap isn't thread-safe everywhere).
Similar tracks (a local lookup) run before the AI call, which can take several
seconds with retries, so they aren't held up behind it.
"""

import logging

import requests
from PyQt5.QtCore import QThread, pyqtSignal

logger = logging.getLogger(__name__)


class FetchDataWorker(QThread):
    album_bytes = pyqtSignal(bytes)
    ai_ready = pyqtSignal(object)  # ChordSuggestions, or None if unavailable
    similar_ready = pyqtSignal(list)
    error = pyqtSignal(str)

    def __init__(self, spotify_track, ai_service, similarity_service, parent=None):
        super().__init__(parent)
        self.track = spotify_track
        self.ai_service = ai_service
        self.similarity_service = similarity_service

    def run(self):
        try:
            if not self.track:
                return

            # Album art
            try:
                album_cover_url = self.track['album']['images'][0]['url']
                resp = requests.get(album_cover_url, timeout=8)
                resp.raise_for_status()
                self.album_bytes.emit(resp.content)
            except Exception:
                logger.exception("Failed to fetch album image")

            # Similar tracks
            try:
                if self.similarity_service:
                    track_id = self.track.get('id')
                    similar = self.similarity_service.find_similar(track_id, k=3)
                    self.similar_ready.emit(similar)
            except Exception:
                logger.exception("Failed to fetch similar tracks")

            # AI suggestions
            try:
                if self.ai_service:
                    track_name = self.track.get('name', 'Unknown')
                    artist = ", ".join(a.get('name') for a in self.track.get('artists', []))
                    suggestions = self.ai_service.get_chord_suggestions(self.track.get('id'), track_name, artist)
                    self.ai_ready.emit(suggestions)
            except Exception:
                logger.exception("Failed to fetch AI suggestions")
                self.ai_ready.emit(None)

        except Exception as e:
            logger.exception("Unhandled error in FetchDataWorker")
            self.error.emit(str(e))
