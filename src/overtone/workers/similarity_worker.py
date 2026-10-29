"""Worker thread that finds similar tracks for the playing song. Separate from
FetchDataWorker because analysing a song outside the catalog can take a
while, and the album art and chord suggestions shouldn't wait for it."""

import logging

from PyQt5.QtCore import QThread, pyqtSignal

logger = logging.getLogger(__name__)


class SimilarTracksWorker(QThread):
    analyzing = pyqtSignal(bool)  # analysis started (song not in the catalog); True if it also loads the models
    similar_ready = pyqtSignal(object)  # list of results, or None if unavailable

    def __init__(self, spotify_track, similarity_service, parent=None):
        super().__init__(parent)
        self.track = spotify_track
        self.similarity_service = similarity_service

    def run(self):
        try:
            similar = self.similarity_service.find_similar(
                self.track,
                k=3,
                on_analyzing=self.analyzing.emit,
                cancelled=self.isInterruptionRequested,
            )
        except Exception:  # the service catches its own failures; this guards the thread
            logger.exception("Unhandled error in SimilarTracksWorker")
            similar = None
        if not self.isInterruptionRequested():
            self.similar_ready.emit(similar)
